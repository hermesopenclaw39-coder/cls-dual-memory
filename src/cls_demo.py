# -*- coding: utf-8 -*-
"""
CLS 双记忆系统迷你实验（海马体 + 皮层）—— 纯 CPU 版
====================================================
对应脑结构 -> 实现：
  皮层（长期知识）   = 冻结的预训练底座（Qwen2.5-0.5B）
  海马体（临时情景） = 每条情景一个专属 LoRA 适配器（彼此隔离）
  门控（回忆/检索）  = 按嵌入余弦相似度选择用哪个适配器
  缓慢巩固           = 回放所有情景 -> 训练语义适配器 -> 合并进底座权重

对照实验：
  单个共享 LoRA 顺序学完所有情景 -> 灾难性遗忘（前面学的被覆盖）

用法：
  python cls_demo.py
（可改下面的 RANK / STEPS / LR 调节快慢与效果）
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import gc
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

# 随机种子：可用环境变量 SEED 覆盖。注意 bitsandbytes / CUDA 部分算子本身不确定，
# 固定种子能显著减小波动，但不保证逐位一致。
from transformers import set_seed
SEED = int(os.environ.get("SEED", "42"))
set_seed(SEED)

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

# ---- 可调参数 ----
RANK = 8                  # LoRA 秩
STEPS_PER_EPISODE = 40    # 每条情景（海马体）训练步数
LR_FAST = 2e-3            # 海马体快速学习率
CONSOLIDATE_STEPS = 40    # 巩固回放轮数（每轮过一遍所有情景）
LR_SLOW = 5e-4            # 巩固（皮层）慢速学习率
MAX_NEW = 12              # 生成的最大新 token 数

# 每个情景：(训练文本, 测试前缀, 期望答案片段)
# 全部是虚构事实，预训练模型不该知道，用来证明"新知识"。
from facts import FACTS_6
EPISODES = FACTS_6  # 事实集合见 src/facts.py

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def lora_cfg(rank=RANK):
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.0,
        target_modules=LORA_TARGETS,
    )


def load_base():
    print("  加载底座模型 ...", flush=True)
    t = time.time()
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    m.eval()
    print(f"  (底座加载耗时 {time.time() - t:.1f}s)", flush=True)
    return m


def mean_embed(model, text):
    """用原始底座算文本的平均 last-hidden-state（用于门控的 key/query）。"""
    inp = {k: v.to(DEVICE) for k, v in tokenizer(text, return_tensors="pt").items()}
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    h = out.hidden_states[-1]                      # (1, L, D)
    mask = inp["attention_mask"].unsqueeze(-1).float()
    return (h * mask).sum(1) / mask.sum(1)         # (1, D)


def cos(a, b):
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-9)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-9)
    return (a * b).sum(-1)


def train_on_text(model, text, steps, lr):
    model.train()
    inp = {k: v.to(DEVICE) for k, v in tokenizer(text, return_tensors="pt").items()}
    ids = inp["input_ids"]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
    return float(loss.detach())


def train_cycle(model, texts, epochs, lr):
    """轮流在每个事实上各训练一步（共享优化器），共 epochs 轮。
    用于巩固：让每个事实独立可召回，而不是拼成一长串。"""
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    for _ in range(epochs):
        for text in texts:
            inp = {k: v.to(DEVICE) for k, v in tokenizer(text, return_tensors="pt").items()}
            ids = inp["input_ids"]
            opt.zero_grad()
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            opt.step()
    return float(loss.detach())


def generate(model, prompt):
    model.eval()
    inp = {k: v.to(DEVICE) for k, v in tokenizer(prompt, return_tensors="pt").items()}
    with torch.no_grad():
        out = model.generate(
            **inp, max_new_tokens=MAX_NEW, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def eval_recall(model, episodes):
    """用每个情景的测试前缀生成并核对答案；返回 (正确数, 明细)。"""
    ok, detail = 0, []
    for i, (_train, q, ans) in enumerate(episodes):
        got = generate(model, q).lower()
        hit = ans.lower() in got
        ok += hit
        detail.append((i, hit, got.strip()))
    return ok, detail


def eval_gated(model, adapter_names, keys, queries, episodes):
    """门控召回：先按相似度路由到某个适配器，再用它生成。"""
    ok, detail = 0, []
    for i, (_train, q, ans) in enumerate(episodes):
        sims = torch.stack([cos(queries[i], k) for k in keys]).squeeze(-1)
        route = int(sims.argmax())
        model.set_adapter(adapter_names[route])
        got = generate(model, q).lower()
        hit = ans.lower() in got
        ok += hit
        detail.append((i, route, hit, got.strip()))
    return ok, detail


def main():
    t0 = time.time()
    n = len(EPISODES)
    print("=" * 60, flush=True)
    print("CLS 双记忆系统实验（海马体 + 皮层）", flush=True)
    print(f"  设备: {DEVICE}   dtype: {DTYPE}", flush=True)
    print("=" * 60, flush=True)

    # ---- Step 0: 冻结底座对新事实的初始召回 ----
    print("\n[0/4] 冻结底座（皮层）对新事实的初始召回 ...", flush=True)
    base = load_base()
    keys = [mean_embed(base, tr) for tr, _, _ in EPISODES]   # 情景 key
    queries = [mean_embed(base, q) for _, q, _ in EPISODES]  # 测试 query
    ok0, det0 = eval_recall(base, EPISODES)
    print(f"  冻结底座召回: {ok0}/{n}  （应该很低 = 证明这些是新知识）", flush=True)
    for i, hit, got in det0:
        print(f"    情景{i}: {'✓' if hit else '✗'} {got!r}", flush=True)

    # ---- Step 1: 快速学习（海马体：每条情景一个专属 LoRA）----
    print("\n[1/4] 快速学习：每条情景训练一个专属 LoRA（海马体）...", flush=True)
    model = get_peft_model(base, lora_cfg())                 # 'default' 作为情景0
    adapter_names = ["default"] + [f"ep{i}" for i in range(1, n)]
    for i in range(1, n):
        model.add_adapter(adapter_names[i], lora_cfg())
    for i, (train_text, _, _) in enumerate(EPISODES):
        name = adapter_names[i]
        model.set_adapter(name)
        t = time.time()
        loss = train_on_text(model, train_text, STEPS_PER_EPISODE, LR_FAST)
        print(f"  情景{i} 适配器[{name}] 训练完, loss={loss:.4f} ({time.time() - t:.1f}s)", flush=True)
    ok1, det1 = eval_gated(model, adapter_names, keys, queries, EPISODES)
    print(f"  海马体+门控 召回: {ok1}/{n}", flush=True)
    for i, route, hit, got in det1:
        print(f"    情景{i}: 路由→适配器[{adapter_names[route]}] {'✓' if hit else '✗'} {got!r}", flush=True)

    del model
    gc.collect()

    # ---- Step 2: 对照 —— 单个共享 LoRA 顺序学习（灾难性遗忘）----
    print("\n[2/4] 对照：单个共享 LoRA 顺序学完所有情景 ...", flush=True)
    base2 = load_base()
    shared = get_peft_model(base2, lora_cfg())
    t = time.time()
    for train_text, _, _ in EPISODES:
        train_on_text(shared, train_text, STEPS_PER_EPISODE, LR_FAST)
    print(f"  共享 LoRA 训练完 ({time.time() - t:.1f}s)", flush=True)
    ok2, det2 = eval_recall(shared, EPISODES)
    print(f"  共享 LoRA 召回: {ok2}/{n}  （期望更低 = 遗忘）", flush=True)
    for i, hit, got in det2:
        print(f"    情景{i}: {'✓' if hit else '✗'} {got!r}", flush=True)

    del shared, base2
    gc.collect()

    # ---- Step 3: 缓慢巩固（回放 -> 语义适配器 -> 合并进底座）----
    print("\n[3/4] 缓慢巩固：回放所有情景，训练语义适配器并合并进底座 ...", flush=True)
    base3 = load_base()
    cons = get_peft_model(base3, lora_cfg())
    t = time.time()
    loss = train_cycle(cons, [tr for tr, _, _ in EPISODES], CONSOLIDATE_STEPS, LR_SLOW)
    print(f"  巩固训练完, loss={loss:.4f} ({time.time() - t:.1f}s)", flush=True)
    merged = cons.merge_and_unload()                        # 语义适配器写进底座 = 皮层
    ok3, det3 = eval_recall(merged, EPISODES)
    print(f"  巩固后底座（无适配器）召回: {ok3}/{n}  （期望高于冻结底座）", flush=True)
    for i, hit, got in det3:
        print(f"    情景{i}: {'✓' if hit else '✗'} {got!r}", flush=True)

    # ---- 总结 ----
    print("\n" + "=" * 60, flush=True)
    print("总结", flush=True)
    print("=" * 60, flush=True)
    print(f"  冻结底座（初始）         : {ok0}/{n}   <- 新知识，不会", flush=True)
    print(f"  海马体（逐条LoRA+门控）  : {ok1}/{n}   <- 快速学到、互不干扰", flush=True)
    print(f"  共享LoRA（对照/遗忘）    : {ok2}/{n}   <- 顺序学习互相覆盖", flush=True)
    print(f"  巩固后底座（皮层）       : {ok3}/{n}   <- 知识搬进长期记忆", flush=True)
    print(f"\n总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
