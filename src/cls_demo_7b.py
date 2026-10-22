# -*- coding: utf-8 -*-
"""
CLS 双记忆系统实验 @ Qwen2.5-7B（4-bit QLoRA）
=================================================
与 0.5B 版区别：
  - 7B fp16 ~14GB > 12GB 显存，故用 4-bit nf4 量化加载（QLoRA）
  - 4-bit 无法 merge 回底座，故"巩固"改为训练一个共享的"语义适配器"（回放所有情景）
    ，对应论文里"情景适配器(海马体) vs 语义适配器(皮层)"的分离。
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import gc
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

# 随机种子：可用环境变量 SEED 覆盖。注意 bitsandbytes / CUDA 部分算子本身不确定，
# 固定种子能显著减小波动，但不保证逐位一致。
from transformers import set_seed
SEED = int(os.environ.get("SEED", "42"))
set_seed(SEED)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

RANK = 8
STEPS_PER_EPISODE = 150
LR_FAST = 1e-3
CONSOLIDATE_STEPS = 150   # 巩固回放轮数
LR_SLOW = 5e-4            # 巩固（皮层）慢速学习率（1e-3 会导致共享适配器内部互相覆盖）
MAX_NEW = 12

from facts import FACTS_6
EPISODES = FACTS_6  # 事实集合见 src/facts.py

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def bnb_cfg():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
        bnb_4bit_use_double_quant=True,
    )


def lora_cfg(rank=RANK):
    return LoraConfig(task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=rank * 2,
                      lora_dropout=0.0, target_modules=LORA_TARGETS)


def load_base():
    t = time.time()
    m = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, quantization_config=bnb_cfg(), device_map="auto")
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)
    m.eval()
    print(f"  (加载 7B 4-bit {time.time() - t:.1f}s)", flush=True)
    return m


def mean_embed(model, text):
    inp = {k: v.to(DEVICE) for k, v in tokenizer(text, return_tensors="pt").items()}
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    h = out.hidden_states[-1]
    mask = inp["attention_mask"].unsqueeze(-1).float()
    return (h * mask).sum(1) / mask.sum(1)


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
        out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def eval_recall(model, episodes):
    ok, detail = 0, []
    for i, (_t, q, ans) in enumerate(episodes):
        got = generate(model, q).lower()
        hit = ans.lower() in got
        ok += hit
        detail.append((i, hit, got.strip()))
    return ok, detail


def eval_gated(model, adapter_names, keys, queries, episodes):
    ok, detail = 0, []
    for i, (_t, q, ans) in enumerate(episodes):
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
    print("CLS 双记忆系统 @ Qwen2.5-7B（4-bit QLoRA）", flush=True)
    print("=" * 60, flush=True)

    # Step 0
    print("\n[0/4] 冻结底座对新事实的初始召回 ...", flush=True)
    base = load_base()
    keys = [mean_embed(base, tr) for tr, _, _ in EPISODES]
    queries = [mean_embed(base, q) for _, q, _ in EPISODES]
    ok0, det0 = eval_recall(base, EPISODES)
    print(f"  冻结底座召回: {ok0}/{n}", flush=True)
    for i, hit, got in det0:
        print(f"    情景{i}: {'✓' if hit else '✗'} {got!r}", flush=True)

    # Step 1
    print("\n[1/4] 快速学习：每条情景一个专属 LoRA（海马体）...", flush=True)
    model = get_peft_model(base, lora_cfg())
    adapter_names = ["default"] + [f"ep{i}" for i in range(1, n)]
    for i in range(1, n):
        model.add_adapter(adapter_names[i], lora_cfg())
    for i, (tr, _, _) in enumerate(EPISODES):
        name = adapter_names[i]
        model.set_adapter(name)
        t = time.time()
        loss = train_on_text(model, tr, STEPS_PER_EPISODE, LR_FAST)
        print(f"  情景{i} 适配器[{name}] loss={loss:.4f} ({time.time() - t:.1f}s)", flush=True)
    ok1, det1 = eval_gated(model, adapter_names, keys, queries, EPISODES)
    print(f"  海马体+门控 召回: {ok1}/{n}", flush=True)
    for i, route, hit, got in det1:
        print(f"    情景{i}: 路由→[{adapter_names[route]}] {'✓' if hit else '✗'} {got!r}", flush=True)
    del model, base
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2
    print("\n[2/4] 对照：单个共享 LoRA 顺序学习（灾难性遗忘）...", flush=True)
    base2 = load_base()
    shared = get_peft_model(base2, lora_cfg())
    t = time.time()
    for tr, _, _ in EPISODES:
        train_on_text(shared, tr, STEPS_PER_EPISODE, LR_FAST)
    print(f"  共享 LoRA 训练完 ({time.time() - t:.1f}s)", flush=True)
    ok2, det2 = eval_recall(shared, EPISODES)
    print(f"  共享 LoRA 召回: {ok2}/{n}", flush=True)
    for i, hit, got in det2:
        print(f"    情景{i}: {'✓' if hit else '✗'} {got!r}", flush=True)
    del shared, base2
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: 巩固 -> 语义适配器（4-bit 无法 merge，保持为独立的共享语义适配器）
    print("\n[3/4] 缓慢巩固：回放所有情景，训练语义适配器 ...", flush=True)
    base3 = load_base()
    cons = get_peft_model(base3, lora_cfg())   # 语义适配器（rank 8 即可装下 6 条事实）
    t = time.time()
    loss = train_cycle(cons, [tr for tr, _, _ in EPISODES], CONSOLIDATE_STEPS, LR_SLOW)
    print(f"  巩固训练完, loss={loss:.4f} ({time.time() - t:.1f}s)", flush=True)
    ok3, det3 = eval_recall(cons, EPISODES)   # 底座 + 语义适配器（无逐条适配器）
    print(f"  巩固后（底座+语义适配器）召回: {ok3}/{n}", flush=True)
    for i, hit, got in det3:
        print(f"    情景{i}: {'✓' if hit else '✗'} {got!r}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("总结", flush=True)
    print("=" * 60, flush=True)
    print(f"  冻结底座（初始）       : {ok0}/{n}", flush=True)
    print(f"  海马体（逐条LoRA+门控）: {ok1}/{n}", flush=True)
    print(f"  共享LoRA（对照/遗忘）  : {ok2}/{n}", flush=True)
    print(f"  巩固后（语义适配器）   : {ok3}/{n}", flush=True)
    print(f"\n总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
