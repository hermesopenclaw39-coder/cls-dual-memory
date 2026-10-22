# -*- coding: utf-8 -*-
"""
CLS 巩固阶段 —— 多适配器分组版
==============================
验证：单共享语义适配器在 50 条上失败（0/50，loss 4~5 震荡），
换成「分组 + 每组一个适配器」能否恢复。

分组按索引等分（50 条事实本就随机排列，等价于随机分组）。
每组用 rank-8 适配器回放巩固，复用 6 条版验证过的配置（LR_SLOW=5e-4、150 epoch）。

可调：GROUP_SIZE / RANK / EPOCHS / LR_SLOW。
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

GROUP_SIZE = 6     # 每组事实数（6 条版已验证 rank8 能干净装下 6 条）
RANK = 8           # 每组适配器的秩
EPOCHS = 150       # 回放轮数
LR_SLOW = 5e-4     # 巩固慢速学习率（1e-3 会内部互相覆盖）
MAX_NEW = 12

# (训练文本, 测试前缀, 期望答案片段) —— 与 cls_demo_7b_50.py 完全一致的 50 条
from facts import FACTS_50
EPISODES = FACTS_50  # 事实集合见 src/facts.py

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
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb_cfg(), device_map="auto")
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)
    m.eval()
    print(f"  (加载 7B 4-bit {time.time() - t:.1f}s)", flush=True)
    return m


def train_cycle(model, texts, epochs, lr):
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    for e in range(epochs):
        for text in texts:
            inp = {k: v.to(DEVICE) for k, v in tokenizer(text, return_tensors="pt").items()}
            ids = inp["input_ids"]
            opt.zero_grad()
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            opt.step()
        if (e + 1) % 50 == 0 or e == 0:
            print(f"    epoch {e + 1}/{epochs}: loss={float(loss):.4f}", flush=True)
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


def main():
    t0 = time.time()
    n = len(EPISODES)
    groups = [EPISODES[i:i + GROUP_SIZE] for i in range(0, n, GROUP_SIZE)]
    G = len(groups)
    print("=" * 60, flush=True)
    print(f"CLS 多适配器分组巩固 @ Qwen2.5-7B（4-bit）—— {n} 条 / {G} 组 / 每组 {GROUP_SIZE} 条 / rank {RANK}", flush=True)
    print("=" * 60, flush=True)

    base = load_base()
    model = get_peft_model(base, lora_cfg(RANK))
    adapter_names = ["default"] + [f"grp{g}" for g in range(1, G)]
    for g in range(1, G):
        model.add_adapter(adapter_names[g], lora_cfg(RANK))

    # 每组用专属适配器回放本组事实
    for g, grp in enumerate(groups):
        name = adapter_names[g]
        model.set_adapter(name)
        t = time.time()
        loss = train_cycle(model, [tr for tr, _, _ in grp], EPOCHS, LR_SLOW)
        print(f"  组{g} ({len(grp)}条) 适配器[{name}] loss={loss:.4f} ({time.time() - t:.0f}s)", flush=True)

    # 评测：每条事实用自己组的适配器
    print("\n[评测] 每条事实用所属组的适配器 ...", flush=True)
    ok_total, misses = 0, []
    for g, grp in enumerate(groups):
        model.set_adapter(adapter_names[g])
        okg, detg = eval_recall(model, grp)
        ok_total += okg
        miss_idx = [i for i, hit, _g in detg if not hit]
        for i in miss_idx:
            misses.append(g * GROUP_SIZE + i)
        print(f"  组{g}: {okg}/{len(grp)}   (miss 事实索引: {[g*GROUP_SIZE+i for i in miss_idx]})", flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"  分组巩固总召回: {ok_total}/{n}   (失败索引: {sorted(misses)})", flush=True)
    print(f"  总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
