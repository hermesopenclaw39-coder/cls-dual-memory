# -*- coding: utf-8 -*-
"""
测试：巩固新记忆时，会不会反过来污染皮层里已巩固的旧记忆？
（互补学习系统 CLS 的核心难题）

流程：
  1. 先把旧知识 1-6 巩固进底座（皮层）
  2a. 天真巩固：再只练新知识 7-9（不回放 1-6）-> 看 1-6 是否被忘掉
  2b. 回放巩固：练 7-9 时同时回放 1-6        -> 看 1-6 是否保住
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

RANK = 8
LR_SLOW = 5e-4
EPOCHS_PHASE1 = 40   # 第一次巩固（旧知识）的轮数
EPOCHS_NAIVE = 40    # 天真巩固：只练新知识、不回放的轮数
EPOCHS_REPLAY = 20   # 回放巩固：新+旧一起练的轮数
MAX_NEW = 12

# 9 条事实：前 6 条"旧"，后 3 条"新"。(训练文本, 测试前缀, 期望答案片段)
from facts import FACTS_9
FACTS = FACTS_9  # 事实集合见 src/facts.py

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


def lora_cfg(rank=RANK):
    return LoraConfig(task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=rank * 2,
                      lora_dropout=0.0, target_modules=LORA_TARGETS)


def load_base():
    t = time.time()
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    m.eval()
    print(f"  (加载 {time.time() - t:.1f}s)", flush=True)
    return m


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


def consolidate(model, facts, epochs, lr):
    """在 model 上训练一个语义 LoRA（轮询这些事实的训练文本）并合并进权重，返回合并后的模型。"""
    texts = [tr for tr, _, _ in facts]
    pm = get_peft_model(model, lora_cfg())
    t = time.time()
    loss = train_cycle(pm, texts, epochs, lr)
    merged = pm.merge_and_unload()
    print(f"  巩固 {len(texts)} 条, loss={loss:.4f} ({time.time() - t:.1f}s)", flush=True)
    return merged


def generate(model, prompt):
    model.eval()
    inp = {k: v.to(DEVICE) for k, v in tokenizer(prompt, return_tensors="pt").items()}
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


def show(tag, model, facts):
    ok, detail = 0, []
    for i, (_t, q, ans) in enumerate(facts):
        got = generate(model, q).lower()
        hit = ans.lower() in got
        ok += hit
        detail.append((i, hit, got.strip()))
    print(f"  {tag}: {ok}/{len(facts)}", flush=True)
    for i, hit, got in detail:
        print(f"    事实{i}: {'✓' if hit else '✗'} {got!r}", flush=True)
    return ok


def main():
    t0 = time.time()
    old = FACTS[:6]
    new = FACTS[6:]

    print("=" * 60, flush=True)
    print("巩固是否会污染旧知识？（CLS 核心难题）", flush=True)
    print(f"设备: {DEVICE}  dtype: {DTYPE}", flush=True)
    print("=" * 60, flush=True)

    # ---- 场景 A：天真巩固（不重放旧知识）----
    print("\n[A] 天真巩固：先巩固 1-6，再只练 7-9（不回放 1-6）", flush=True)
    base_a = load_base()
    cortex_a = consolidate(base_a, old, EPOCHS_PHASE1, LR_SLOW)
    show("  巩固旧知识后 旧(1-6)", cortex_a, old)
    cortex_a = consolidate(cortex_a, new, EPOCHS_NAIVE, LR_SLOW)
    ok_old_a = show("  再巩固新知识后 旧(1-6)", cortex_a, old)
    ok_new_a = show("  新(7-9)", cortex_a, new)
    del cortex_a, base_a
    gc.collect()

    # ---- 场景 B：回放巩固（重放旧知识）----
    print("\n[B] 回放巩固：练 7-9 时同时回放 1-6", flush=True)
    base_b = load_base()
    cortex_b = consolidate(base_b, old, EPOCHS_PHASE1, LR_SLOW)
    cortex_b = consolidate(cortex_b, new + old, EPOCHS_REPLAY, LR_SLOW)
    ok_old_b = show("  巩固后 旧(1-6)", cortex_b, old)
    ok_new_b = show("  新(7-9)", cortex_b, new)

    print("\n" + "=" * 60, flush=True)
    print("总结", flush=True)
    print("=" * 60, flush=True)
    print(f"  天真巩固（只练新，不回放）: 旧保住 {ok_old_a}/6, 新学到 {ok_new_a}/3", flush=True)
    print(f"  回放巩固（新+旧一起练）    : 旧保住 {ok_old_b}/6, 新学到 {ok_new_b}/3", flush=True)
    print(f"\n总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
