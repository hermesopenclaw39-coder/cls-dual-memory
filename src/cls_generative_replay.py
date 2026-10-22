# -*- coding: utf-8 -*-
"""
生成式回放（做梦）：不存旧数据原文，只用模型自己的"记忆再激活"来巩固新知识。

对照（前面两个实验已得）：
  天真巩固（不回放）      : 旧知识保住 2/6
  原始回放（重放原文）    : 旧知识保住 6/6
本实验（生成式回放）：
  巩固新知识 7-9 时，回放模型自己对旧事实前缀的"梦境"（续写），而非原文。
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
EPOCHS_REPLAY = 20   # 回放巩固的轮数
MAX_NEW = 12

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


def consolidate(model, texts, epochs, lr):
    """在 model 上训练一个语义 LoRA（轮询 texts）并合并进权重，返回合并后的模型。"""
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
    print("生成式回放（做梦）：不存原文，只回放模型自己的记忆", flush=True)
    print(f"设备: {DEVICE}  dtype: {DTYPE}", flush=True)
    print("=" * 60, flush=True)

    base = load_base()
    cortex = consolidate(base, [tr for tr, _, _ in old], EPOCHS_PHASE1, LR_SLOW)
    show("  巩固旧知识后 旧(1-6)", cortex, old)

    # 做梦：让模型自己对旧前缀续写，得到"梦境"（旧记忆的再激活，而非原文）
    print("\n模型生成的'梦境'（对旧前缀的续写 = 旧记忆的再激活）:", flush=True)
    dreams = []
    for _tr, q, _ans in old:
        cont = generate(cortex, q)
        dream = (q + " " + cont).strip()
        dreams.append(dream)
        print(f"  · {dream!r}", flush=True)

    # 巩固新知识 + 回放"梦境"（而非原文）
    cortex = consolidate(cortex, [tr for tr, _, _ in new] + dreams, EPOCHS_REPLAY, LR_SLOW)
    ok_old = show("  回放梦境巩固后 旧(1-6)", cortex, old)
    ok_new = show("  新(7-9)", cortex, new)

    print("\n" + "=" * 60, flush=True)
    print("总结", flush=True)
    print("=" * 60, flush=True)
    print(f"  生成式回放（回放梦境，非原文）: 旧保住 {ok_old}/6, 新学到 {ok_new}/3", flush=True)
    print(f"  （对照：天真巩固 2/6，原始回放 6/6）", flush=True)
    print(f"\n总耗时 {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
