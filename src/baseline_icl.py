# -*- coding: utf-8 -*-
"""ICL 基线：把全部事实直接放进 prompt，逐条提问，测召回。不训练任何适配器。

用途：回答一个绕不开的问题——「把记忆写进权重」相比「把记忆塞进上下文」
到底有没有优势？本脚本只提供对照数字，结论留给 README。

    python src/baseline_icl.py              # 50 条（默认）
    N_FACTS=6 python src/baseline_icl.py    # 6 条
    USE_4BIT=0 python src/baseline_icl.py   # 不量化（需要 ~16GB 显存）
"""

import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

from facts import FACTS_6, FACTS_50

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
SEED = int(os.environ.get("SEED", "42"))
N_FACTS = int(os.environ.get("N_FACTS", "50"))
USE_4BIT = os.environ.get("USE_4BIT", "1") != "0"
MAX_NEW = 12


def pick_facts(n):
    if n <= len(FACTS_6):
        return FACTS_6[:n]
    return FACTS_50[:n]


def build_system_prompt(facts):
    lines = ["You are a careful assistant. Use only the facts below to answer.", ""]
    for i, (text, _, _) in enumerate(facts, start=1):
        lines.append(f"{i}. {text}")
    return "\n".join(lines)


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    kwargs = {"device_map": "auto"}
    if USE_4BIT:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    return tokenizer, model


def ask(tokenizer, model, system_prompt, question):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    set_seed(SEED)
    facts = pick_facts(N_FACTS)
    system_prompt = build_system_prompt(facts)
    print(f"ICL baseline | model={MODEL_ID} | facts={len(facts)} | 4bit={USE_4BIT} | seed={SEED}")
    print(f"prompt tokens (approx): {len(system_prompt) // 4}")

    tokenizer, model = load_model()
    hits = 0
    started = time.time()
    for i, (_, question, answer) in enumerate(facts, start=1):
        reply = ask(tokenizer, model, system_prompt, question)
        ok = answer.lower() in reply.lower()
        hits += int(ok)
        print(f"[{i:2d}/{len(facts)}] {'HIT ' if ok else 'MISS'} expect={answer!r} got={reply[:70]!r}")

    elapsed = time.time() - started
    print(f"\nICL recall: {hits}/{len(facts)}  ({hits / len(facts) * 100:.0f}%)  in {elapsed:.0f}s")
    print("对照：README 中分组巩固（oracle 分组）50/50，海马体 44/50。")


if __name__ == "__main__":
    main()
