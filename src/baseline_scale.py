# -*- coding: utf-8 -*-
"""ICL 规模扫描：事实从 50 条涨到 1600 条，看「上下文里找得到」这件事何时失效。

动机：50 条时 ICL 已经 50/50（见 results/logs/）。要判断权重记忆有没有存在意义，
就得知道基线在哪失效——它失效的规模，才是适配器方案真正的主场。

    python src/baseline_scale.py                        # 默认 50,100,200,400,800,1600
    SIZES=50,200,800 PROBES=60 python src/baseline_scale.py
"""

import os
import random
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

from facts_scaled import make_facts, make_facts_diverse

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
SIZES = [int(x) for x in os.environ.get("SIZES", "50,100,200,400,800,1600").split(",")]
PROBES = int(os.environ.get("PROBES", "100"))
MAX_NEW = int(os.environ.get("MAX_NEW", "24"))
USE_4BIT = os.environ.get("USE_4BIT", "1") != "0"
FACTS_SOURCE = os.environ.get("FACTS_SOURCE", "scaled")   # scaled | diverse


def build_system_prompt(facts):
    lines = ["You are a careful assistant. Use only the facts below to answer.", ""]
    for i, (text, _, _) in enumerate(facts, start=1):
        lines.append("%d. %s" % (i, text))
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
        out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    set_seed(42)
    tokenizer, model = load_model()
    if FACTS_SOURCE == "diverse":
        pool = make_facts_diverse(max(SIZES))
    else:
        pool = make_facts(max(SIZES))
    rng = random.Random(0)
    print("model=%s facts=%s probes=%s max_new=%s" % (MODEL_ID, FACTS_SOURCE, PROBES, MAX_NEW))
    print("%6s %8s %10s %8s" % ("N", "tokens", "recall", "sec"))
    for n in SIZES:
        facts = pool[:n]
        prompt = build_system_prompt(facts)
        ntok = len(tokenizer(prompt).input_ids)
        idx = list(range(n))
        rng.shuffle(idx)
        idx = idx[: min(PROBES, n)]
        hits = 0
        started = time.time()
        for i in idx:
            _, question, answer = facts[i]
            reply = ask(tokenizer, model, prompt, question)
            hits += int(answer.lower() in reply.lower())
        elapsed = time.time() - started
        print("%6d %8d %10s %8.0f" % (n, ntok, "%d/%d" % (hits, len(idx)), elapsed), flush=True)


if __name__ == "__main__":
    main()
