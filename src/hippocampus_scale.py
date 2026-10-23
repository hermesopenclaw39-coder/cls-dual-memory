# -*- coding: utf-8 -*-
"""把真实的 per-fact LoRA 接进来：存储侧在规模下还成立吗？

每条事实训练一个 rank-8 LoRA（S 步），权重快照留 CPU；评测时先路由再加载。

评测提示词有两种形态，原实验用的是 raw（直接喂测试前缀，不加对话模板）：
  raw  —— tokenizer(prefix) → generate
  chat —— apply_chat_template([{user: prefix}]) → generate
两者差别很大：适配器是在原始句子上训练的，套上对话模板会把它推到分布外。

    N=50 STEPS=100 python src/hippocampus_scale.py
"""

import os
import random
import re
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from facts_scaled import make_facts, make_facts_paraphrase

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
N = int(os.environ.get("N", "50"))
FACTS_STYLE = os.environ.get("FACTS_STYLE", "paraphrase")   # prefix | paraphrase
STEPS = int(os.environ.get("STEPS", "100"))
PROBES = int(os.environ.get("PROBES", "50"))
MAX_NEW = 24
RANK = 8
LR = 1e-3
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
    )
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=RANK, lora_alpha=RANK * 2,
                     lora_dropout=0.0, target_modules=TARGETS)
    model = get_peft_model(base, cfg)

    facts = make_facts(N) if FACTS_STYLE == "prefix" else make_facts_paraphrase(N)
    print("model=%s facts=%d steps=%d probes=%d style=%s" % (MODEL_ID, N, STEPS, PROBES, FACTS_STYLE), flush=True)

    snaps = []
    started = time.time()
    for i, (text, _, _) in enumerate(facts):
        model.train()
        ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        for _ in range(STEPS):
            opt.zero_grad()
            loss = model(input_ids=ids, labels=ids).loss
            loss.backward()
            opt.step()
        snaps.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k})
        if (i + 1) % 10 == 0:
            print("  trained %d/%d  (%.0fs)" % (i + 1, N, time.time() - started), flush=True)

    model.eval()
    state = model.state_dict()

    def load_snapshot(snap):
        for k, v in snap.items():
            state[k].copy_(v.to(state[k].device))

    def gen(prompt, fmt):
        if fmt == "chat":
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                   tokenize=False, add_generation_prompt=True)
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

    def tok(s):
        return re.findall(r"[a-z0-9']+", s.lower())

    bm25 = BM25Okapi([tok(t) for t, _, _ in facts])
    probes = list(range(N))
    random.Random(0).shuffle(probes)
    probes = probes[: min(PROBES, N)]

    for fmt in ["raw", "chat"]:
        for mode in ["oracle", "bm25"]:
            hits = 0
            t0 = time.time()
            for i in probes:
                j = i
                if mode == "bm25":
                    j = int(bm25.get_scores(tok(facts[i][1])).argmax())
                load_snapshot(snaps[j])
                reply = gen(facts[i][1], fmt)
                hits += int(facts[i][2].lower() in reply.lower())
            print("%-5s %-6s recall %d/%d  (%.0fs)" % (fmt, mode, hits, len(probes), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
