# -*- coding: utf-8 -*-
"""真实 per-fact LoRA：每条事实一个独立适配器，评测「能不能被问出来」。

重要修正：每条事实训练前把 LoRA 权重恢复到初始状态。
否则第 j 个快照其实是「顺序学完第 0..j 条之后」的权重，不是独立适配器。

三种事实/提问设置：
  prefix     —— 用事实自身的前缀提问（原实验设置，测的是补全）
  paraphrase —— 用一条没见过的问句提问（测的是问答）
  multi      —— 训练时见「陈述句 + 两种问法」，测试用第三种没见过的问法
                （Allen-Zhu & Li, Physics of LM 3.1；reversal curse 同类现象）

    N=50 STEPS=100 FACTS_STYLE=prefix     python src/hippocampus_scale.py
    N=50 STEPS=40  FACTS_STYLE=multi      python src/hippocampus_scale.py
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

from facts_scaled import make_facts, make_facts_paraphrase, make_facts_qvariants

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
N = int(os.environ.get("N", "50"))
STEPS = int(os.environ.get("STEPS", "100"))
PROBES = int(os.environ.get("PROBES", "50"))
FACTS_STYLE = os.environ.get("FACTS_STYLE", "paraphrase")   # prefix | paraphrase | multi
MAX_NEW = 24
RANK = 8
LR = 1e-3
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_items(style, n):
    """统一成 (train_texts, test_question, answer, fact_text)。"""
    if style == "multi":
        out = []
        for fact, q_train, q_test, ans in make_facts_qvariants(n):
            texts = [fact] + ["Q: %s\nA: %s" % (q, ans) for q in q_train]
            out.append((texts, q_test, ans, fact))
        return out
    raw = make_facts(n) if style == "prefix" else make_facts_paraphrase(n)
    return [([fact], q, ans, fact) for fact, q, ans in raw]


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

    items = build_items(FACTS_STYLE, N)
    print("model=%s style=%s facts=%d steps=%d probes=%d" % (MODEL_ID, FACTS_STYLE, N, STEPS, PROBES), flush=True)

    state = model.state_dict()

    def load_snapshot(snap):
        for k, v in snap.items():
            state[k].copy_(v.to(state[k].device))

    init = {k: v.detach().clone() for k, v in state.items() if "lora_" in k}

    snaps = []
    started = time.time()
    for i, (texts, _, _, _) in enumerate(items):
        load_snapshot(init)                      # <-- 每条事实从同一个初始 LoRA 开始
        model.train()
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
        for text in texts:
            ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
            for _ in range(STEPS):
                opt.zero_grad()
                loss = model(input_ids=ids, labels=ids).loss
                loss.backward()
                opt.step()
        snaps.append({k: v.detach().clone() for k, v in state.items() if "lora_" in k})
        if (i + 1) % 10 == 0:
            print("  trained %d/%d  (%.0fs)" % (i + 1, N, time.time() - started), flush=True)

    model.eval()

    def gen(prompt):
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)

    def tok(s):
        return re.findall(r"[a-z0-9']+", s.lower())

    facts = [it[3] for it in items]
    bm25 = BM25Okapi([tok(f) for f in facts])
    probes = list(range(N))
    random.Random(0).shuffle(probes)
    probes = probes[: min(PROBES, N)]

    for mode in ["oracle", "bm25"]:
        hits = 0
        t0 = time.time()
        for i in probes:
            j = i if mode == "oracle" else int(bm25.get_scores(tok(items[i][1])).argmax())
            load_snapshot(snaps[j])
            reply = gen(items[i][1])
            hits += int(items[i][2].lower() in reply.lower())
        print("%-6s recall %d/%d  (%.0fs)" % (mode, hits, len(probes), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
