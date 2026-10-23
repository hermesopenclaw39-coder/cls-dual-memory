# -*- coding: utf-8 -*-
"""把真正的 per-fact LoRA 接进来：存储侧在 400 条规模还成立吗？

前面所有实验只测了检索。这一步训练真实的适配器：
  每条事实一个 rank-8 LoRA，训练 S 步，权重快照存 CPU；
  评测时先用 BM25 路由（400 条时 100%），set_adapter 后直接问问题。

同时报告两个数：
  oracle recall —— 直接加载正确适配器（衡量存储）
  e2e recall    —— 先路由再加载（衡量整条链路）

    N=400 STEPS=20 python src/hippocampus_scale.py
"""

import os
import re
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from facts_scaled import make_facts_paraphrase

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
N = int(os.environ.get("N", "50"))
STEPS = int(os.environ.get("STEPS", "20"))
PROBES = int(os.environ.get("PROBES", "100"))
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
    model.eval()

    facts = make_facts_paraphrase(N)
    print("model=%s facts=%d steps=%d probes=%d" % (MODEL_ID, N, STEPS, PROBES), flush=True)

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
        if (i + 1) % 50 == 0 or i == 0:
            print("  trained %d/%d  (%.0fs)" % (i + 1, N, time.time() - started), flush=True)

    # ---- oracle recall ----
    model.eval()
    probes = list(range(N))
    rng = __import__("random").Random(0)
    rng.shuffle(probes)
    probes = probes[: min(PROBES, N)]

    state = model.state_dict()
    def load_snapshot(snap):
        for k, v in snap.items():
            state[k].copy_(v.to(state[k].device))

    oracle = 0
    t0 = time.time()
    for i in probes:
        load_snapshot(snaps[i])
        messages = [{"role": "user", "content": facts[i][1]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        reply = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        oracle += int(facts[i][2].lower() in reply.lower())
    print("oracle recall  %d/%d  (%.0fs)" % (oracle, len(probes), time.time() - t0), flush=True)

    # ---- end to end with BM25 routing ----
    def tok(s):
        return re.findall(r"[a-z0-9']+", s.lower())
    bm25 = BM25Okapi([tok(t) for t, _, _ in facts])
    e2e = 0
    t0 = time.time()
    for i in probes:
        scores = bm25.get_scores(tok(facts[i][1]))
        j = int(scores.argmax())
        load_snapshot(snaps[j])
        messages = [{"role": "user", "content": facts[i][1]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        reply = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        e2e += int(facts[i][2].lower() in reply.lower())
    print("e2e (bm25)     %d/%d  (%.0fs)" % (e2e, len(probes), time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
