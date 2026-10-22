# -*- coding: utf-8 -*-
"""诊断2：7B 4-bit 巩固，rank=8, lr=5e-4, 150 epochs，看能否收敛到同时记住 6 条事实。"""
import os
import sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from facts import FACTS_6
FACTS = FACTS_6  # 事实集合见 src/facts.py

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token_id = tokenizer.eos_token_id

RANK, LR, EPOCHS = 8, 5e-4, 150


def bnb():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float32,
                              bnb_4bit_use_double_quant=True)


t0 = time.time()
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb(), device_map="auto")
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=RANK, lora_alpha=RANK * 2,
                 lora_dropout=0.0, target_modules=TARGETS)
m = get_peft_model(base, cfg)
nparams = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"rank={RANK} lr={LR} epochs={EPOCHS} trainable={nparams} (load {time.time()-t0:.0f}s)", flush=True)

m.train()
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)
texts = [t for t, _, _ in FACTS]
for e in range(EPOCHS):
    for text in texts:
        ids = {k: v.to(DEVICE) for k, v in tokenizer(text, return_tensors="pt").items()}["input_ids"]
        opt.zero_grad()
        loss = m(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
    if (e + 1) % 20 == 0 or e == 0:
        print(f"  epoch {e + 1:3d}: loss={float(loss):.4f}", flush=True)

m.eval()


def gen(prompt):
    inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=12, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)


ok = 0
for i, (_tr, q, ans) in enumerate(FACTS):
    got = gen(q).lower()
    hit = ans.lower() in got
    ok += hit
    print(f"  事实{i}: {'✓' if hit else '✗'} {got!r}", flush=True)
print(f"召回: {ok}/6  (总耗时 {time.time()-t0:.0f}s)", flush=True)
