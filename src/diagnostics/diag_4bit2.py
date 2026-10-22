# -*- coding: utf-8 -*-
"""诊断2：7B 4-bit，fp32 计算 + 更低 lr，能否稳定记住单条事实。"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FACT = "The Zorblax protocol uses a passcode: VELVET-77."
STEPS = 300
RANK = 8
LR = 1e-3

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token_id = tokenizer.eos_token_id

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.float32, bnb_4bit_use_double_quant=True)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=False)
base.eval()

ids = {k: v.to(DEVICE) for k, v in tokenizer(FACT, return_tensors="pt").items()}["input_ids"]
cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=RANK, lora_alpha=RANK * 2,
                 lora_dropout=0.0, target_modules=TARGETS)
m = get_peft_model(base, cfg)
m.train()
opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=LR)

print(f"=== 7B 4-bit(fp32计算) 记忆诊断 rank={RANK} lr={LR} {STEPS}步 ===", flush=True)
for s in range(STEPS):
    opt.zero_grad()
    loss = m(input_ids=ids, labels=ids).loss
    loss.backward()
    opt.step()
    if (s + 1) % 50 == 0 or s == STEPS - 1:
        print(f"  step {s + 1}: loss={float(loss):.4f}", flush=True)

m.eval()
inp = {k: v.to(DEVICE) for k, v in tokenizer("The Zorblax protocol uses a passcode:", return_tensors="pt").items()}
with torch.no_grad():
    out = m.generate(**inp, max_new_tokens=12, do_sample=False, pad_token_id=tokenizer.eos_token_id)
print("生成:", repr(tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)), flush=True)
