# -*- coding: utf-8 -*-
"""诊断：7B 4-bit 上，单条事实能否被 LoRA 记住（不同 rank / lr）。"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import gc
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
FACT = "The Zorblax protocol uses a passcode: VELVET-77."
STEPS = 200

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token_id = tokenizer.eos_token_id


def bnb():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)


def load():
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb(), device_map="auto")
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)
    m.eval()
    return m


ids = {k: v.to(DEVICE) for k, v in tokenizer(FACT, return_tensors="pt").items()}["input_ids"]


def test(rank, lr):
    base = load()
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=rank * 2,
                     lora_dropout=0.0, target_modules=TARGETS)
    m = get_peft_model(base, cfg)
    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    final = None
    for s in range(STEPS):
        opt.zero_grad()
        loss = m(input_ids=ids, labels=ids).loss
        loss.backward()
        opt.step()
        final = float(loss.detach())
        if (s + 1) % 50 == 0:
            print(f"    step {s + 1}: loss={final:.4f}", flush=True)
    del m, base
    gc.collect()
    torch.cuda.empty_cache()
    return final


print("=== 7B 4-bit 单条事实记忆诊断（200 步）===", flush=True)
for rank, lr in [(8, 1e-2), (32, 1e-2), (64, 1e-2)]:
    t = time.time()
    fl = test(rank, lr)
    print(f"  rank={rank} lr={lr} -> 最终 loss={fl:.4f} ({time.time() - t:.0f}s)", flush=True)
