# -*- coding: utf-8 -*-
"""诊断：7B 4-bit 下，巩固步骤（train_cycle 交错训练）为什么 loss 卡住。
对比 rank=8 vs rank=64，并打印 loss 轨迹。"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda"
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

FACTS = [
    "The Zorblax protocol uses a passcode: VELVET-77.",
    "Planet Threya orbits a star called Umbral.",
    "The Guild of Veils was founded in the year 1407.",
    "Agent Corvin's cover name is Linh.",
    "The mineral Xylium is harvested on the moon of Argon-9.",
    "Starberry jam needs exactly three grams of moon-salt.",
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token_id = tokenizer.eos_token_id


def bnb():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.float32,
                              bnb_4bit_use_double_quant=True)


def load():
    m = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb(), device_map="auto")
    m = prepare_model_for_kbit_training(m, use_gradient_checkpointing=False)
    return m


def test(rank, lr, epochs):
    t0 = __import__("time").time()
    base = load()
    cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=rank * 2,
                     lora_dropout=0.0, target_modules=TARGETS)
    m = get_peft_model(base, cfg)
    nparams = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"\n=== rank={rank} lr={lr} epochs={epochs}  trainable={nparams} ===", flush=True)

    m.train()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    for e in range(epochs):
        for text in FACTS:
            inp = {k: v.to(DEVICE) for k, v in tokenizer(text, return_tensors="pt").items()}
            ids = inp["input_ids"]
            opt.zero_grad()
            loss = m(input_ids=ids, labels=ids).loss
            loss.backward()
            opt.step()
        if (e + 1) % 10 == 0 or e == 0:
            print(f"  epoch {e + 1:3d}: loss={float(loss):.4f}", flush=True)

    m.eval()
    inp = tokenizer("The Zorblax protocol uses a passcode:", return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=12, do_sample=False,
                         pad_token_id=tokenizer.eos_token_id)
    got = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  生成: {got!r}  (耗时 {__import__('time').time() - t0:.0f}s)", flush=True)
    del m, base
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    test(8, 1e-3, 100)
    test(64, 1e-3, 100)
