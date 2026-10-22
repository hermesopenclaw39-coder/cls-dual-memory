# -*- coding: utf-8 -*-
"""路由随规模的变化：只算 embedding，不训练任何适配器。

CLS-on-LoRA 的存储侧是「每条事实一个适配器」，理论上有多少条就能背多少条；
真正的瓶颈是路由——用问题去 N 个 key 里挑一个。这里把路由单独拿出来测，
因为它决定了这套架构在 400/800 条时还有没有意义。

key   = 事实原句的 mean-pooled 最后一层隐状态（与原实验一致）
query = 测试前缀的同一个量
预测  = 与 query 余弦相似度最大的 key

    python src/routing_scale.py
"""

import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from facts import FACTS_50
from facts_scaled import make_facts, make_facts_diverse

FACTS_SOURCE = os.environ.get("FACTS_SOURCE", "scaled")   # scaled | hand

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
SIZES = [int(x) for x in os.environ.get("SIZES", "50,100,200,400,800").split(",")]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
    )
    model.eval()
    return tokenizer, model


def mean_embed(tokenizer, model, text):
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    h = out.hidden_states[-1]
    mask = inp["attention_mask"].unsqueeze(-1).float()
    return (h * mask).sum(1) / mask.sum(1)


def main():
    tokenizer, model = load_model()
    if FACTS_SOURCE == "hand":
        pool = FACTS_50
    elif FACTS_SOURCE == "diverse":
        pool = make_facts_diverse(max(SIZES))
    else:
        pool = make_facts(max(SIZES))
    print("model=%s facts=%s" % (MODEL_ID, FACTS_SOURCE))
    print("%6s %10s %10s %10s %10s" % ("N", "top-1", "top-5", "margin", "sec"))
    for n in SIZES:
        started = time.time()
        facts = pool[:n]
        keys = torch.cat([mean_embed(tokenizer, model, t) for t, _, _ in facts])
        qs = torch.cat([mean_embed(tokenizer, model, q) for _, q, _ in facts])
        keys = keys / (keys.norm(dim=-1, keepdim=True) + 1e-9)
        qs = qs / (qs.norm(dim=-1, keepdim=True) + 1e-9)
        sims = qs @ keys.T
        gold = torch.arange(n, device=sims.device)
        top1 = (sims.argmax(dim=1) == gold).float().mean().item()
        top5_idx = sims.topk(min(5, n), dim=1).indices
        top5 = (top5_idx == gold.unsqueeze(1)).any(dim=1).float().mean().item()
        top2 = sims.topk(2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).mean().item()
        print("%6d %9.1f%% %9.1f%% %10.3f %10.0f" % (n, top1 * 100, top5 * 100, margin, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
