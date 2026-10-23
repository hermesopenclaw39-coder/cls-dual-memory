# -*- coding: utf-8 -*-
"""路由能救吗？——四种打分函数 + top-k 短名单 + ICL 的混合管线。

上一轮（routing_scale.py）证明：per-fact LoRA 的瓶颈是门控，不是容量。
这里问一个建设性的问题：换一种打分方式，检索能救回多少？

四种打分，全部只算 embedding：
  mean   —— 末层隐状态按 token 平均（原方案）
  last   —— 末层最后一个 token
  max    —— 末层逐维取最大
  maxsim —— ColBERT 式 late interaction：每个 query token 在各 key token 上取最大相似度再求和

然后拿最好的那种做混合：先由门控选 top-k 条事实，再把它们放进 prompt 让模型回答。
这才是这套架构真正能被部署的形态。

    python src/routing_improve.py
"""

import os
import random
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

from facts_scaled import make_facts_diverse

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
SIZES = [int(x) for x in os.environ.get("SIZES", "400,800").split(",")]
KS = [int(x) for x in os.environ.get("KS", "5,10").split(",")]
PROBES = int(os.environ.get("PROBES", "100"))
MAX_NEW = 24


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


def token_matrix(tokenizer, model, text):
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    h = out.hidden_states[-1][0].float()
    mask = inp["attention_mask"][0].bool()
    return h[mask]


def pool(h, kind):
    if kind == "mean":
        return h.mean(0)
    if kind == "last":
        return h[-1]
    if kind == "max":
        return h.max(0).values
    raise ValueError(kind)


def pad_keys(key_tok):
    """把变长 key 的 token 矩阵补成 [N, Tmax, D] + mask。"""
    tmax = max(h.shape[0] for h in key_tok)
    dim = key_tok[0].shape[1]
    out = torch.zeros(len(key_tok), tmax, dim, device=key_tok[0].device)
    mask = torch.zeros(len(key_tok), tmax, dtype=torch.bool, device=key_tok[0].device)
    for i, h in enumerate(key_tok):
        out[i, : h.shape[0]] = h
        mask[i, : h.shape[0]] = True
    return out, mask


def maxsim_matrix(q_tok, key_tok):
    """ColBERT 式 late interaction：每个 query token 取最大 key-token 相似度再求和。"""
    K, mask = pad_keys(key_tok)
    K = K / (K.norm(dim=-1, keepdim=True) + 1e-9)
    rows = []
    for q in q_tok:
        qn = q / (q.norm(dim=-1, keepdim=True) + 1e-9)
        sims = torch.einsum("td,nkd->ntk", qn, K)          # [N, Tq, Tk]
        sims = sims.masked_fill(~mask[:, None, :], -1e4)
        rows.append(sims.max(dim=2).values.sum(dim=1) / qn.shape[0])
    return torch.stack(rows)

def build_prompt(facts):
    lines = ["You are a careful assistant. Use only the facts below to answer.", ""]
    for i, (text, _, _) in enumerate(facts, start=1):
        lines.append("%d. %s" % (i, text))
    return "\n".join(lines)


def ask(tokenizer, model, prompt, question):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    set_seed(42)
    tokenizer, model = load_model()
    pool_facts = make_facts_diverse(max(SIZES))
    rng = random.Random(0)
    print("model=%s probes=%d" % (MODEL_ID, PROBES), flush=True)

    for n in SIZES:
        facts = pool_facts[:n]
        started = time.time()
        key_tok = [token_matrix(tokenizer, model, t) for t, _, _ in facts]
        q_tok = [token_matrix(tokenizer, model, q) for _, q, _ in facts]
        emb_seconds = time.time() - started
        keys = torch.stack([pool(h, "mean") for h in key_tok])
        emb = torch.stack([pool(h, "mean") for h in q_tok])
        gold = torch.arange(n, device=keys.device)

        print("\n== N=%d  (embeddings %.0fs) ==" % (n, emb_seconds), flush=True)
        print("%8s %8s %8s %8s" % ("scoring", "top-1", "top-5", "top-20"))
        best = None
        for kind in ["mean", "last", "max"]:
            qv = torch.stack([pool(h, kind) for h in q_tok])
            kv = torch.stack([pool(h, kind) for h in key_tok])
            qv = qv / (qv.norm(dim=-1, keepdim=True) + 1e-9)
            kv = kv / (kv.norm(dim=-1, keepdim=True) + 1e-9)
            sims = qv @ kv.T
            ok1 = (sims.argmax(1) == gold).float().mean().item()
            ok5 = (sims.topk(5, 1).indices == gold[:, None]).any(1).float().mean().item()
            ok20 = (sims.topk(20, 1).indices == gold[:, None]).any(1).float().mean().item()
            print("%8s %7.1f%% %7.1f%% %7.1f%%" % (kind, ok1 * 100, ok5 * 100, ok20 * 100), flush=True)
            if best is None or ok1 > best[1]:
                best = (kind, ok1, sims)

        # late interaction
        sims_ms = maxsim_matrix(q_tok, key_tok)
        ok1 = (sims_ms.argmax(1) == gold).float().mean().item()
        ok5 = (sims_ms.topk(5, 1).indices == gold[:, None]).any(1).float().mean().item()
        ok20 = (sims_ms.topk(20, 1).indices == gold[:, None]).any(1).float().mean().item()
        print("%8s %7.1f%% %7.1f%% %7.1f%%" % ("maxsim", ok1 * 100, ok5 * 100, ok20 * 100), flush=True)
        if ok1 > best[1]:
            best = ("maxsim", ok1, sims_ms)

        # hybrid: shortlist -> ICL
        kind, _, sims = best
        idx = list(range(n))
        rng.shuffle(idx)
        probes = idx[: min(PROBES, n)]
        print("  hybrid with %s, %d probes" % (kind, len(probes)), flush=True)
        for k in KS:
            hits = 0
            covered = 0
            for i in probes:
                short = sims[i].topk(k).indices.tolist()
                covered += int(i in short)
                sub = [facts[j] for j in short]
                reply = ask(tokenizer, model, build_prompt(sub), facts[i][1])
                hits += int(facts[i][2].lower() in reply.lower())
            print("  hybrid top-%-3d recall %d/%d   gold-in-shortlist %d/%d"
                  % (k, hits, len(probes), covered, len(probes)), flush=True)


if __name__ == "__main__":
    main()
