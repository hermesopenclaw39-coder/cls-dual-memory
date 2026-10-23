# -*- coding: utf-8 -*-
"""问句型 query 下的路由：检验 maxsim 的 100% 是不是只靠词面重合。

prefix 版（routing_improve.py）里 query 就是事实的前缀，词元几乎逐个对得上；
这里把 query 换成真正的问句（What passcode does the X protocol use?），
名字仍在句子里，但句式完全不同。

    python src/routing_paraphrase.py
"""

import os
import random
import time

os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

import routing_improve as RI
from facts_scaled import make_facts_paraphrase

SIZES = [int(x) for x in os.environ.get("SIZES", "400,800").split(",")]
KS = [int(x) for x in os.environ.get("KS", "5").split(",")]
PROBES = int(os.environ.get("PROBES", "100"))


def main():
    RI.set_seed(42)
    tokenizer, model = RI.load_model()
    pool = make_facts_paraphrase(max(SIZES))
    rng = random.Random(0)

    for n in SIZES:
        facts = pool[:n]
        started = time.time()
        key_tok = [RI.token_matrix(tokenizer, model, t) for t, _, _ in facts]
        q_tok = [RI.token_matrix(tokenizer, model, q) for _, q, _ in facts]
        print("\n== paraphrase queries, N=%d (embeddings %.0fs) ==" % (n, time.time() - started), flush=True)
        print("%8s %8s %8s %8s" % ("scoring", "top-1", "top-5", "top-20"))
        gold = torch.arange(n, device=key_tok[0].device)
        best = (None, -1, None)
        for kind in ["mean", "last", "max"]:
            qv = torch.stack([RI.pool(h, kind) for h in q_tok])
            kv = torch.stack([RI.pool(h, kind) for h in key_tok])
            qv = qv / (qv.norm(dim=-1, keepdim=True) + 1e-9)
            kv = kv / (kv.norm(dim=-1, keepdim=True) + 1e-9)
            sims = qv @ kv.T
            ok1 = (sims.argmax(1) == gold).float().mean().item()
            ok5 = (sims.topk(5, 1).indices == gold[:, None]).any(1).float().mean().item()
            ok20 = (sims.topk(20, 1).indices == gold[:, None]).any(1).float().mean().item()
            print("%8s %7.1f%% %7.1f%% %7.1f%%" % (kind, ok1 * 100, ok5 * 100, ok20 * 100), flush=True)
            if ok1 > best[1]:
                best = (kind, ok1, sims)

        sims_ms = RI.maxsim_matrix(q_tok, key_tok)
        ok1 = (sims_ms.argmax(1) == gold).float().mean().item()
        ok5 = (sims_ms.topk(5, 1).indices == gold[:, None]).any(1).float().mean().item()
        ok20 = (sims_ms.topk(20, 1).indices == gold[:, None]).any(1).float().mean().item()
        print("%8s %7.1f%% %7.1f%% %7.1f%%" % ("maxsim", ok1 * 100, ok5 * 100, ok20 * 100), flush=True)
        if ok1 >= best[1]:
            best = ("maxsim", ok1, sims_ms)

        kind, _, sims = best
        idx = list(range(n))
        rng.shuffle(idx)
        probes = idx[: min(PROBES, n)]
        print("  hybrid with %s" % kind, flush=True)
        for k in KS:
            hits = covered = 0
            for i in probes:
                short = sims[i].topk(k).indices.tolist()
                covered += int(i in short)
                reply = RI.ask(tokenizer, model, RI.build_prompt([facts[j] for j in short]), facts[i][1])
                hits += int(facts[i][2].lower() in reply.lower())
            print("  hybrid top-%-2d recall %d/%d   gold-in-shortlist %d/%d"
                  % (k, hits, len(probes), covered, len(probes)), flush=True)


if __name__ == "__main__":
    main()
