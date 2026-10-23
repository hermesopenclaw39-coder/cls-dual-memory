# -*- coding: utf-8 -*-
"""检索基线：BM25 与 bge-small，和冻结的 7B 隐状态做对照。

问题：MaxSim 80.5%/74.0%（400/800 条）已经比 mean-pool 好很多，
但它用的是「冻结的 7B 隐状态」——那本来就不是为检索训练的。
换成专用检索器会怎样？这决定「训练 key encoder」值不值得做。

    python src/retrieval_baselines.py
"""

import os
import re
import time

import numpy as np
from rank_bm25 import BM25Okapi

from facts_scaled import make_facts_paraphrase

SIZES = [int(x) for x in os.environ.get("SIZES", "400,800").split(",")]
BGE_ID = os.environ.get("BGE_ID", "BAAI/bge-small-en-v1.5")


def tok(text):
    return re.findall(r"[a-z0-9']+", text.lower())


def evaluate(scores, n):
    top1 = (scores.argmax(1) == np.arange(n)).mean()
    top5 = np.mean([i in np.argsort(-scores[i])[:5] for i in range(n)])
    top20 = np.mean([i in np.argsort(-scores[i])[:20] for i in range(n)])
    return top1, top5, top20


def main():
    pool = make_facts_paraphrase(max(SIZES))
    model = None

    for n in SIZES:
        facts = pool[:n]
        docs = [tok(t) for t, _, _ in facts]
        queries = [tok(q) for _, q, _ in facts]
        started = time.time()

        bm25 = BM25Okapi(docs)
        scores = np.stack([bm25.get_scores(q) for q in queries])
        t1, t5, t20 = evaluate(scores, n)
        print("%5d  BM25      top1 %5.1f%%  top5 %5.1f%%  top20 %5.1f%%  (%.1fs)"
              % (n, t1 * 100, t5 * 100, t20 * 100, time.time() - started), flush=True)

        if model is None:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(BGE_ID)
        started = time.time()
        D = model.encode([t for t, _, _ in facts], normalize_embeddings=True, batch_size=64)
        Q = model.encode([q for _, q, _ in facts], normalize_embeddings=True, batch_size=64)
        scores = Q @ D.T
        t1, t5, t20 = evaluate(scores, n)
        print("%5d  bge-small top1 %5.1f%%  top5 %5.1f%%  top20 %5.1f%%  (%.1fs)"
              % (n, t1 * 100, t5 * 100, t20 * 100, time.time() - started), flush=True)


if __name__ == "__main__":
    main()
