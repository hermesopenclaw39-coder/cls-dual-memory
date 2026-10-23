# CLS Dual-Memory: LoRA as Memory Slots

**English** | [简体中文](README.zh-CN.md)

> Can a language model keep factual memories in **weights** instead of context? This repo is a series of experiments on that question: one rank-8 LoRA per fact, cosine gating, replay consolidation — plus the baselines and the negative results that came out of testing them honestly.
>
> Qwen2.5-7B-Instruct, 4-bit QLoRA, synthetic facts, one RTX 5090 / 4090.

---

## Status (2026-10-23)

Five claims, ranked by how much I trust them. Details and raw logs are in the sections below and in [`docs/report.md`](docs/report.md).

1. **A per-fact LoRA does answer questions it was never asked during training.** 50 independent adapters, each trained 100 steps on one sentence, then asked a paraphrased question: **46/50 (92%)**. My earlier report of 16% was a training-loop bug — the adapter weights were never reset between facts, so each "adapter" had been sequentially trained on all previous facts. Fixed, rerun, 92%.
2. **But 50 facts is the scale where in-context learning is already at 100%.** With a 24-token answer budget, pasting all 50 facts into the prompt scores 50/50. On this scale the architecture has not been shown to beat the baseline it was built to replace.
3. **A frozen LM's pooled hidden state is a bad retriever.** Mean-pooling the last hidden state gives 59% top-1 at 400 facts and 53% at 800. BM25 scores 100% and a 33M-parameter bge-small scores 97–99% on the same task. Late interaction (ColBERT-style MaxSim) recovers the 7B embeddings to 80%/74%.
4. **Retrieve-then-read works — and it is RAG, not memory-in-weights.** MaxSim top-5 → read those five facts in the prompt → answer: 94% at 400 facts and 83% at 800, against 93%/85% for full-context in-context learning, with **~110 prompt tokens instead of 7,865–15,807**. No adapter is involved in that pipeline; the earlier phrasing "the memory lives in weights" was wrong and is retracted.
5. **What is still unknown.** Whether per-fact LoRA holds at 400+ facts (400 adapters × 100 steps ≈ 1.6 h, not yet run), and whether training each adapter on several surface forms helps it answer held-out question forms. The first attempt at the latter scored 29/50 (58%), but it is confounded: fewer steps per form and a different held-out question set, so it is not a clean test.

## What is actually new here

- The **routing failure is measurable and fixable**: pooling a sentence into one vector destroys the entity name; late interaction keeps it.
- **One missing reset flips the result.** With independent adapters and real questions the same recipe scores 46/50 instead of 8/50; the difference is a single `load_snapshot(init)` at the top of the training loop. Adapter training loops silently turn independent memories into sequential ones when that line is absent.
- A **retriever hierarchy** on identical facts: BM25 100% > bge-small 97–99% > 7B MaxSim 74–80% > 7B mean-pool 11–59%.

## 1. The original experiment

| Stage | Recall (50 facts) |
|---|---|
| Frozen base | 0/50 |
| Hippocampus: 50 × rank-8 LoRA + cosine gate | 44/50 |
| Shared LoRA, sequential | 1/50 |
| Single-adapter consolidation (rank-32) | 0/50 |
| Grouped consolidation (9 × rank-8) | 50/50 (oracle grouping) |

The 44/50 is measured with **prefix queries** — the query is the beginning of the stored sentence, so the model is completing a sentence it memorised. Section 6 reruns the hippocampal stage with real questions, and with adapters reset per fact.

## 2. The in-context baseline

All 50 facts in the system prompt: **44/50** with a 12-token answer budget, **50/50** with 24. The six misses at 12 tokens are truncations, not errors. Zero training, 22 seconds.

## 3. Scaling the baseline

| Facts | ICL recall | Prompt tokens | Routing mean-pool | Routing MaxSim |
|---|---|---|---|---|
| 50 | 100% | 954 | 88.0% | — |
| 100 | 98% | 1,890 | 82.0% | — |
| 200 | 93% | 3,880 | 66.0% | — |
| 400 | 97% | 7,865 | 59.2% | 100% |
| 800 | 79% | 15,807 | 53.2% | 100% |

Prefix-style queries. At 1,600 facts the prompt reaches 33,466 tokens and no longer fits the 32k window.

> **Measurement trap, kept on purpose.** The first version of this sweep used one uniform sentence template for every fact. The gate scored 48% at N=50 on that set, versus 94% on the hand-written facts with identical code — the sentences were too similar for any embedding to separate. Uniform synthetic facts measure the template, not the method. The old run is kept in `results/icl_scaling.csv` as a warning.

## 4. With real questions

Six question templates, one per fact type (`What passcode does the X protocol use?`):

| Facts | ICL | Retrieval + in-context reading | MaxSim routing | mean-pool routing |
|---|---|---|---|---|
| 400 | 93% (7,865 tok) | **94%** (~110 tok) | 80.5% | 13.5% |
| 800 | 85% (15,807 tok) | 83% (~110 tok) | 74.0% | 10.7% |

![Fixing the gate](results/fig_routing_fix.png)

The hybrid column is RAG: it retrieves the top-5 fact **strings** and puts them in the prompt. It is the best cost/accuracy trade in this repo, and it does not use adapters at all.

## 5. Retrievers, same task

| Retriever | 400 | 800 | Per query |
|---|---|---|---|
| BM25 | 100% | 100% | 0.1 ms |
| bge-small-en-v1.5 (33M) | 98.8% | 97.4% | 0.5 ms |
| Qwen 7B, late interaction | 80.5% | 74.0% | 2 ms |
| Qwen 7B, mean-pooled | 13.5% | 10.7% | 2 ms |

Caveat: every question contains the entity name, which is exactly what BM25 exploits.

## 6. Per-fact LoRA: completion vs questions

50 adapters, each trained 100 steps on one sentence, weights reset before every fact:

| Query | Recall |
|---|---|
| Paraphrased question, unseen during training | **46/50 (92%)** |
| Same, with BM25 routing instead of oracle | 46/50 |

Earlier numbers in this repo (16%, "completion only") came from a bug: the adapter was never reset, so adapter *j* had been trained on facts 0…*j*. That is sequential learning with snapshots, not independent memories. After the fix the adapters answer real questions.

**The multi-form experiment** (train on the statement plus two question paraphrases, test on a third) scored 29/50 oracle, 25/50 routed. It is **not** a clean test: each form got 40 steps instead of 100, and the held-out question set differs from the one in the table above. Treat it as a pilot.

## 7. What is not established

- **Scale.** Per-fact LoRA has only been measured at 50 facts. 400 adapters is ≈1.6 h of training; it is the single most informative missing number.
- **Win condition.** At every scale tested, in-context learning is at least as accurate. The architecture's case currently rests on *cost* (110 vs 15,807 prompt tokens), not accuracy.
- **Real questions.** All questions here are templated. Natural paraphrases that avoid the entity name would hurt both BM25 and MaxSim.
- **Single seed, single model.** Every number is one run on Qwen2.5-7B-Instruct.

## Reproduce

| Script | What it measures |
|---|---|
| `src/cls_demo_7b_50.py` | the original five-stage pipeline |
| `src/baseline_icl.py` | in-context baseline at a fixed fact count |
| `src/baseline_scale.py` | ICL vs fact count |
| `src/routing_scale.py` | routing accuracy vs fact count |
| `src/routing_improve.py` | scoring functions + hybrid (RAG) |
| `src/routing_paraphrase.py` | same, with real questions |
| `src/retrieval_baselines.py` | BM25 and bge-small |
| `src/hippocampus_scale.py` | per-fact LoRA, reset per fact |

```bash
pip install -r requirements.txt
python src/baseline_icl.py          # needs the 7B checkpoint
python src/retrieval_baselines.py   # no LLM needed for BM25
```

`src/facts_scaled.py` generates the fact sets (uniform / diverse / question-variant), all seeded and reproducible. Raw logs are in `results/logs/`.

## Related work

- **Complementary learning systems** — McClelland, McNaughton & O'Reilly (1995), *Psychological Review* 102(3); Kumaran, Hassabis & McClelland (2016), *TiCS* 20(7).
- **Knowledge extraction and the reversal curse** — Allen-Zhu & Li, *Physics of Language Models* Part 3.1; Berglund et al. (2023), arXiv:2309.12288. Relevant here: a fact trained in one surface form may not be answerable in another.
- **Catastrophic forgetting** — French (1999); Kirkpatrick et al. (2017), EWC.
- **LoRA and adapter banks** — Hu et al. (2021); Zadouri et al. (2023), MoLE; Huang et al. (2024), LoraHub.
- **Retrieval** — Lewis et al. (2020), RAG; Khattab & Zaharia (2020), ColBERT (late interaction); Robertson & Zaragoza (2009), BM25.
- **In-context memory** — Brown et al. (2020); Packer et al. (2023), MemGPT.

## Citation / License

See [`CITATION.cff`](CITATION.cff). Code MIT, report CC BY 4.0. Model weights are not distributed.
