# CLS Dual-Memory: LoRA as Memory Slots

**[English](README.md)** | **简体中文**

用 LoRA 适配器模拟互补学习系统（Complementary Learning Systems, CLS）的「海马体 + 皮层」，在 Qwen2.5-7B-Instruct（4-bit QLoRA）上做的一次小规模工程验证。

A small-scale engineering probe of **Complementary Learning Systems** on LoRA adapters: one adapter per episodic fact (hippocampus), replay-based consolidation into semantic adapters (cortex), and embedding-cosine gating (routing).

> **定位 / Status**：这是一份**探索性实验笔记**，不是论文，也不是对 CLS 理论的完整验证。数字来自 2026-10-08 ~ 10-10 的原始运行，仓库整理时未重跑。结论的适用边界见报告 §8。

---

## TL;DR

50 条虚构事实上的五阶段结果（Qwen2.5-7B-Instruct, 4-bit QLoRA, RTX 4090 24GB）：

| 阶段 | 召回 | 说明 |
|---|---|---|
| 冻结底座 | 0/50 | 基线：底座不知道这些虚构事实 |
| 海马体（50 个 rank-8 LoRA + 余弦门控） | **44/50** | 路由准确率 47/50；路由错误两轮一致 |
| 共享 LoRA 顺序学习 | 1/50 | 灾难性遗忘 |
| 单适配器巩固（rank-32） | 0/50 | loss 震荡在 4~5，未收敛 |
| 分组巩固（9 × rank-8，每组 6 条） | **50/50** | 组内召回；评测为 oracle 分组 |

![各阶段召回](results/fig_recall.png)

![巩固阶段 loss：单适配器 vs 分组](results/fig_consolidation_loss.png)

6 条版对照见 [`docs/report.md`](docs/report.md)。

### ICL 基线（2026-10-22 补充）

把 50 条事实原样塞进 system prompt（约 700 token），不训练、贪婪解码，脚本 `src/baseline_icl.py`：

| 生成长度 | 召回 | 耗时 | 说明 |
|---|---|---|---|
| 12 token（与其它阶段一致） | 44/50 | 22 秒 | 与海马体打平；6 条 MISS **全是被截断**，不是答错 |
| 24 token | **50/50** | 24 秒 | 全部一次答对 |

这是原文缺的那个对照，而且对 LoRA 方案不利：**50 条这个规模上，把事实放进 prompt 的准确率就等于最好的巩固配置**，且不需要适配器、不需要训练、也不需要 oracle 路由。权重记忆真正可能带来的东西是**恒定的上下文成本**（0 token vs 约 700）和事实量超出窗口后的优雅退化——但在这个规模上，这还只是论点，不是测量。12 token 那 6 条也很值得看：模型把事实原句复述了一遍，答案被截断在中间，这正是所有阶段共用同一个解码预算时会产生的假失败。
### 两侧同时测（同日补跑）

真正该问的不是「ICL 会不会衰减」，而是「谁先衰减」——prompt 的容量，还是余弦门控在 N 个适配器里挑对的概率。两侧用同一批事实：固定种子生成、六种句式、六种答案类型（`src/facts_scaled.py` 的 `make_facts_diverse`），句子在语义上真的不一样，而不只是名字不同。

| 事实数 | ICL 召回 | prompt token | 路由 top-1 | 路由 top-5 | 门控 margin |
|---|---|---|---|---|---|
| 50 | 50/50 · 100% | 954 | 88.0% | 100% | 0.013 |
| 100 | 98/100 · 98% | 1,890 | 82.0% | 98.0% | 0.009 |
| 200 | 93/100 · 93% | 3,880 | 66.0% | 89.0% | 0.007 |
| 400 | 97/100 · 97% | 7,865 | 59.2% | 81.2% | 0.006 |
| 800 | 79/100 · 79% | 15,807 | 53.2% | 75.1% | 0.005 |

![两条上限](results/fig_scale_compare.png)

路由是对每条事实都算的（精确值，无采样误差）；ICL 每档 100 个探针，400 条约 ±5 个点，800 条约 ±8 个点。

**两条曲线的失败方式不同。** prompt 容量比预想的耐用：到 400 条还有 93–100%，800 条仍有 79%；而门控在 prompt 还有 97% 的时候就已经掉到 59%。每个适配器只背自己那一句，所以**存储从来不是瓶颈，检索才是**。到 400 条，per-fact LoRA 已经打不过「把事实贴进 prompt」，而且是在还没算训练成本的情况下输的。

> **一个值得记下来的测量陷阱。** 这轮扫描的第一版，所有事实用的是同一个句式。在那种数据集上，门控在 N=50 时只有 **48%**，而手写的 50 条用同一份代码是 **94%**——句子太像，mean-pooled embedding 的余弦差只有 0.005，任何检索都会崩。同质数据也让 ICL 曲线显得更差（400 条时 86%，而不是 97%）。做检索类实验，合成事实必须有句式多样性，否则测的是模板，不是方法。那一版结果保留在 `results/icl_scaling.csv` 与 `results/logs/` 里备查。

### 门控能救吗？能 —— 而且结论要再改一次

上一节结尾那个 53% 不是架构的性质，是**把整句话压成一个向量**的性质。同一批 key 和 query，换四种打分，全部不训练：

| 打分 | N=400 top-1 | N=400 top-5 | N=800 top-1 | N=800 top-5 |
|---|---|---|---|---|
| 末层隐状态平均（原方案） | 59.2% | 81.2% | 53.2% | 75.1% |
| 最后一个 token | 2.5% | 10.2% | 1.5% | 5.7% |
| 逐维取最大 | 89.7% | 98.7% | 86.5% | 97.7% |
| **late interaction（ColBERT 式 MaxSim）** | **100%** | 100% | **100%** | 100% |

late interaction 不压缩：每个 query token 去事实的所有 token 里找最像的那个。在前缀式 query 上这是精确匹配，所以 100% —— 这也说明这批 query 太简单。

于是换成**真正的问句**（`What passcode does the X protocol use?`）重跑，问句和事实没有共同的表层形式：

| 事实数 | ICL（全部进 prompt） | 混合：MaxSim top-5 + ICL | MaxSim 路由 top-1 | 均值路由 top-1 |
|---|---|---|---|---|
| 400 | 93%（7,865 token） | **94%**（约 110 token） | 80.5% | 13.5% |
| 800 | 85%（15,807 token） | 83%（约 110 token） | 74.0% | 10.7% |

![修好门控之后](results/fig_routing_fix.png)

两件事变了。第一，门控是可修的：端到端从 13% 回到 94%（400 条）。第二，之前那个结论**诊断错了**——这套架构从来不是「权重记忆不 scalable」输掉的，是**我给它的检索器太差**。换成 late interaction 之后，它在 400 条与 ICL 打平（94% vs 93%），800 条只差 2 个点（83% vs 85%），而 prompt 里只有**约 110 个 token**，不是 7,865–15,807。

真正的卖点是这个 token 账，不是准确率：答案一样，上下文只占 1%，记忆存在权重里而不是每次请求里。

**再补一个对照，因为它改变了建议。** 同样的问句检索任务，四种检索器：

| 检索器 | N=400 top-1 | N=800 top-1 | 单次耗时 |
|---|---|---|---|
| BM25（纯词面，无需模型） | 100% | 100% | ~0.1 ms |
| bge-small-en-v1.5（33M 参数） | 98.8% | 97.4% | ~0.5 ms |
| Qwen2.5-7B late interaction | 80.5% | 74.0% | ~2 ms |
| Qwen2.5-7B mean-pool（原方案） | 13.5% | 10.7% | ~2 ms |

一个 33M 的检索器比 7B 的隐状态高 20 多个点，纯词面的 BM25 直接满分。所以结论不是「训练 key encoder」，而是**别拿冻结语言模型的池化隐状态当检索器**。注意前提：这里每个问句都带着实体名，正是 BM25 能吃到的信息；如果问句把实体名换掉，词面匹配会崩，那时才需要训练检索器。
**注意**：单模型、单种子、每个实体只有一条合成事实；问句只有六种模板，所以这些数字是真实问句下的上界。800 条时用 7B 隐状态做检索会重新成为瓶颈（top-1 74%），但换个 33M 的检索器就解决了——所以要委托检索，而不是把 key 训练得更大。

### 原实验那个 44/50 到底测的是什么

按原配方训练真实的 per-fact 适配器（rank 8，100 步），然后用四种方式评：

| 事实数 | 提问方式 | prompt 格式 | 召回 |
|---|---|---|---|
| 10 | 事实的前缀（`... uses a passcode:`） | 原始文本 | **7/10** |
| 10 | 同上 | 对话模板 | 2/10 |
| 50 | 真正的问句（`What passcode does the X protocol use?`） | 原始文本 | 8/50 |
| 50 | 真正的问句 | 对话模板 | 8/50 |

训练 100 步的适配器，**只有在 query 是它背下来那句话的前缀时**才复现原结果（7/10，对应 50 条时的 44/50）；换成关于同一条事实的真问句，掉到 16%。

所以原表里的海马体召回是一个**句子补全**指标，不是问答指标：适配器把字符串记住了，被前缀一激就接着往下写。这是真实的能力（事实确实进了权重），但它和「能回答问题」是两件事，原文把两者混在了一起。

两个后果。第一，§12 那个混合管线才是可用的形态：先检索出事实，再放进上下文读。第二，对话模板那一行提醒我们：在原始文本上训练的适配器，一旦套上对话模板就出了分布。

### 三个能带走的结论

1. **海马体侧可扩展**：逐条 LoRA + 余弦门控从 6 条扩到 50 条，仍有 44/50 召回、47/50 路由。
2. **顺序学习会遗忘**：单共享 LoRA 顺序学 50 条只剩 1 条，和 6 条版的 1/6 一致。
3. **单适配器巩固是当前配方的瓶颈**：batch=1、固定顺序、rank-32、150 epoch 下 0/50；拆成 9 个 rank-8 适配器后组内恢复到 50/50。
4. **天花板是检索器，不是架构。** 把整句话压成一个向量才是路由崩掉的原因（53–59%）；换成 late interaction 就恢复了（前缀式 query 100%，真问句 74–80%），再把 top-5 短名单喂回 prompt，400 条时与 ICL 打平（94% vs 93%），800 条只差 2 个点（83% vs 85%）——而 prompt 只用了约 110 个 token，不是 8k–16k。

### 这些结论**不**支持什么

- 分组 50/50 是**按已知分组选择适配器**后测的（oracle routing），**没有**验证“只凭问题内容在 9 个语义适配器之间路由”的端到端准确率。
- 单适配器 0/50 是**这套训练配方**的失败；没做 rank sweep，也没做“50 条同时进 batch 的联合训练”对照，不能直接断言是容量上限。
- ICL 基线是 **2026-10-22 补跑**的（晚于原始数字），单种子单模型；RAG 等检索对照仍然缺失。本实验关心的是「把记忆写进权重」，不是「给定 context 能否答对」——而补上基线之后，这个区分对适配器方案更不利了。
- 单次运行、召回为子串匹配；表中数字来自**未固定随机种子**的原始运行（脚本现已默认 `SEED=42`，但未重跑）。完整局限见报告 §8。

---

## 仓库结构

```
.
├── README.md
├── LICENSE                  # 代码：MIT
├── CITATION.cff
├── requirements.txt
├── setup.sh                 # AutoDL / Linux 环境搭建
├── docs/
│   ├── report.md            # 实验报告（中文）
│   └── LICENSE-CC-BY-4.0.md # 报告：CC BY 4.0
├── src/
│   ├── facts.py                 # 所有脚本共用的虚构事实集合（6 / 9 / 50 条）
│   ├── cls_demo.py              # 0.5B，本地 CPU 版
│   ├── cls_consolidation_test.py
│   ├── cls_generative_replay.py
│   ├── cls_demo_7b.py           # 7B，6 条事实完整流程
│   ├── cls_demo_7b_50.py        # 7B，50 条事实完整流程
│   ├── baseline_icl.py          # ICL 基线（脚本已提供，尚未运行）
│   ├── consolidate_grouped.py   # 分组巩固（9 × rank-8）
│   └── diagnostics/             # 4-bit / 8-bit / 巩固阶段的诊断脚本
└── results/
    ├── README.md            # 原始日志说明
    ├── summary.csv          # 五阶段结果（摘自报告）
    ├── loss_curves.csv      # 单适配器 / 分组巩固 loss（摘自报告）
    ├── plot_results.py      # 由 CSV 生成下面两张图
    ├── fig_recall.png
    └── fig_consolidation_loss.png
```

---

## 复现

### 硬件与环境

- NVIDIA RTX 4090 24GB（50 条版；12GB 卡会 OOM）；0.5B 版可在 CPU 跑。
- Python 3.10+；`torch 2.5.1+cu124`、`transformers 4.57.6`、`peft 0.21.2`、`bitsandbytes 0.50.2`。
- ⚠️ `transformers` 必须 `<5.0`：5.x 依赖 `torch.accelerator`，与 torch 2.5.1 不兼容。

```bash
bash setup.sh              # 安装依赖 + 下载 Qwen2.5-7B-Instruct
export HF_ENDPOINT=https://hf-mirror.com   # 大陆网络
export HF_HOME=/root/autodl-tmp/hf         # AutoDL 数据盘，按需修改
```

### 运行

```bash
python src/cls_demo.py            # 0.5B CPU 演示
python src/cls_demo_7b.py         # 7B，6 条事实
python src/cls_demo_7b_50.py      # 7B，50 条事实（约 57 分钟）
python src/consolidate_grouped.py # 分组巩固（约 24 分钟）
python src/baseline_icl.py        # ICL 基线：50 条塞进 prompt，不训练
```

事实集合统一放在 `src/facts.py`（50 条版两个脚本共用 `FACTS_50`）；随机种子默认 42，可用 `SEED=123 python src/...` 覆盖（bitsandbytes / CUDA 部分算子本身不确定，固定种子能减小波动，但不保证逐位一致）；`GROUP_SIZE`、`RANK`、`EPOCHS`、`LR_SLOW` 可在文件顶部调整。关键超参：海马体 `rank=8, LR=1e-3, 100 步/条`；巩固 `LR_SLOW=5e-4, 150 epoch`（`1e-3` 会导致适配器内部互相覆盖）。

---

## 已知坑：PEFT 包装器的显存泄漏

`get_peft_model(base, ...)` 返回的 `LoraModel` 仍通过 `self.model` 引用底层 `base`。Step 1 结束只 `del model` 不 `del base`，19GB 显存不释放，Step 2 重新加载底座时 OOM。修复：

```python
del model, base, keys, queries
```

并设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

---

## 相关工作

- **互补学习系统** — McClelland, McNaughton & O'Reilly (1995), *Psychological Review* 102(3)；Kumaran, Hassabis & McClelland (2016), *TiCS* 20(7)。本文借用的双系统框架。
- **灾难性遗忘** — French (1999), *TiCS* 3(4)；Kirkpatrick et al. (2017), EWC, *PNAS* 114(13)。阶段③在适配器上的复现。
- **LoRA** — Hu et al. (2021), arXiv:2106.09685。被当作记忆槽使用的适配器本体。
- **适配器库与混合** — Zadouri et al. (2023), MoLE, arXiv:2309.05444；Huang et al. (2024), LoraHub, arXiv:2307.13269。最接近的先行工作：多个小适配器 + 一个路由器。区别在于本实验的适配器装的是**情景记忆**而非**技能**，并且显式测试了巩固阶段。
- **检索与上下文记忆** — Lewis et al. (2020), RAG, arXiv:2005.11401；Brown et al. (2020), arXiv:2005.14165；Packer et al. (2023), MemGPT, arXiv:2310.08560。本文**尚未跑**的对照基线。

## 引用 / Citation

见 [`CITATION.cff`](CITATION.cff)。如果这个仓库对你有帮助，欢迎开 issue 讨论实验设计或指出错误。

## License

- 代码：MIT（[`LICENSE`](LICENSE)）
- 报告与文档：CC BY 4.0（[`docs/LICENSE-CC-BY-4.0.md`](docs/LICENSE-CC-BY-4.0.md)）
- 模型权重不随仓库分发；Qwen2.5 使用 Apache-2.0。
