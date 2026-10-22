# CLS Dual-Memory: LoRA as Memory Slots

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

### 三个能带走的结论

1. **海马体侧可扩展**：逐条 LoRA + 余弦门控从 6 条扩到 50 条，仍有 44/50 召回、47/50 路由。
2. **顺序学习会遗忘**：单共享 LoRA 顺序学 50 条只剩 1 条，和 6 条版的 1/6 一致。
3. **单适配器巩固是当前配方的瓶颈**：batch=1、固定顺序、rank-32、150 epoch 下 0/50；拆成 9 个 rank-8 适配器后组内恢复到 50/50。

### 这些结论**不**支持什么

- 分组 50/50 是**按已知分组选择适配器**后测的（oracle routing），**没有**验证“只凭问题内容在 9 个语义适配器之间路由”的端到端准确率。
- 单适配器 0/50 是**这套训练配方**的失败；没做 rank sweep，也没做“50 条同时进 batch 的联合训练”对照，不能直接断言是容量上限。
- 没有和 in-context learning / RAG 比较。本实验关心的是“把记忆写进权重”，不是“给定 context 能否答对”。
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

## 引用 / Citation

见 [`CITATION.cff`](CITATION.cff)。如果这个仓库对你有帮助，欢迎开 issue 讨论实验设计或指出错误。

## License

- 代码：MIT（[`LICENSE`](LICENSE)）
- 报告与文档：CC BY 4.0（[`docs/LICENSE-CC-BY-4.0.md`](docs/LICENSE-CC-BY-4.0.md)）
- 模型权重不随仓库分发；Qwen2.5 使用 Apache-2.0。
