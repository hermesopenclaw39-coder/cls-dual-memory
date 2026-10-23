# 结果文件说明

这里的 CSV 是从 `docs/report.md` 的表格整理的**派生数据**，用于画图或二次分析；不是原始日志。

原始运行日志在实验实例上生成：

- `exp.log` — 50 条事实完整流程（约 3453s）
- `exp_grouped.log` — 9 组分组巩固（约 1418s）

这两份日志未随仓库分发。

`logs/` 目录下是 2026-10-22 补跑的 ICL 基线日志（`icl_50_max12.log`、`icl_50_max24.log`），已随仓库发布：只含事实文本与模型输出，不含主机名或凭据。

文件：

| 文件 | 内容 |
|---|---|
| `summary.csv` | 五个阶段的召回与路由准确率 |
| `loss_curves.csv` | 单适配器巩固的 loss 曲线、分组巩固的最终 loss 与组内召回 |
| `plot_results.py` | 读取上面两个 CSV，生成 `fig_recall.png` 与 `fig_consolidation_loss.png`（`python results/plot_results.py`，需 matplotlib） |
| `logs/icl_50_*.log` | ICL 基线原始输出（12 / 24 token 两种预算） |
| `scale_comparison.csv` | **规模化对照**：多样化事实下 ICL 召回 vs 余弦门控的路由准确率 |
| `fig_scale_compare.png` | 上表的图：两条上限（prompt 容量 vs 检索） |
| `icl_scaling.csv` | ⚠️ 已废弃：统一句式事实的 ICL 扫描，仅作「同质数据会让检索失真」的反面样本保留 |
| `fig_icl_scaling.png` | 上表的折线图（召回 vs prompt 长度） |
| `logs/icl_scale_sweep.log` | ⚠️ 同上，统一句式那版 |
| `logs/icl_scale_diverse.log` | 多样化事实的 ICL 扫描原始输出 |
| `logs/routing_scale_diverse.log` | 多样化事实的路由扫描原始输出 |
| `logs/routing_hand50.log` | 手写 50 条的路由对照（94%，用于校验实现） |
| `routing_comparison.csv` | **门控修复对照**：四种打分 + 混合管线，前缀式与问句式两种 query |
| `fig_routing_fix.png` | 门控修复后的对比图（真实问句） |
| `logs/routing_improve_prefix.log` | 四种打分在前缀式 query 下的原始输出 |
| `logs/routing_paraphrase.log` | 四种打分在问句式 query 下的原始输出 |
| `logs/icl_paraphrase.log` | 问句 query 下的 ICL 对照 |
| `hippocampus_completion.csv` | **真实 per-fact LoRA**（已修 bug）：单形态训练 vs 多形态训练 |
| `logs/hippocampus_fixed.log` | 修正后的原始输出（46/50 与 29/50） |
| `logs/hippocampus_paraphrase_50.log` | ⚠️ 旧版（有 bug）的 8/50，仅作对照保留 |

注意：分组巩固的召回是 oracle 分组（按已知组选择适配器）下的组内结果，不是端到端路由准确率。
