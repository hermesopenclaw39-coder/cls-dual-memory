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
| `icl_scaling.csv` | ICL 规模扫描：事实数 50→1600 的 prompt token、召回、耗时 |
| `fig_icl_scaling.png` | 上表的折线图（召回 vs prompt 长度） |
| `logs/icl_scale_sweep.log` | 规模扫描原始输出（2026-10-22） |

注意：分组巩固的召回是 oracle 分组（按已知组选择适配器）下的组内结果，不是端到端路由准确率。
