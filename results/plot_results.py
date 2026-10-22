# -*- coding: utf-8 -*-
"""Draw the two README figures from results/*.csv.

Usage:  python results/plot_results.py
Output: results/fig_recall.png, results/fig_consolidation_loss.png
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE = "#2a78d6"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

LABELS = {
    "frozen_base": "Frozen base",
    "hippocampus_gated": "Hippocampus (50 LoRAs + gate)",
    "shared_lora_sequential": "Shared LoRA, sequential",
    "single_adapter_consolidation": "Single-adapter consolidation (r=32)",
    "grouped_consolidation_oracle": "Grouped consolidation (9 x r=8)*",
    "icl_baseline_max12": "In-context baseline (12-token budget)",
    "icl_baseline_max24": "In-context baseline (24-token budget)",
}


def read(name):
    with open(os.path.join(HERE, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def fig_recall():
    rows = read("summary.csv")[::-1]          # top-to-bottom = pipeline order
    names = [LABELS[r["stage"]] for r in rows]
    vals = [int(r["recall"]) for r in rows]
    total = int(rows[0]["total"])

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.barh(names, vals, height=0.55, color=BLUE)
    for y, v in enumerate(vals):
        ax.text(v + 0.8, y, "%d/%d" % (v, total), va="center", color=INK, fontsize=10)
    ax.set_xlim(0, total + 6)
    ax.set_xlabel("Recall (facts answered correctly, out of %d)" % total)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style(ax)
    ax.set_title("Recall by stage - Qwen2.5-7B-Instruct, 50 synthetic facts",
                 loc="left", color=INK, fontsize=12)
    fig.text(0.01, 0.01, "* evaluated with oracle group selection (not end-to-end routing)",
             color=INK2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(HERE, "fig_recall.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_loss():
    rows = read("loss_curves.csv")
    single = [r for r in rows if r["phase"] == "single_adapter"]
    grouped = [r for r in rows if r["phase"] == "grouped"]
    ep = [int(r["unit"].split("_")[1]) for r in single]
    sl = [float(r["loss"]) for r in single]
    gn = [r["unit"].split("_")[1] for r in grouped]
    gl = [float(r["loss"]) for r in grouped]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True,
                                 gridspec_kw={"width_ratios": [1.15, 1]})
    a1.plot(ep, sl, color=BLUE, linewidth=2, marker="o", markersize=5)
    a1.set_title("Single adapter (rank 32, all 50 facts)", loc="left", color=INK, fontsize=11)
    a1.set_xlabel("Epoch")
    a1.set_ylabel("Training loss")
    a1.annotate("recall 0/50", xy=(ep[-1], sl[-1]), xytext=(-70, 12),
                textcoords="offset points", color=INK2, fontsize=9)

    a2.bar(gn, gl, width=0.6, color=BLUE)
    a2.set_title("Grouped (9 adapters x rank 8), final loss", loc="left", color=INK, fontsize=11)
    a2.set_xlabel("Group (group 8 has 2 facts)")
    a2.text(0.98, 0.92, "recall 50/50 (oracle groups)", transform=a2.transAxes,
            ha="right", color=INK2, fontsize=9)

    for ax in (a1, a2):
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        style(ax)
    a1.set_ylim(0, 5.6)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_consolidation_loss.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out



def fig_icl_scaling():
    rows = read("icl_scaling.csv")
    toks = [int(r["prompt_tokens"]) for r in rows]
    rec = [100.0 * int(r["hits"]) / int(r["probes"]) for r in rows]
    ns = [r["n_facts"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    ax.plot(toks, rec, color=BLUE, linewidth=2, marker="o", markersize=6, zorder=3)
    for x, y, n in zip(toks, rec, ns):
        dx, dy = (-3, 12) if int(n) > 1000 else (0, 10)
        ax.annotate("%s facts" % n, (x, y), textcoords="offset points", xytext=(dx, dy),
                    ha="right" if int(n) > 1000 else "center", color=INK2, fontsize=8.5)
    ax.axvline(32768, color="#c0392b", linestyle="--", linewidth=1.2, zorder=2)
    ax.text(32768 * 0.97, 8, "Qwen2.5-7B context window (32,768 tokens)", color="#c0392b",
            fontsize=8.5, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("Prompt size (tokens, log scale) - labelled by fact count")
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style(ax)
    ax.set_title("In-context baseline degrades as the fact set grows", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_icl_scaling.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_scale_compare():
    rows = read("scale_comparison.csv")
    n = [int(r["n_facts"]) for r in rows]
    icl = [100.0 * int(r["icl_hits"]) / int(r["icl_probes"]) for r in rows]
    top1 = [float(r["routing_top1_pct"]) for r in rows]
    top5 = [float(r["routing_top5_pct"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.plot(n, icl, color=BLUE, linewidth=2, marker="o", markersize=6, label="In-context (all facts in prompt)")
    ax.plot(n, top1, color="#c0392b", linewidth=2, marker="s", markersize=5, label="Routing top-1 (cosine gate)")
    ax.plot(n, top5, color="#c0392b", linewidth=1.2, linestyle="--", marker="^", markersize=4,
            label="Routing top-5")
    ax.set_xscale("log")
    ax.set_xticks(n)
    ax.set_xticklabels([str(x) for x in n])
    ax.set_xlabel("Facts in memory")
    ax.set_ylabel("Recall / routing accuracy (%)")
    ax.set_ylim(40, 105)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    style(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("Two different ceilings: prompt capacity vs retrieval", loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    out = os.path.join(HERE, "fig_scale_compare.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out

if __name__ == "__main__":
    print(fig_recall())
    print(fig_loss())
    print(fig_icl_scaling())
    print(fig_scale_compare())
