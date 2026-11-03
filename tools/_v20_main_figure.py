# -*- coding: utf-8 -*-
"""Main figures for the SLC-MARS (frozen v20) results: two figures, (a)(b) and (c)(d) side by side."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_BASE = "#9ecae1"   # baselines
COLOR_SLC    = "#d6604d" # SLC-MARS (highlight)

# ---------------- (a) Main results: accuracy over chance -------------------
tasks = ["A·Mem", "A·Pair", "B·Mem", "B·Pair"]
systems = ["LLM", "LitRAG", "AutoGen", "MDAgents", "MetaGPT", "SLC-MARS"]
acc = np.array([
    [56.1, 70.7, 27.5, 40.2],   # pure_llm
    [50.9, 64.1, 31.2, 49.6],   # LitRAG
    [62.6, 68.3, 43.5, 48.0],   # AutoGen
    [33.9, 27.8, 34.1, 21.3],   # MDAgents
    [60.9, 71.4, 42.8, 51.2],   # MetaGPT
    [67.8, 73.7, 56.5, 52.8],   # SLC-MARS (best of heuristic/auto)
])
# ========================= Figure 1: (a) + (b), horizontal ==================
fig1, axes1 = plt.subplots(2, 2, figsize=(11, 10))
ax = axes1[0,0]
x = np.arange(len(tasks))
w = 0.15
for i, name in enumerate(systems):
    c = COLOR_SLC if name == "SLC-MARS" else COLOR_BASE
    e = [x + i * w - 2.5 * w, acc[i, :], w]
    if name == "SLC-MARS":
        ax.bar(*e, color=c, edgecolor="black", linewidth=0.6, label=name, zorder=3)
    else:
        ax.bar(*e, color=c, label=name)
ax.axhline(50, color="gray", ls="--", lw=1)
ax.set_xlim(-0.9, 3.5)
ax.text(-0.4, 50.5, "chance", color="gray", fontsize=7.5, ha="right", va="bottom")
ax.set_xticks(x)
ax.set_xticklabels(tasks)
ax.set_ylim(0, 95)
ax.set_ylabel("Accuracy (%)")
ax.set_title("(a) Main results: decision tasks")
ax.legend(ncol=3, fontsize=6.5, loc="upper left", framealpha=0.9)

# ---------------- (b) TopK ranking: NDCG@5 + hit@5 -------------------------
ax = axes1[0,1]
names = ["LLM", "MetaGPT", "AutoGen", "MDAgents", "LitRAG",
         "SLC-MARS\n(graph pool)", "SLC-MARS\n(LLM pool)"]
ndcg = [0.3587, 0.3357, 0.3553, 0.1972, 0.3086, 0.3838, 0.1349]
hit  = [0.848, 0.818, 0.818, 0.273, 0.667, 0.909, 0.545]
colors = [COLOR_BASE] * 5 + ["#b30000", "#999999"]
y = np.arange(len(names))[::-1]
ax.barh(y - 0.2, ndcg, height=0.4, color=colors, label="NDCG@5")
ax.barh(y + 0.2, hit, height=0.4, color=colors, alpha=0.45, hatch="//", label="hit@5")
for yi, v in zip(y, ndcg):
    ax.text(v + 0.01, yi - 0.2, f"{v:.3f}", va="center", fontsize=7)
ax.set_yticks(y)
ax.set_yticklabels(names, fontsize=7.5)
ax.set_xlim(0, 1.05)
ax.set_xlabel("Score (higher is better)")
ax.set_title("(b) TopK ranking (11 CRISPR questions)")
ax.legend(fontsize=7.5, loc="lower right")

# ========================= Figure 2: (c) + (d), horizontal ==================
ax = axes1[1,0]
pools = ["Full", "w/o knowledge agent", "w/o computing agent"]
graph = [0.3460, 0.3631, 0.3704]
llmp  = [0.1187, 0.1349, 0.1315]
x = np.arange(len(pools))
b1 = ax.bar(x - 0.18, graph, 0.36, color="#74c476", label="Graph candidate pool",
            edgecolor="black", linewidth=0.5)
b2 = ax.bar(x + 0.18, llmp, 0.36, color="#fdae6b", label="LLM self-generated pool",
            edgecolor="black", linewidth=0.5)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
            f"{b.get_height():.3f}", ha="center", fontsize=7)
for i in range(len(pools)):
    ax.annotate("", xy=(x[i] + 0.10, llmp[i] + 0.015),
                xytext=(x[i] - 0.10, graph[i] - 0.015),
                arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
    ax.text(x[i], 0.46, f"\u2212{graph[i]-llmp[i]:.2f}", color="red",
            ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(pools, fontsize=8)
ax.set_ylim(0, 0.62)
ax.set_ylabel("NDCG@5")
ax.set_title("(c) Pool ablation")
ax.legend(fontsize=7.5, loc="upper right")

# ---------------- (d) Agent ablation: overall accuracy ---------------------
ax = axes1[1,1]
configs = ["SLC-MARS\nfull", "w/o knowledge agent", "w/o computing agent"]
m_acc = [69.8, 56.6, 67.9]
p_acc = [76.6, 61.7, 61.7]
a_acc = [73.0, 59.0, 65.0]
x = np.arange(len(configs))
ax.bar(x - 0.26, m_acc, 0.25, color="#6baed6", label="Membership")
ax.bar(x,        p_acc, 0.25, color="#4292c6", label="Pairwise")
ax.bar(x + 0.26, a_acc, 0.25, color="#08519c", label="All tasks")
for xi, (mv, pv, av) in enumerate(zip(m_acc, p_acc, a_acc)):
    for dx, v in [(-0.26, mv), (0, pv), (0.26, av)]:
        ax.text(xi + dx, v + 0.8, f"{v:.1f}", ha="center", fontsize=7)
ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=8)
ax.set_ylim(0, 88)
ax.set_ylabel("Accuracy (%)")
ax.set_title("(d) Agent ablation in decision tasks")
ax.legend(fontsize=7.5, loc="upper right")

# ---------------- save both figures ----------------
OUTDIR = str(Path(__file__).resolve().parents[1] / "paper" / "figures")
for f, name in [(fig1, "fig_ab_components")]:
    f.tight_layout(pad=1.5)
    pdf = rf"{OUTDIR}\{name}.pdf"
    f.savefig(pdf, dpi=300, bbox_inches="tight")
    f.savefig(pdf.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print("saved:", pdf)
