# -*- coding: utf-8 -*-
"""Regenerate figure6 (semantic fallback CI) and figure11 (circular shift p)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from pathlib import Path
from plot_style import configure_fonts, finish_fonts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/figures"
C = {"navy":"#234E70","blue":"#3B82C4","cyan":"#58B7C5","orange":"#E58B3A",
     "red":"#C84B4B","green":"#4D9078","gray":"#7A8793"}
plt.rcParams.update({"font.sans-serif":["Microsoft YaHei","SimHei","DejaVu Sans"],
                     "axes.unicode_minus":False,"savefig.dpi":300,"font.size":11})

configure_fonts()
# ---------- Figure 6: semantic residual, fallback vs candidate moving-block CI ----------
ct = pd.read_csv(ROOT/"outputs/tables/v2/candidate_test_results.csv")
s = ct[ct["model"]=="semantic_residual"].copy()
s["task"] = s["domain"]+" H"+s["horizon"].astype(str)
# point = improvement vs fallback (%); positive favors text
s = s.sort_values("improvement_vs_fallback_pct", ascending=True).reset_index(drop=True)  # top = largest
def color_row(row):
    lo, hi = row["block_ci_low"], row["block_ci_high"]
    if lo > 0 and hi > 0: return C["green"]
    if lo < 0 and hi < 0: return C["red"]
    return C["gray"]
s["color"] = s.apply(color_row, axis=1)

fig, ax = plt.subplots(figsize=(11, 7.2))
y = np.arange(len(s))
fallback = s["mse"]/(1-s["improvement_vs_fallback_pct"]/100.0)
left = (s["improvement_vs_fallback_pct"] - 100*s["block_ci_low"]/fallback).values
right = (100*s["block_ci_high"]/fallback - s["improvement_vs_fallback_pct"]).values
pts = s["improvement_vs_fallback_pct"].values
for yi, p, l, r, col in zip(y, pts, left, right, s["color"]):
    ax.errorbar(p, yi, xerr=[[l],[r]], fmt="o", ms=7, ecolor=col, color=col,
                elinewidth=1.6, capsize=4, zorder=2)
ax.axvline(0, color="#333333", lw=1.0)
ax.set_yticks(y); ax.set_yticklabels(s["task"])
ax.set_xlabel("相对 MSE 损失差 / %：回退 − 语义候选（95% 移动块 Bootstrap 区间）")
ax.set_title("语义文本路径的不确定性：正值表示文本更优")
ax.grid(axis="x", ls="--", alpha=0.4)
fig.tight_layout()
finish_fonts(fig)
fig.savefig(OUT/"refig6.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)

# ---------- Figure 11: circular-shift permutation p values ----------
rs = pd.read_csv(ROOT/"outputs/reviewer_sensitivity/reviewer_sensitivity_results.csv")
order = [("Climate",4),("Climate",12),("Climate",24),
         ("Energy",4),("Energy",12),("Energy",24),
         ("Economy",3),("Economy",6),("Economy",12),
         ("Traffic",3),("Traffic",6),("Traffic",12),
         ("Agriculture",3),("Agriculture",6),("Agriculture",12),
         ("Security",3),("Security",6),("Security",12)]
lab = [f"{d} H{h}" for d,h in order]
def pivot(variant):
    sub = rs[rs["variant"]==variant].set_index(["domain","horizon"])["circular_shift_p_value"]
    return [sub[(d,h)] for d,h in order]
sem = pivot("semantic_residual"); freq = pivot("frequency_residual")
x = np.arange(len(lab)); w=0.4
fig, ax = plt.subplots(figsize=(11.6, 6.7))
ax.bar(x-w/2, sem, w, color=C["blue"], label="语义残差候选")
ax.bar(x+w/2, freq, w, color=C["orange"], label="频率语义残差候选")
ax.axhline(0.025, color=C["red"], ls="--", lw=1.6, label="门槛 0.025")
ax.set_xticks(x); ax.set_xticklabels(lab, rotation=40, ha="right", fontsize=9)
ax.set_ylabel("循环移位置换检验 p 值")
ax.set_title("历史 v2 的 999 次循环移位诊断（原混合求解器口径）")
ax.legend(loc="upper left", ncol=3, frameon=False)
ax.grid(axis="y", ls="--", alpha=0.4)
fig.tight_layout()
finish_fonts(fig)
fig.savefig(OUT/"refig11.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("done", len(s), "tasks fig6; fig11 sem/freq bars")
