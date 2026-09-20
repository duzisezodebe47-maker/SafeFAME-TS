"""Redraw the historical residual-ablation figure with accurate terminology."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import configure_fonts, finish_fonts


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "reviewer_sensitivity" / "reviewer_sensitivity_results.csv"
OUTPUT = ROOT / "outputs" / "reviewer_sensitivity" / "fig_numeric_residual_ablation.png"


def main() -> None:
    frame = pd.read_csv(SOURCE)
    task_order = list(frame[["domain", "horizon"]].drop_duplicates().itertuples(index=False, name=None))
    configure_fonts()
    colors = {"semantic_residual": "#2878B5", "frequency_residual": "#D95F02"}
    titles = {"semantic_residual": "语义残差候选", "frequency_residual": "频率语义残差候选"}
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 7.2), sharey=True)
    for ax, variant in zip(axes, titles, strict=True):
        subset = frame[frame.variant.eq(variant)].set_index(["domain", "horizon"]).loc[task_order].reset_index()
        positions = np.arange(len(subset))
        ax.errorbar(subset.test_gain_vs_matched_control_pct, positions,
                    xerr=np.vstack([subset.test_gain_vs_matched_control_pct - subset.test_gain_ci_low_pct,
                                    subset.test_gain_ci_high_pct - subset.test_gain_vs_matched_control_pct]),
                    fmt="o", capsize=2.5, color=colors[variant])
        ax.axvline(0, color="#555555", linewidth=1)
        ax.set_yticks(positions, [f"{d} H{h}" for d, h in task_order])
        ax.set_xlabel("相对纯数值残差消融的MSE改善（%）")
        ax.set_title(titles[variant])
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("测试期文本相关增量诊断", fontsize=15)
    finish_fonts(fig)
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=240)
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
