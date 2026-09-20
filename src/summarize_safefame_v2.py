"""Create audited SafeFAME-TS v2 tables and publication-ready figures."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plot_style import configure_fonts, finish_fonts


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs" / "safefame_v2"
TABLES = ROOT / "outputs" / "tables" / "v2"
FIGURES = ROOT / "outputs" / "figures" / "v2"


def save_figure(fig: plt.Figure, name: str) -> None:
    finish_fonts(fig)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(figures_only: bool = False) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(INPUT / "safefame_v2_metrics.csv")
    audit = pd.read_csv(INPUT / "safefame_v2_selection_audit.csv")
    if len(audit) != 18 or len(metrics) != 72:
        raise ValueError("Unexpected SafeFAME-TS v2 task count")

    candidates = metrics[metrics.model.isin(["semantic_residual", "frequency_residual"])].copy()
    candidates["ci_conclusion"] = np.select(
        [candidates.block_ci_low > 0, candidates.block_ci_high < 0],
        ["显著优于回退", "显著劣于回退"], default="区间跨零",
    )
    if not figures_only:
        candidates.to_csv(TABLES / "candidate_test_results.csv", index=False)

    summary = audit[[
        "domain", "horizon", "numeric_fallback", "calibration_windows",
        "decision_windows", "test_windows", "selected_path",
        "semantic_residual_decision_mse", "semantic_residual_permutation_p_value",
        "semantic_residual_segment_wins", "frequency_residual_decision_mse",
        "frequency_residual_permutation_p_value", "frequency_residual_segment_wins",
    ]].copy()
    if not figures_only:
        summary.to_csv(TABLES / "selection_results.csv", index=False)

    semantic = candidates[candidates.model.eq("semantic_residual")]
    frequency = candidates[candidates.model.eq("frequency_residual")]
    claims = {
        "tasks": int(len(audit)),
        "selected_text_tasks": int(audit.selected_path.ne("numeric_fallback").sum()),
        "semantic_point_improvement_tasks": int((semantic.improvement_vs_fallback_pct > 0).sum()),
        "semantic_significant_improvement_tasks": int((semantic.block_ci_low > 0).sum()),
        "semantic_significant_harm_tasks": int((semantic.block_ci_high < 0).sum()),
        "frequency_point_improvement_tasks": int((frequency.improvement_vs_fallback_pct > 0).sum()),
        "frequency_significant_improvement_tasks": int((frequency.block_ci_low > 0).sum()),
        "frequency_significant_harm_tasks": int((frequency.block_ci_high < 0).sum()),
        "permutation_passes_at_0_025": int(sum(
            (audit[f"{variant}_permutation_p_value"] <= 0.025).sum()
            for variant in ("semantic_residual", "frequency_residual")
        )),
        "segment_stability_passes": int(sum(
            audit[f"{variant}_segment_wins"].astype(bool).sum()
            for variant in ("semantic_residual", "frequency_residual")
        )),
        "total_candidate_pathways": 36,
        "test_windows": int(audit.test_windows.sum()),
        "calibration_windows": int(audit.calibration_windows.sum()),
        "decision_windows": int(audit.decision_windows.sum()),
    }
    if not figures_only:
        (TABLES / "verified_claims.json").write_text(
            json.dumps(claims, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False})
    configure_fonts()
    palette = {"semantic_residual": "#2F6690", "frequency_residual": "#D97706"}

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    plot = candidates.copy()
    plot["task"] = plot.domain + " H" + plot.horizon.astype(str)
    x = np.arange(18)
    width = 0.38
    for offset, model in [(-width / 2, "semantic_residual"), (width / 2, "frequency_residual")]:
        block = plot[plot.model.eq(model)]
        ax.bar(x + offset, block.improvement_vs_fallback_pct, width, label=model.replace("_residual", ""), color=palette[model])
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.set_xticks(x, plot[plot.model.eq("semantic_residual")].task, rotation=55, ha="right")
    ax.set_ylabel("相对验证选定数值回退的测试 MSE 改善 / %")
    ax.set_title("文本候选路径的跨任务收益与负迁移")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "fig_v2_candidate_gains")

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    sem = semantic.copy()
    sem["task"] = sem.domain + " H" + sem.horizon.astype(str)
    fallback_mse = sem.mse / (1.0 - sem.improvement_vs_fallback_pct / 100.0)
    sem["relative_delta"] = 100.0 * sem.loss_difference_vs_fallback / fallback_mse
    sem["relative_low"] = 100.0 * sem.block_ci_low / fallback_mse
    sem["relative_high"] = 100.0 * sem.block_ci_high / fallback_mse
    sem = sem.sort_values("relative_delta")
    y = np.arange(len(sem))
    lower = sem.relative_delta - sem.relative_low
    upper = sem.relative_high - sem.relative_delta
    colors = np.where(sem.block_ci_low > 0, "#287D5A", np.where(sem.block_ci_high < 0, "#B5473C", "#78828A"))
    for i, color in enumerate(colors):
        ax.errorbar(
            sem.relative_delta.iloc[i], y[i],
            xerr=[[lower.iloc[i]], [upper.iloc[i]]], fmt="none",
            ecolor=color, capsize=3, alpha=0.9,
        )
    ax.scatter(sem.relative_delta, y, c=colors, s=30, zorder=3)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_yticks(y, sem.task)
    ax.set_xlabel("相对 MSE 损失差 / %：回退 − 语义候选（95%移动块Bootstrap区间）")
    ax.set_title("语义文本路径的不确定性：正值表示文本更优")
    ax.grid(axis="x", alpha=0.2)
    save_figure(fig, "fig_v2_semantic_block_ci")

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    p_sem = audit.semantic_residual_permutation_p_value.to_numpy()
    p_freq = audit.frequency_residual_permutation_p_value.to_numpy()
    task = audit.domain + " H" + audit.horizon.astype(str)
    ax.scatter(np.arange(18) - 0.12, p_sem, label="semantic", color=palette["semantic_residual"], s=32)
    ax.scatter(np.arange(18) + 0.12, p_freq, label="frequency", color=palette["frequency_residual"], s=32)
    ax.axhline(0.025, color="#B5473C", linestyle="--", linewidth=1.2, label="Bonferroni 阈值 0.025")
    ax.set_xticks(np.arange(18), task, rotation=55, ha="right")
    ax.set_ylabel("验证期文本置换检验 p 值")
    ax.set_title("文本路径启用前的反事实门槛")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "fig_v2_permutation_gate")

    counts = audit.numeric_fallback.value_counts().reindex(["Last", "SeasonalNaive", "AR-Ridge", "DLinear-M", "PatchTST"], fill_value=0)
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.bar(counts.index, counts.values, color=["#8B95A1", "#557A95", "#2F6690", "#D97706", "#287D5A"])
    ax.set_ylabel("被验证期校准选中的任务数")
    ax.set_title("不存在跨领域通吃的数值专家")
    ax.set_ylim(0, max(counts.values) + 1.5)
    for i, value in enumerate(counts.values):
        ax.text(i, value + 0.12, str(value), ha="center")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, "fig_v2_fallback_experts")

    print(json.dumps(claims, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--figures-only', action='store_true')
    main(parser.parse_args().figures_only)
