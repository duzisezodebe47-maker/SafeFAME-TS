"""Generate consistent high-resolution figures for the course-design report."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns

from data_utils import load_time_ordered_frame


COLORS = {
    "navy": "#234E70",
    "blue": "#3B82C4",
    "cyan": "#58B7C5",
    "orange": "#E58B3A",
    "red": "#C84B4B",
    "green": "#4D9078",
    "gray": "#7A8793",
}


def configure() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
        }
    )


def save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pipeline(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    boxes = [
        (0.3, 2.0, 2.1, 1.2, "数值序列审计\n稳定时间排序", COLORS["navy"]),
        (2.9, 3.4, 2.1, 1.2, "严格时点文本\n排除 preds/pred", COLORS["blue"]),
        (2.9, 0.6, 2.1, 1.2, "数值专家\nLast / Ridge / PatchTST", COLORS["cyan"]),
        (5.6, 3.4, 2.2, 1.2, "MiniLM语义与\n10项质量特征", COLORS["orange"]),
        (5.6, 0.6, 2.2, 1.2, "频谱统计与\n频率—语义交互", COLORS["green"]),
        (8.5, 2.0, 2.0, 1.2, "反事实验证\n对齐 vs 置换", COLORS["red"]),
        (11.0, 2.0, 1.7, 1.2, "安全选择\n启用或回退", COLORS["navy"]),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=color, ec="none", alpha=0.95))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color="white", fontsize=11, weight="bold")
    arrows = [((2.4, 2.6), (2.9, 1.2)), ((2.4, 2.6), (2.9, 4.0)), ((5.0, 4.0), (5.6, 4.0)),
              ((5.0, 1.2), (5.6, 1.2)), ((7.8, 4.0), (8.5, 2.8)), ((7.8, 1.2), (8.5, 2.4)),
              ((10.5, 2.6), (11.0, 2.6))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14, color=COLORS["gray"], lw=1.8))
    ax.text(6.5, 5.0, "SafeFAME-TS：从防泄漏数据链到可证伪的安全融合", ha="center", fontsize=16, weight="bold", color=COLORS["navy"])
    ax.text(11.85, 1.55, "证据充分：融合\n证据不足：数值回退", ha="center", va="top", color=COLORS["gray"], fontsize=9)
    save(fig, output, "fig01_technical_route")


def coverage(root: Path, output: Path) -> None:
    data = pd.read_csv(root / "data_processed/text/point_in_time_coverage.csv")
    data = data[data["split"] == "test"].groupby("domain", as_index=False)[
        ["report_coverage_pct", "search_coverage_pct"]
    ].mean()
    long = data.melt("domain", var_name="source", value_name="coverage")
    long["source"] = long["source"].map({"report_coverage_pct": "Report", "search_coverage_pct": "Search"})
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(data=long, x="domain", y="coverage", hue="source", palette=[COLORS["blue"], COLORS["orange"]], ax=ax)
    ax.set(title="测试时段严格时点文本覆盖率", xlabel="领域", ylabel="覆盖率（%）", ylim=(0, 108))
    ax.legend(title="文本来源", ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.27))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", padding=2, fontsize=8)
    save(fig, output, "fig02_text_coverage")


def development_gain(root: Path, output: Path) -> None:
    data = pd.read_csv(root / "outputs/tables/final/table_development_multimodal.csv")
    data["task"] = data["domain"] + "-H" + data["horizon"].astype(str)
    values = data["aligned_semantic_mse"] / data["best_numeric_mse"]
    colors = [COLORS["green"] if value < 1 else COLORS["red"] for value in values]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(data["task"], values, color=colors)
    ax.axhline(1, color="#333333", lw=1)
    ax.set_yscale("log")
    ax.set_ylim(0.75, max(2.6, float(values.max()) * 1.08))
    ax.set(title="开发性回测：MiniLM早期融合相对最强数值基线的误差", xlabel="领域—预测跨度", ylabel="MSE比值（语义融合 / 最强数值）")
    ax.tick_params(axis="x", rotation=38)
    save(fig, output, "fig03_development_semantic_gain")


def alignment_test(root: Path, output: Path) -> None:
    data = pd.read_csv(root / "outputs/tables/final/table_development_multimodal.csv")
    matrix = data.pivot(index="domain", columns="horizon", values="alignment_gain_vs_shuffle_pct")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    sns.heatmap(matrix, annot=True, fmt=".1f", center=0, cmap="RdYlGn", linewidths=0.7, cbar_kws={"label": "对齐文本相对置换文本改善率（%）"}, ax=ax)
    ax.set(title="正确时点对齐是否优于随机置换文本", xlabel="预测跨度", ylabel="领域")
    save(fig, output, "fig04_alignment_vs_permutation")


def confirmation_safety(root: Path, output: Path) -> None:
    data = pd.read_csv(root / "outputs/tables/final/table_confirmation_safety.csv")
    data["task"] = data["domain"] + "-H" + data["horizon"].astype(str)
    ratios = pd.DataFrame(
        {
            "task": np.repeat(data["task"].to_numpy(), 3),
            "path": np.tile(["无频率语义", "频率—语义", "SafeFAME选择"], len(data)),
            "ratio": np.column_stack(
                [
                    data["semantic_mse"] / data["internal_numeric_mse"],
                    data["frequency_semantic_mse"] / data["internal_numeric_mse"],
                    data["selective_mse"] / data["internal_numeric_mse"],
                ]
            ).ravel(),
        }
    )
    fig, ax = plt.subplots(figsize=(11, 5.2))
    sns.barplot(data=ratios, x="task", y="ratio", hue="path", palette=[COLORS["orange"], COLORS["red"], COLORS["green"]], ax=ax)
    ax.axhline(1.0, color="#222222", lw=1.2, ls="--")
    ax.set(title="冻结确认实验：安全选择避免语义负迁移", xlabel="未见领域—预测跨度", ylabel="相对内部数值路径的MSE比值")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="候选路径", ncol=3, loc="upper center")
    save(fig, output, "fig05_confirmation_safety")


def ett_results(root: Path, output: Path) -> None:
    data = pd.read_csv(root / "outputs/ett_benchmark/ett_metrics_summary.csv")
    models = ["Last", "SeasonalNaive24", "ARRidge", "DLinear-M", "PatchTST-M"]
    palette = dict(zip(models, [COLORS["gray"], COLORS["orange"], COLORS["blue"], COLORS["cyan"], COLORS["navy"]]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for ax, dataset in zip(axes, ["ETTh1", "ETTh2"]):
        part = data[data["dataset"] == dataset]
        for model in models:
            series = part[part["model"] == model].sort_values("horizon")
            ax.plot(series["horizon"], series["mse_mean"], marker="o", lw=2, label=model, color=palette[model])
            if model in ("DLinear-M", "PatchTST-M"):
                ax.fill_between(series["horizon"], series["mse_mean"] - series["mse_std"].fillna(0), series["mse_mean"] + series["mse_std"].fillna(0), color=palette[model], alpha=0.12)
        ax.set(title=dataset, xlabel="预测跨度（小时）", ylabel="归一化MSE")
        ax.set_xticks([96, 336, 720])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=5, frameon=False)
    fig.suptitle("ETT外部单目标预测：不同数值模型随跨度的误差", y=0.99, fontsize=14, weight="bold")
    fig.subplots_adjust(top=0.84, bottom=0.19, wspace=0.20)
    save(fig, output, "fig06_ett_external_results")


def stability(root: Path, output: Path) -> None:
    data = pd.read_csv(root / "outputs/ett_benchmark/ett_metrics_summary.csv")
    data = data[data["model"].isin(["DLinear-M", "PatchTST-M"])].copy()
    data["task"] = data["dataset"] + "-H" + data["horizon"].astype(str)
    data["relative_std_pct"] = 100 * data["mse_std"] / data["mse_mean"]
    fig, ax = plt.subplots(figsize=(10, 4.7))
    sns.barplot(data=data, x="task", y="relative_std_pct", hue="model", palette=[COLORS["cyan"], COLORS["navy"]], ax=ax)
    ax.set(title="ETT三次随机种子稳定性", xlabel="数据集—预测跨度", ylabel="MSE变异系数（%）")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="模型")
    save(fig, output, "fig07_seed_stability")


def security_drift(root: Path, output: Path) -> None:
    frame, _ = load_time_ordered_frame(root / "references/external/Time-MMD/numerical/Security/Security.csv")
    values = pd.to_numeric(frame["OT"], errors="coerce").to_numpy(float)
    train_end, validation_end = int(len(values) * 0.7), int(len(values) * 0.8)
    z = (values - values[:train_end].mean()) / values[:train_end].std()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.plot(np.arange(len(z)), z, color=COLORS["navy"], lw=1.5)
    ax.axvspan(0, train_end, color=COLORS["green"], alpha=0.10, label="训练")
    ax.axvspan(train_end, validation_end, color=COLORS["orange"], alpha=0.14, label="验证")
    ax.axvspan(validation_end, len(z), color=COLORS["red"], alpha=0.12, label="测试")
    ax.axhline(0, color=COLORS["gray"], lw=0.8)
    ax.set(title="Security目标序列的末段分布漂移", xlabel="按月排序后的时间索引", ylabel="按训练段标准化的OT")
    ax.legend(ncol=3)
    ax.text(
        0.01, 0.03,
        "测试段大幅超出训练范围；回退只能避免额外文本损失，不能消除目标漂移",
        transform=ax.transAxes, color=COLORS["gray"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.86, "pad": 3},
    )
    save(fig, output, "fig08_security_distribution_shift")


def robustness(root: Path, output: Path) -> None:
    data = pd.read_csv(root / "outputs/tables/final/table_robustness.csv")
    data = data[(data["model_family"] == "PatchTST-M") & (data["corruption"] != "clean")].copy()
    labels = {
        "gaussian_0.10": "高斯噪声\nσ=0.10",
        "random_missing_0.10": "随机缺失\n10%",
        "tail_missing_12": "末段缺失\n12步",
        "spikes_0.02": "尖峰污染\n2%×3σ",
    }
    data["condition"] = data["corruption"].map(labels)
    data["task"] = data["dataset"] + "-H" + data["horizon"].astype(str)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    sns.barplot(
        data=data, x="task", y="degradation_pct_mean", hue="condition",
        palette=[COLORS["blue"], COLORS["cyan"], COLORS["orange"], COLORS["red"]], ax=ax,
    )
    ax.axhline(0, color="#222222", lw=0.8)
    ax.set(
        title="PatchTST数值骨干的输入扰动稳健性",
        xlabel="数据集—预测跨度",
        ylabel="相对干净输入的MSE退化（%）",
    )
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="扰动条件", ncol=4, loc="upper left")
    save(fig, output, "fig09_input_robustness")


def run(root: Path, output: Path) -> None:
    configure()
    output.mkdir(parents=True, exist_ok=True)
    pipeline(output)
    coverage(root, output)
    development_gain(root, output)
    alignment_test(root, output)
    confirmation_safety(root, output)
    ett_results(root, output)
    stability(root, output)
    security_drift(root, output)
    robustness(root, output)
    expected = {f"fig{index:02d}" for index in range(1, 10)}
    stems = {path.stem.split("_")[0] for path in output.glob("*.png")}
    if not expected.issubset(stems):
        raise AssertionError("Not all expected figures were generated")
    print(f"generated {len(list(output.glob('*.png')))} PNG and {len(list(output.glob('*.pdf')))} PDF figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/figures"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.root, args.output)
