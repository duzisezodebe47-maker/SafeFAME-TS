"""Reviewer-requested sensitivity analyses for SafeFAME-TS v2.

This script does not alter the frozen v2 routing decisions. It adds:
1. circular-shift text permutations that preserve within-split temporal order;
2. matched-capacity numeric residual controls for text-attribution diagnostics;
3. a descriptive block-based power audit for the pathway-decision segment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from data_utils import load_time_ordered_frame
from run_baselines import ALL_CONFIGS, make_windows, metrics
from run_famets_selective import frequency_statistics, semantic_rows
from run_safefame_v2 import (
    RIDGE_ALPHAS,
    VARIANTS,
    ResidualBuilder,
    fit_residual,
    moving_block_interval,
    refit_residual,
)


DEFAULT_SHIFTS = 999


class NumericControlBuilder:
    """Match the residual candidate's numerical capacity without text inputs."""

    def __init__(self, variant: str):
        self.variant = variant
        self.numeric_scaler = StandardScaler()
        self.frequency_scaler = StandardScaler()

    def fit(self, x: np.ndarray) -> "NumericControlBuilder":
        self.numeric_scaler.fit(x)
        if self.variant == "frequency_residual":
            self.frequency_scaler.fit(frequency_statistics(x))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        numeric = self.numeric_scaler.transform(x)
        if self.variant == "semantic_residual":
            return numeric
        return np.c_[numeric, self.frequency_scaler.transform(frequency_statistics(x))]


def fit_control(
    variant: str,
    train: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    calibration: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[NumericControlBuilder, Ridge, float, np.ndarray]:
    builder = NumericControlBuilder(variant).fit(train[0])
    residual = train[3] - train[0][:, -1, None]
    choices = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, solver="cholesky").fit(builder.transform(train[0]), residual)
        prediction = calibration[0][:, -1, None] + model.predict(builder.transform(calibration[0]))
        choices.append((metrics(calibration[3], prediction)["mse"], alpha, model, prediction))
    _, alpha, model, prediction = min(choices, key=lambda item: item[0])
    return builder, model, alpha, prediction


def refit_control(
    variant: str,
    alpha: float,
    fit_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    builder = NumericControlBuilder(variant).fit(fit_data[0])
    model = Ridge(alpha=alpha, solver="cholesky").fit(
        builder.transform(fit_data[0]), fit_data[3] - fit_data[0][:, -1, None]
    )
    return test[0][:, -1, None] + model.predict(builder.transform(test[0]))


def shifted_order(length: int, rng: np.random.Generator, guard: int) -> np.ndarray:
    if length < 2:
        return np.arange(length)
    lower = min(max(1, guard), length - 1)
    eligible = np.arange(lower, length - lower + 1)
    if len(eligible) == 0:
        eligible = np.arange(1, length)
    shift = int(rng.choice(eligible))
    return np.roll(np.arange(length), shift)


def circular_shift_scores(
    builder: ResidualBuilder,
    train: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    calibration: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    decision: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    count: int,
    guard: int,
    seed: int,
    alpha: float,
) -> list[float]:
    rng = np.random.default_rng(seed)
    residual = train[3] - train[0][:, -1, None]
    scores = []
    for _ in range(count):
        train_order = shifted_order(len(train[0]), rng, guard)
        decision_order = shifted_order(len(decision[0]), rng, guard)
        train_design = builder.transform(train[0], train[1][train_order], train[2][train_order])
        # Keep the alpha selected on aligned calibration data fixed. The
        # sensitivity check changes only temporal alignment, not model capacity.
        model = Ridge(alpha=alpha, solver="cholesky").fit(train_design, residual)
        decision_design = builder.transform(
            decision[0], decision[1][decision_order], decision[2][decision_order]
        )
        prediction = decision[0][:, -1, None] + model.predict(decision_design)
        scores.append(metrics(decision[3], prediction)["mse"])
    return scores


def block_power_audit(
    actual: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    block_length: int,
) -> tuple[int, float, float]:
    difference = np.mean((control - actual) ** 2 - (candidate - actual) ** 2, axis=1)
    block_count = len(difference) // block_length
    if block_count < 3:
        return block_count, float("nan"), float("nan")
    trimmed = difference[: block_count * block_length]
    block_means = trimmed.reshape(block_count, block_length).mean(axis=1)
    standard_error = block_means.std(ddof=1) / np.sqrt(block_count)
    critical = NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(0.80)
    minimum_detectable = critical * standard_error
    control_mse = metrics(actual, control)["mse"]
    return block_count, float(minimum_detectable), float(100.0 * minimum_detectable / control_mse)


def make_figures(frame: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"semantic_residual": "#2878B5", "frequency_residual": "#D95F02"}

    domain_order = list(ALL_CONFIGS)
    task_order = [(domain, horizon) for domain in domain_order for horizon in ALL_CONFIGS[domain].horizons]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 7.2), sharey=True)
    titles = {"semantic_residual": "语义残差候选", "frequency_residual": "频率语义残差候选"}
    for ax, variant in zip(axes, VARIANTS, strict=True):
        subset = frame[frame.variant.eq(variant)].set_index(["domain", "horizon"]).loc[task_order].reset_index()
        positions = np.arange(len(subset))
        ax.errorbar(
            subset["test_gain_vs_matched_control_pct"], positions,
            xerr=np.vstack([
                subset["test_gain_vs_matched_control_pct"] - subset["test_gain_ci_low_pct"],
                subset["test_gain_ci_high_pct"] - subset["test_gain_vs_matched_control_pct"],
            ]),
            fmt="o", capsize=2.5, color=colors[variant],
        )
        ax.axvline(0, color="#555555", linewidth=1)
        ax.set_yticks(positions, [f"{d} H{h}" for d, h in task_order])
        ax.set_xlabel("相对匹配容量数值残差对照的MSE改善（%）")
        ax.set_title(titles[variant])
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle("封存测试期的文本增量归因检查", fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "fig_v3_matched_control_ci.png", dpi=220)
    fig.savefig(output / "fig_v3_matched_control_ci.pdf")
    plt.close(fig)

    pivot = frame.pivot(index=["domain", "horizon"], columns="variant", values="circular_shift_p_value").loc[task_order]
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    x = np.arange(len(pivot))
    width = 0.38
    for offset, variant in zip((-width / 2, width / 2), VARIANTS, strict=True):
        ax.bar(x + offset, pivot[variant], width, label=titles[variant], color=colors[variant])
    ax.axhline(0.025, color="#B5473C", linestyle="--", linewidth=1.2, label="门槛 0.025")
    ax.set_xticks(x, [f"{d}\nH{h}" for d, h in pivot.index], rotation=35, ha="right")
    ax.set_ylabel("循环移位置换检验p值")
    ax.set_title("保留时间结构的999次循环移位置换敏感性")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output / "fig_v3_circular_shift_gate.png", dpi=220)
    fig.savefig(output / "fig_v3_circular_shift_gate.pdf")
    plt.close(fig)


def run(root: Path, semantic_root: Path, output: Path, shifts: int) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for domain, config in ALL_CONFIGS.items():
        ordered, _ = load_time_ordered_frame(root / "numerical" / domain / f"{domain}.csv")
        raw = pd.to_numeric(ordered["OT"], errors="coerce").to_numpy(float)
        train_end, validation_end = int(len(raw) * 0.7), int(len(raw) * 0.8)
        target = (raw - raw[:train_end].mean()) / raw[:train_end].std(ddof=0)
        cache = np.load(semantic_root / f"{domain}.npz")
        for horizon in config.horizons:
            arrays = {}
            for split, origin_range in {
                "train": range(config.input_len, train_end - horizon + 1),
                "validation": range(train_end, validation_end - horizon + 1),
                "test": range(validation_end, len(raw) - horizon + 1),
            }.items():
                x, y, origins = make_windows(target, config.input_len, horizon, origin_range)
                embedding, quality = semantic_rows(cache, origins)
                arrays[split] = (x, embedding, quality, y)
            midpoint = len(arrays["validation"][0]) // 2
            calibration = tuple(value[:midpoint] for value in arrays["validation"])
            decision = tuple(value[midpoint:] for value in arrays["validation"])
            fit_data = tuple(np.concatenate([arrays["train"][i], arrays["validation"][i]]) for i in range(4))
            block_length = max(horizon, min(config.seasonal_period, 24))

            for variant_index, variant in enumerate(VARIANTS):
                builder, model, alpha, _ = fit_residual(variant, arrays["train"], calibration)
                decision_prediction = decision[0][:, -1, None] + model.predict(
                    builder.transform(decision[0], decision[1], decision[2])
                )
                control_builder, control_model, control_alpha, _ = fit_control(
                    variant, arrays["train"], calibration
                )
                decision_control = decision[0][:, -1, None] + control_model.predict(
                    control_builder.transform(decision[0])
                )
                shifted = circular_shift_scores(
                    builder, arrays["train"], calibration, decision, shifts, block_length,
                    seed=3100 + 100 * list(ALL_CONFIGS).index(domain) + 10 * horizon + variant_index,
                    alpha=alpha,
                )
                aligned_mse = metrics(decision[3], decision_prediction)["mse"]
                p_value = (1 + sum(value <= aligned_mse for value in shifted)) / (shifts + 1)

                test_candidate = refit_residual(variant, alpha, fit_data, arrays["test"])
                test_control = refit_control(variant, control_alpha, fit_data, arrays["test"])
                delta, low, high = moving_block_interval(
                    arrays["test"][3], test_control, test_candidate, block_length,
                    seed=4100 + 100 * list(ALL_CONFIGS).index(domain) + 10 * horizon + variant_index,
                )
                control_test_mse = metrics(arrays["test"][3], test_control)["mse"]
                block_count, mde, mde_pct = block_power_audit(
                    decision[3], decision_control, decision_prediction, block_length
                )
                rows.append({
                    "domain": domain,
                    "horizon": horizon,
                    "variant": variant,
                    "decision_windows": len(decision[0]),
                    "block_length": block_length,
                    "nonoverlapping_decision_blocks": block_count,
                    "aligned_decision_mse": aligned_mse,
                    "matched_control_decision_mse": metrics(decision[3], decision_control)["mse"],
                    "circular_shift_permutations": shifts,
                    "circular_shift_p_value": p_value,
                    "approx_mde_loss": mde,
                    "approx_mde_vs_control_pct": mde_pct,
                    "test_candidate_mse": metrics(arrays["test"][3], test_candidate)["mse"],
                    "test_matched_control_mse": control_test_mse,
                    "test_gain_vs_matched_control_pct": 100.0 * (1.0 - metrics(arrays["test"][3], test_candidate)["mse"] / control_test_mse),
                    "test_loss_difference_vs_control": delta,
                    "test_gain_ci_low_pct": 100.0 * low / control_test_mse,
                    "test_gain_ci_high_pct": 100.0 * high / control_test_mse,
                })
            print(f"{domain} H={horizon}: sensitivity complete", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "reviewer_sensitivity_results.csv", index=False)
    summary = {
        "tasks": int(frame[["domain", "horizon"]].drop_duplicates().shape[0]),
        "pathways": int(len(frame)),
        "circular_shift_permutations": shifts,
        "circular_shift_alpha_policy": "fixed at the value selected on aligned calibration data",
        "circular_shift_passes_at_0_025": int((frame.circular_shift_p_value <= 0.025).sum()),
        "matched_control_test_ci_positive": int((frame.test_gain_ci_low_pct > 0).sum()),
        "matched_control_test_ci_negative": int((frame.test_gain_ci_high_pct < 0).sum()),
        "power_audit_unavailable_lt3_blocks": int(frame.approx_mde_loss.isna().sum()),
        "note": "Sensitivity analyses are diagnostic and do not change the frozen v2 routing decisions.",
    }
    (output / "reviewer_sensitivity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figures(frame, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return frame


def self_check() -> None:
    rng = np.random.default_rng(9)
    x = rng.normal(size=(40, 12))
    y = np.repeat(x[:, -1:], 3, axis=1) + rng.normal(scale=0.2, size=(40, 3))
    data = (x, rng.normal(size=(40, 16)), rng.normal(size=(40, 10)), y)
    calibration = tuple(value[25:32] for value in data)
    train = tuple(value[:25] for value in data)
    builder, model, alpha, prediction = fit_control("frequency_residual", train, calibration)
    assert prediction.shape == (7, 3) and alpha > 0
    assert builder.transform(x[32:]).shape[0] == 8 and model.coef_.shape[0] == 3
    order = shifted_order(20, rng, 4)
    assert sorted(order.tolist()) == list(range(20)) and not np.array_equal(order, np.arange(20))
    print("self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("references/external/Time-MMD"))
    parser.add_argument("--semantic-root", type=Path, default=Path("data_processed/semantic_features"))
    parser.add_argument("--output", type=Path, default=Path("outputs/reviewer_sensitivity"))
    parser.add_argument("--shifts", type=int, default=DEFAULT_SHIFTS)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        self_check()
    else:
        run(args.root, args.semantic_root, args.output, args.shifts)
