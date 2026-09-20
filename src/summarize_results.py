"""Create publication-ready result tables from verified experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def development_table(root: Path) -> pd.DataFrame:
    numeric = pd.read_csv(root / "outputs/tables/baseline_comparison.csv")
    numeric = numeric.loc[numeric.groupby(["domain", "horizon"])["mse_mean"].idxmin(), [
        "domain", "horizon", "model", "mse_mean"
    ]].rename(columns={"model": "best_numeric_model", "mse_mean": "best_numeric_mse"})
    semantic = pd.read_csv(root / "outputs/semantic_probe/semantic_probe_metrics.csv")
    semantic = semantic[semantic["model"] == "MiniLM-EarlyFusion"].pivot(
        index=["domain", "horizon"], columns="alignment", values="mse"
    ).reset_index().rename(columns={"aligned": "aligned_semantic_mse", "shuffled": "shuffled_semantic_mse"})
    selective = pd.read_csv(root / "outputs/famets_selective/famets_selective_metrics.csv")
    selective = selective[selective["model"] == "FAME-TS-Selective"][[
        "domain", "horizon", "mse", "mse_ci_low", "mse_ci_high"
    ]].rename(columns={"mse": "selective_mse"})
    audit = pd.read_csv(root / "outputs/famets_selective/famets_selection_audit.csv")[[
        "domain", "horizon", "selected_variant", "selected_gate_fraction"
    ]]
    result = numeric.merge(semantic, on=["domain", "horizon"]).merge(
        selective, on=["domain", "horizon"]
    ).merge(audit, on=["domain", "horizon"])
    result["semantic_improvement_vs_best_numeric_pct"] = 100 * (
        1 - result["aligned_semantic_mse"] / result["best_numeric_mse"]
    )
    result["alignment_gain_vs_shuffle_pct"] = 100 * (
        1 - result["aligned_semantic_mse"] / result["shuffled_semantic_mse"]
    )
    result["aligned_beats_shuffled"] = result["aligned_semantic_mse"] < result["shuffled_semantic_mse"]
    return result.sort_values(["domain", "horizon"])


def confirmation_table(root: Path) -> pd.DataFrame:
    baseline = pd.read_csv(root / "outputs/confirmation_baselines/baseline_metrics.csv")
    baseline = baseline.loc[baseline.groupby(["domain", "horizon"])["mse"].idxmin(), [
        "domain", "horizon", "model", "mse"
    ]].rename(columns={"model": "best_traditional_model", "mse": "best_traditional_mse"})
    fame = pd.read_csv(root / "outputs/famets_confirmation/famets_selective_metrics.csv").pivot(
        index=["domain", "horizon"], columns="model", values="mse"
    ).reset_index().rename(
        columns={
            "FAME-TS-numeric": "internal_numeric_mse",
            "FAME-TS-no_frequency": "semantic_mse",
            "FAME-TS-full": "frequency_semantic_mse",
            "FAME-TS-Selective": "selective_mse",
        }
    )
    audit = pd.read_csv(root / "outputs/famets_confirmation/famets_selection_audit.csv")[[
        "domain", "horizon", "selected_variant", "selected_gate_fraction"
    ]]
    result = baseline.merge(fame, on=["domain", "horizon"]).merge(audit, on=["domain", "horizon"])
    result["least_harmful_semantic_mse"] = result[["semantic_mse", "frequency_semantic_mse"]].min(axis=1)
    result["avoided_semantic_loss_pct"] = 100 * (
        result["least_harmful_semantic_mse"] - result["selective_mse"]
    ) / result["least_harmful_semantic_mse"]
    result["selective_vs_internal_numeric_pct"] = 100 * (
        1 - result["selective_mse"] / result["internal_numeric_mse"]
    )
    return result.sort_values(["domain", "horizon"])


def ett_table(root: Path) -> pd.DataFrame:
    metrics = pd.read_csv(root / "outputs/ett_benchmark/ett_metrics_summary.csv")
    best = metrics.loc[metrics.groupby(["dataset", "horizon"])["mse_mean"].idxmin(), [
        "dataset", "horizon", "model", "mse_mean", "mse_std"
    ]].rename(columns={"model": "best_model", "mse_mean": "best_mse", "mse_std": "best_mse_std"})
    non_patch = metrics[metrics["model"] != "PatchTST-M"]
    non_patch = non_patch.loc[non_patch.groupby(["dataset", "horizon"])["mse_mean"].idxmin(), [
        "dataset", "horizon", "model", "mse_mean"
    ]].rename(columns={"model": "best_non_patch_model", "mse_mean": "best_non_patch_mse"})
    result = best.merge(non_patch, on=["dataset", "horizon"])
    result["patchtst_gain_vs_best_non_patch_pct"] = 100 * (
        1 - result["best_mse"] / result["best_non_patch_mse"]
    )
    return result.sort_values(["dataset", "horizon"])


def robustness_table(root: Path) -> pd.DataFrame:
    data = pd.read_csv(root / "outputs/robustness/robustness_summary.csv")
    return data.sort_values(["dataset", "horizon", "model_family", "corruption"])


def run(root: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    development = development_table(root)
    confirmation = confirmation_table(root)
    ett = ett_table(root)
    robustness = robustness_table(root)
    if len(development) != 12 or len(confirmation) != 6 or len(ett) != 6 or len(robustness) != 90:
        raise AssertionError("Unexpected task count in result tables")
    if not (confirmation["selected_variant"] == "numeric").all():
        raise AssertionError("Confirmation audit no longer matches frozen result")
    if not np.allclose(confirmation["selective_mse"], confirmation["internal_numeric_mse"]):
        raise AssertionError("Numeric fallback predictions must equal internal numeric predictions")
    if not (ett["best_model"] == "PatchTST-M").all():
        raise AssertionError("ETT winner statement does not match metrics")
    development.to_csv(output / "table_development_multimodal.csv", index=False)
    confirmation.to_csv(output / "table_confirmation_safety.csv", index=False)
    ett.to_csv(output / "table_ett_external.csv", index=False)
    robustness.to_csv(output / "table_robustness.csv", index=False)
    patch_robustness = robustness[robustness["model_family"] == "PatchTST-M"]
    mean_degradation = patch_robustness.groupby("corruption")["degradation_pct_mean"].mean()
    claims = {
        "development_tasks": len(development),
        "development_aligned_beats_shuffled": int(development["aligned_beats_shuffled"].sum()),
        "development_semantic_beats_best_numeric": int(
            (development["semantic_improvement_vs_best_numeric_pct"] > 0).sum()
        ),
        "confirmation_tasks": len(confirmation),
        "confirmation_semantic_routes_selected": int((confirmation["selected_variant"] != "numeric").sum()),
        "confirmation_harmful_routes_rejected": int(
            (confirmation["least_harmful_semantic_mse"] > confirmation["internal_numeric_mse"]).sum()
        ),
        "confirmation_mean_avoided_semantic_loss_pct": float(
            confirmation["avoided_semantic_loss_pct"].mean()
        ),
        "ett_tasks": len(ett),
        "ett_patchtst_wins": int((ett["best_model"] == "PatchTST-M").sum()),
        "ett_mean_patchtst_gain_pct": float(ett["patchtst_gain_vs_best_non_patch_pct"].mean()),
        "robustness_tasks": 6,
        "robustness_patchtst_gaussian_mean_degradation_pct": float(mean_degradation["gaussian_0.10"]),
        "robustness_patchtst_random_missing_mean_degradation_pct": float(mean_degradation["random_missing_0.10"]),
        "robustness_patchtst_tail_missing_mean_degradation_pct": float(mean_degradation["tail_missing_12"]),
        "robustness_patchtst_spikes_mean_degradation_pct": float(mean_degradation["spikes_0.02"]),
    }
    (output / "verified_claims.json").write_text(
        json.dumps(claims, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return claims


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("outputs/tables/final"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(run(args.root, args.output), ensure_ascii=False, indent=2))
