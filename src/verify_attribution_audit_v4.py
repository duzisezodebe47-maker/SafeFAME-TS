"""Recompute the saved v4 attribution-audit claims without model training."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from run_safefame_v4 import interval, metrics


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "attribution_audit_v4"


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-10):
        raise AssertionError(f"{label}: {actual} != {expected}")


def main() -> None:
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    results = pd.read_csv(OUTPUT / "selected_route_attribution.csv")
    comparisons = pd.read_csv(OUTPUT / "selected_route_comparisons.csv")
    blocks = pd.read_csv(OUTPUT / "block_length_sensitivity.csv")
    corrected = pd.read_csv(OUTPUT / "corrected_frequency_permutation.csv")
    families = pd.read_csv(OUTPUT / "cross_fold_replication.csv")
    assert len(results) == summary["selected_routes"] == 10
    assert len(comparisons) == 50 and len(corrected) == 1
    assert (results.full_dimension == results.dimension_matched_numeric_dimension).all()
    assert summary["original_selected_ci_positive"] == int((results.frozen_selected_ci_low_pct > 0).sum())
    assert summary["joint_posthoc_supported_routes"] == int(results.joint_posthoc_support.sum()) == 1
    assert summary["exploratory_cross_fold_replicated_families"] == int(families.exploratory_replicated.sum())

    for result in results.itertuples(index=False):
        stem = f"{result.domain}_h{result.horizon}_f{result.fold}_{result.variant}.npz"
        evidence = np.load(OUTPUT / "evidence" / stem)
        actual, fallback = evidence["actual"], evidence["fallback"]
        original, full = evidence["original_candidate"], evidence["full"]
        dimension = evidence["dimension_matched_numeric"]
        fallback_mse = metrics(actual, fallback)["mse"]
        full_mse = metrics(actual, full)["mse"]
        dimension_mse = metrics(actual, dimension)["mse"]
        close(result.original_gain_vs_fallback_pct,
              100 * (1 - metrics(actual, original)["mse"] / fallback_mse), "original gain")
        close(result.fallback_head_gain_vs_fallback_pct, 100 * (1 - full_mse / fallback_mse), "head gain")
        close(result.full_gain_vs_dimension_matched_numeric_pct,
              100 * (1 - full_mse / dimension_mse), "dimension gain")
        _, low, high = interval(actual, fallback, full, int(result.block_length),
                                int(result.fallback_bootstrap_seed), summary["bootstrap_repeats"])
        close(result.fallback_head_ci_low_pct, 100 * low / fallback_mse, "head CI low")
        close(result.fallback_head_ci_high_pct, 100 * high / fallback_mse, "head CI high")
        _, low, high = interval(actual, dimension, full, int(result.block_length),
                                int(result.dimension_bootstrap_seed), summary["bootstrap_repeats"])
        close(result.full_vs_dimension_ci_low_pct, 100 * low / dimension_mse, "dimension CI low")
        close(result.full_vs_dimension_ci_high_pct, 100 * high / dimension_mse, "dimension CI high")

        local = comparisons[(comparisons.domain == result.domain) &
                            (comparisons.horizon == result.horizon) &
                            (comparisons.fold == result.fold)]
        assert len(local) == 5
        for comparison in local.itertuples(index=False):
            control, candidate = evidence[comparison.control], evidence[comparison.candidate]
            control_mse = metrics(actual, control)["mse"]
            candidate_mse = metrics(actual, candidate)["mse"]
            close(comparison.control_mse, control_mse, "control MSE")
            close(comparison.candidate_mse, candidate_mse, "candidate MSE")
            delta, low, high = interval(actual, control, candidate, int(result.block_length),
                                        int(comparison.bootstrap_seed), summary["bootstrap_repeats"])
            close(comparison.loss_delta, delta, "comparison delta")
            close(comparison.ci_low, low, "comparison CI low")
            close(comparison.ci_high, high, "comparison CI high")

        local_blocks = blocks[(blocks.domain == result.domain) &
                              (blocks.horizon == result.horizon) &
                              (blocks.fold == result.fold)]
        for block in local_blocks.itertuples(index=False):
            delta, low, high = interval(actual, fallback, original, int(block.block_length),
                                        int(block.bootstrap_seed), summary["bootstrap_repeats"])
            close(block.loss_delta, delta, "block delta")
            close(block.ci_low_pct, 100 * low / fallback_mse, "block CI low")
            close(block.ci_high_pct, 100 * high / fallback_mse, "block CI high")

    corrected_row = corrected.iloc[0]
    nulls = np.load(OUTPUT / "evidence" /
                    f"{corrected_row.domain}_h{corrected_row.horizon}_f{corrected_row.fold}_corrected_frequency_nulls.npz")
    count = int(corrected_row.permutations)
    aligned = float(nulls["aligned_decision_mse"])
    close(corrected_row.corrected_row_p,
          (1 + int((nulls["row_null"] <= aligned).sum())) / (count + 1), "corrected row p")
    close(corrected_row.corrected_circular_p,
          (1 + int((nulls["circular_null"] <= aligned).sum())) / (count + 1), "corrected circular p")
    assert corrected_row.corrected_row_p > 0.025
    print("PASS: 10 selected routes, 50 factorial comparisons, block sensitivity and corrected frequency nulls recomputed")


if __name__ == "__main__":
    main()
