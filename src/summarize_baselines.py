"""Combine deterministic and neural baseline outputs into paper-ready tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_neural(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)[
        ["domain", "horizon", "model", "mse_mean", "mse_std", "mae_mean", "mae_std", "rmse_mean", "rmse_std"]
    ]


def run(root: Path, output: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    deterministic = pd.read_csv(root / "baselines" / "baseline_metrics.csv").rename(
        columns={"mse": "mse_mean", "mae": "mae_mean", "rmse": "rmse_mean"}
    )
    deterministic = deterministic[["domain", "horizon", "model", "mse_mean", "mae_mean", "rmse_mean"]]
    for column in ("mse_std", "mae_std", "rmse_std"):
        deterministic[column] = 0.0
    neural = [
        load_neural(root / "dlinear" / "dlinear_metrics_summary.csv"),
        load_neural(root / "dlinear_u" / "dlinear_metrics_summary.csv"),
        load_neural(root / "patchtst" / "patchtst_metrics_summary.csv"),
    ]
    comparison = pd.concat([deterministic, *neural], ignore_index=True)
    comparison["mse_rank"] = comparison.groupby(["domain", "horizon"])["mse_mean"].rank(method="min")
    best_simple = (
        comparison[comparison["model"].isin(["Last", "SeasonalNaive"])]
        .groupby(["domain", "horizon"])["mse_mean"]
        .min()
        .rename("best_simple_mse")
    )
    comparison = comparison.join(best_simple, on=["domain", "horizon"])
    comparison["improvement_vs_best_simple_pct"] = (
        100.0 * (comparison["best_simple_mse"] - comparison["mse_mean"]) / comparison["best_simple_mse"]
    )
    comparison = comparison.sort_values(["domain", "horizon", "mse_rank", "model"]).reset_index(drop=True)
    winners = comparison.loc[
        comparison.groupby(["domain", "horizon"])["mse_mean"].idxmin(),
        ["domain", "horizon", "model", "mse_mean", "mse_std", "mae_mean", "improvement_vs_best_simple_pct"],
    ].sort_values(["domain", "horizon"])
    assert len(comparison) == 72 and len(winners) == 12
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "baseline_comparison.csv", index=False)
    winners.to_csv(output / "baseline_winners.csv", index=False)
    return comparison, winners


if __name__ == "__main__":
    full, best = run(Path("outputs"), Path("outputs/tables"))
    print(best.to_string(index=False))

