"""Run leakage-safe univariate forecasting baselines on selected Time-MMD domains."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import load_time_ordered_frame


@dataclass(frozen=True)
class DomainConfig:
    input_len: int
    horizons: tuple[int, ...]
    seasonal_period: int


CONFIGS = {
    "Climate": DomainConfig(52, (4, 12, 24), 52),
    "Energy": DomainConfig(52, (4, 12, 24), 52),
    "Economy": DomainConfig(24, (3, 6, 12), 12),
    "Traffic": DomainConfig(24, (3, 6, 12), 12),
}
CONFIRMATION_CONFIGS = {
    "Agriculture": DomainConfig(24, (3, 6, 12), 12),
    "Security": DomainConfig(24, (3, 6, 12), 12),
}
ALL_CONFIGS = {**CONFIGS, **CONFIRMATION_CONFIGS}
ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def make_windows(values: np.ndarray, input_len: int, horizon: int, origins: range) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = [i for i in origins if i >= input_len and i + horizon <= len(values)]
    x = np.stack([values[i - input_len : i] for i in valid])
    y = np.stack([values[i : i + horizon] for i in valid])
    return x, y, np.asarray(valid)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    design = np.c_[np.ones(len(x)), (x - mean) / scale]
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + alpha * penalty, design.T @ y)
    return weights, np.stack([mean, scale])


def predict_ridge(x: np.ndarray, model: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    weights, stats = model
    design = np.c_[np.ones(len(x)), (x - stats[0]) / stats[1]]
    return design @ weights


def seasonal_naive(x: np.ndarray, horizon: int, period: int) -> np.ndarray:
    offsets = np.arange(horizon) % period
    return x[:, -period:][:, offsets]


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def evaluate_series(values: np.ndarray, config: DomainConfig) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    if np.isnan(values).any():
        raise ValueError("OT contains missing values; define and record an imputation rule first")
    n = len(values)
    train_end, val_end = int(n * 0.7), int(n * 0.8)
    train_mean, train_std = values[:train_end].mean(), values[:train_end].std()
    if train_std < 1e-8:
        raise ValueError("Training target has near-zero variance")
    scaled = (values - train_mean) / train_std
    result_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []

    for horizon in config.horizons:
        x_train, y_train, _ = make_windows(scaled, config.input_len, horizon, range(config.input_len, train_end - horizon + 1))
        x_val, y_val, _ = make_windows(scaled, config.input_len, horizon, range(train_end, val_end - horizon + 1))
        x_test, y_test, origins = make_windows(scaled, config.input_len, horizon, range(val_end, n - horizon + 1))
        if min(len(x_train), len(x_val), len(x_test)) == 0:
            raise ValueError(f"Not enough samples for horizon={horizon}")

        val_scores = {}
        for alpha in ALPHAS:
            val_pred = predict_ridge(x_val, fit_ridge(x_train, y_train, alpha))
            val_scores[alpha] = metrics(y_val, val_pred)["mse"]
        best_alpha = min(val_scores, key=val_scores.get)
        ridge_model = fit_ridge(np.r_[x_train, x_val], np.r_[y_train, y_val], best_alpha)

        forecasts = {
            "Last": np.repeat(x_test[:, -1:], horizon, axis=1),
            "SeasonalNaive": seasonal_naive(x_test, horizon, config.seasonal_period),
            "ARRidge": predict_ridge(x_test, ridge_model),
        }
        seasonal_mse = metrics(y_test, forecasts["SeasonalNaive"])["mse"]
        for model_name, forecast in forecasts.items():
            score = metrics(y_test, forecast)
            score.update(
                {
                    "horizon": horizon,
                    "model": model_name,
                    "input_len": config.input_len,
                    "train_windows": len(x_train),
                    "validation_windows": len(x_val),
                    "test_windows": len(x_test),
                    "selected_alpha": best_alpha if model_name == "ARRidge" else np.nan,
                    "skill_vs_seasonal": 1.0 - score["mse"] / seasonal_mse if seasonal_mse else np.nan,
                }
            )
            result_rows.append(score)
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "horizon": horizon,
                        "model": model_name,
                        "origin_index": np.repeat(origins, horizon),
                        "step": np.tile(np.arange(1, horizon + 1), len(origins)),
                        "actual_z": y_test.ravel(),
                        "prediction_z": forecast.ravel(),
                    }
                )
            )
    return result_rows, prediction_frames


def run(root: Path, output: Path, configs: dict[str, DomainConfig] = CONFIGS) -> pd.DataFrame:
    all_results: list[dict[str, object]] = []
    all_predictions: list[pd.DataFrame] = []
    for domain, config in configs.items():
        path = root / "numerical" / domain / f"{domain}.csv"
        ordered, _ = load_time_ordered_frame(path)
        values = pd.to_numeric(ordered["OT"], errors="coerce").to_numpy(float)
        results, predictions = evaluate_series(values, config)
        for row in results:
            row["domain"] = domain
        for frame in predictions:
            frame.insert(0, "domain", domain)
        all_results.extend(results)
        all_predictions.extend(predictions)

    output.mkdir(parents=True, exist_ok=True)
    result_frame = pd.DataFrame(all_results)
    result_frame.to_csv(output / "baseline_metrics.csv", index=False)
    pd.concat(all_predictions, ignore_index=True).to_csv(output / "baseline_predictions.csv", index=False)
    (output / "baseline_config.json").write_text(
        json.dumps(
            {
                "split": {"train": 0.7, "validation": 0.1, "test": 0.2},
                "metric_scale": "z-score using training-segment mean and standard deviation",
                "alpha_candidates": ALPHAS,
                "domains": {name: asdict(config) for name, config in configs.items()},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result_frame


def self_check() -> None:
    values = np.arange(200, dtype=float)
    config = DomainConfig(input_len=12, horizons=(3,), seasonal_period=12)
    rows, predictions = evaluate_series(values, config)
    assert len(rows) == 3 and len(predictions) == 3
    assert all(row["test_windows"] > 0 for row in rows)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "numerical" / "Demo").mkdir(parents=True)
        demo = pd.DataFrame(
            {"date": pd.date_range("2020-01-01", periods=len(values), freq="D"), "OT": values}
        )
        demo.to_csv(root / "numerical" / "Demo" / "Demo.csv", index=False)
        result = run(root, root / "out", {"Demo": config})
        assert len(result) == 3
        assert (root / "out" / "baseline_predictions.csv").exists()
        demo.sample(frac=1.0, random_state=7).to_csv(
            root / "numerical" / "Demo" / "Demo.csv", index=False
        )
        shuffled_result = run(root, root / "out_shuffled", {"Demo": config})
        pd.testing.assert_frame_equal(result, shuffled_result)
    print("self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("references/external/Time-MMD"))
    parser.add_argument("--output", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--domains", nargs="+", choices=tuple(ALL_CONFIGS), default=list(CONFIGS))
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        self_check()
    else:
        selected = {name: ALL_CONFIGS[name] for name in args.domains}
        print(run(args.root, args.output, selected).to_string(index=False))
