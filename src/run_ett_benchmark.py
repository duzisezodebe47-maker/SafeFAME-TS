"""External ETTh1/ETTh2 benchmark for the numerical forecasting backbone."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_baselines import ALPHAS, fit_ridge, metrics, predict_ridge, seasonal_naive
from train_dlinear import DLinearTarget, SEEDS, TrainConfig, make_windows, predict, score, train_one
from train_patchtst import PatchTSTTarget


DATASETS = ("ETTh1", "ETTh2")
HORIZONS = (96, 336, 720)
INPUT_LEN = 96
TRAIN_END = 12 * 30 * 24
VALIDATION_END = TRAIN_END + 4 * 30 * 24
TOTAL_END = VALIDATION_END + 4 * 30 * 24


def load_ett(path: Path) -> tuple[np.ndarray, list[str], int, dict[str, object]]:
    frame = pd.read_csv(path)
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError(f"{path} must contain unique ascending timestamps")
    if len(frame) < TOTAL_END:
        raise ValueError(f"{path} has {len(frame)} rows; at least {TOTAL_END} required")
    numeric = frame.drop(columns=["date"]).apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or "OT" not in numeric:
        raise ValueError(f"{path} has missing numeric values or no OT")
    features = ["OT", *[column for column in numeric if column != "OT"]]
    numeric = numeric[features].iloc[:TOTAL_END]
    mean = numeric.iloc[:TRAIN_END].mean()
    std = numeric.iloc[:TRAIN_END].std(ddof=0).replace(0.0, 1.0)
    values = ((numeric - mean) / std).to_numpy(np.float32)
    audit = {
        "rows_used": len(values),
        "start": dates.iloc[0].isoformat(),
        "end": dates.iloc[TOTAL_END - 1].isoformat(),
        "features": features,
        "target": "OT",
        "normalization": "training-segment mean and population standard deviation",
        "train_target_mean": float(mean["OT"]),
        "train_target_std": float(std["OT"]),
    }
    return values, features, 0, audit


def append_origin_errors(
    rows: list[pd.DataFrame], dataset: str, horizon: int, model: str, seed: int | None,
    origins: np.ndarray, actual: np.ndarray, prediction: np.ndarray,
) -> None:
    rows.append(
        pd.DataFrame(
            {
                "dataset": dataset,
                "horizon": horizon,
                "model": model,
                "seed": seed,
                "origin_index": origins,
                "origin_mse": np.mean((prediction - actual) ** 2, axis=1),
                "origin_mae": np.mean(np.abs(prediction - actual), axis=1),
            }
        )
    )


def run(root: Path, output: Path, device: torch.device) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    origin_error_rows: list[pd.DataFrame] = []
    sample_rows: list[pd.DataFrame] = []
    audits: dict[str, object] = {}
    training = TrainConfig(
        batch_size=128,
        learning_rate=3e-4,
        weight_decay=1e-4,
        max_epochs=60,
        patience=8,
        min_delta=1e-6,
    )

    for dataset in DATASETS:
        path = root / f"{dataset}.csv"
        values, features, target_index, audit = load_ett(path)
        audits[dataset] = audit
        for horizon in HORIZONS:
            x_train, y_train, _ = make_windows(
                values, target_index, INPUT_LEN, horizon,
                range(INPUT_LEN, TRAIN_END - horizon + 1),
            )
            x_val, y_val, _ = make_windows(
                values, target_index, INPUT_LEN, horizon,
                range(TRAIN_END, VALIDATION_END - horizon + 1),
            )
            x_test, y_test, origins = make_windows(
                values, target_index, INPUT_LEN, horizon,
                range(VALIDATION_END, TOTAL_END - horizon + 1),
            )
            arrays = (x_train, y_train, x_val, y_val, x_test, y_test)
            univariate_train = x_train[:, :, target_index].numpy()
            univariate_val = x_val[:, :, target_index].numpy()
            univariate_test = x_test[:, :, target_index].numpy()
            y_train_np, y_val_np, y_test_np = y_train.numpy(), y_val.numpy(), y_test.numpy()
            validation_scores = {}
            for alpha in ALPHAS:
                forecast = predict_ridge(univariate_val, fit_ridge(univariate_train, y_train_np, alpha))
                validation_scores[alpha] = metrics(y_val_np, forecast)["mse"]
            best_alpha = min(validation_scores, key=validation_scores.get)
            ridge = fit_ridge(
                np.r_[univariate_train, univariate_val], np.r_[y_train_np, y_val_np], best_alpha
            )
            simple_predictions = {
                "Last": np.repeat(univariate_test[:, -1:], horizon, axis=1),
                "SeasonalNaive24": seasonal_naive(univariate_test, horizon, 24),
                "ARRidge": predict_ridge(univariate_test, ridge),
            }
            for model_name, forecast in simple_predictions.items():
                result = metrics(y_test_np, forecast)
                metric_rows.append(
                    {
                        "dataset": dataset,
                        "horizon": horizon,
                        "model": model_name,
                        "seed": np.nan,
                        **result,
                        "best_epoch": np.nan,
                        "training_seconds": 0.0,
                        "parameters": 0,
                        "selected_alpha": best_alpha if model_name == "ARRidge" else np.nan,
                        "train_windows": len(x_train),
                        "validation_windows": len(x_val),
                        "test_windows": len(x_test),
                    }
                )
                append_origin_errors(
                    origin_error_rows, dataset, horizon, model_name, None, origins, y_test_np, forecast
                )

            model_specs = {
                "DLinear-M": (
                    DLinearTarget,
                    {"input_len": INPUT_LEN, "channels": len(features), "horizon": horizon, "kernel_size": 25},
                ),
                "PatchTST-M": (
                    PatchTSTTarget,
                    {
                        "input_len": INPUT_LEN, "channels": len(features), "horizon": horizon,
                        "patch_len": 16, "stride": 8, "d_model": 32, "n_heads": 4,
                        "e_layers": 2, "d_ff": 64, "dropout": 0.1,
                    },
                ),
            }
            for model_name, (model_class, model_args) in model_specs.items():
                for seed in SEEDS:
                    model, history, best_epoch, seconds = train_one(
                        arrays, model_args, training, seed, device, model_class=model_class
                    )
                    forecast = predict(model, x_test, device, training.batch_size)
                    result = score(y_test_np, forecast)
                    metric_rows.append(
                        {
                            "dataset": dataset,
                            "horizon": horizon,
                            "model": model_name,
                            "seed": seed,
                            **result,
                            "best_epoch": best_epoch,
                            "training_seconds": seconds,
                            "parameters": sum(parameter.numel() for parameter in model.parameters()),
                            "selected_alpha": np.nan,
                            "train_windows": len(x_train),
                            "validation_windows": len(x_val),
                            "test_windows": len(x_test),
                        }
                    )
                    append_origin_errors(
                        origin_error_rows, dataset, horizon, model_name, seed, origins, y_test_np, forecast
                    )
                    tail = min(32, len(origins))
                    sample_rows.append(
                        pd.DataFrame(
                            {
                                "dataset": dataset,
                                "horizon": horizon,
                                "model": model_name,
                                "seed": seed,
                                "origin_index": np.repeat(origins[-tail:], horizon),
                                "step": np.tile(np.arange(1, horizon + 1), tail),
                                "actual_z": y_test_np[-tail:].ravel(),
                                "prediction_z": forecast[-tail:].ravel(),
                            }
                        )
                    )
                    torch.save(
                        {"state_dict": model.state_dict(), "model_args": model_args, "features": features},
                        output / "models" / f"{dataset}_h{horizon}_{model_name}_s{seed}.pt",
                    )
                    pd.DataFrame(history).to_csv(
                        output / "logs" / f"{dataset}_h{horizon}_{model_name}_s{seed}.csv", index=False
                    )
                    print(
                        f"{dataset} H={horizon} {model_name} seed={seed} "
                        f"MSE={result['mse']:.6f} epoch={best_epoch} time={seconds:.1f}s",
                        flush=True,
                    )

    metrics_frame = pd.DataFrame(metric_rows)
    summary = (
        metrics_frame.groupby(["dataset", "horizon", "model"], as_index=False, dropna=False)
        .agg(
            mse_mean=("mse", "mean"), mse_std=("mse", "std"),
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            training_seconds_mean=("training_seconds", "mean"),
            parameters=("parameters", "first"), test_windows=("test_windows", "first"),
        )
    )
    metrics_frame.to_csv(output / "ett_metrics_by_seed.csv", index=False)
    summary.to_csv(output / "ett_metrics_summary.csv", index=False)
    pd.concat(origin_error_rows, ignore_index=True).to_csv(output / "ett_origin_errors.csv", index=False)
    pd.concat(sample_rows, ignore_index=True).to_csv(output / "ett_prediction_samples.csv", index=False)
    (output / "ett_run.json").write_text(
        json.dumps(
            {
                "datasets": DATASETS,
                "horizons": HORIZONS,
                "input_len": INPUT_LEN,
                "split_boundaries": {"train_end": TRAIN_END, "validation_end": VALIDATION_END, "total_end": TOTAL_END},
                "seeds": SEEDS,
                "train_config": asdict(training),
                "device": str(device),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "audits": audits,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def self_check() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(8, INPUT_LEN))
    stats = frequency = np.abs(np.fft.rfft(x, axis=1))
    assert stats.shape == frequency.shape and stats.shape[1] == INPUT_LEN // 2 + 1
    model = PatchTSTTarget(INPUT_LEN, 7, 96, 16, 8)
    assert model(torch.randn(2, INPUT_LEN, 7)).shape == (2, 96)
    print("self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("references/external/ETDataset/ETT-small")
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/ett_benchmark"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        self_check()
    else:
        print(run(args.root, args.output, torch.device(args.device)).to_string(index=False))
