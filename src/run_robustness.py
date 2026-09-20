"""Input-corruption robustness evaluation for the frozen ETT backbone models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_baselines import ALPHAS, fit_ridge, metrics, predict_ridge
from run_ett_benchmark import (
    DATASETS,
    HORIZONS,
    INPUT_LEN,
    TOTAL_END,
    TRAIN_END,
    VALIDATION_END,
    load_ett,
)
from train_dlinear import make_windows, predict
from train_patchtst import PatchTSTTarget


CORRUPTIONS = ("clean", "gaussian_0.10", "random_missing_0.10", "tail_missing_12", "spikes_0.02")
CORRUPTION_SEEDS = (31, 37, 43)
MODEL_SEEDS = (2026, 2027, 2028)


def forward_fill_windows(x: np.ndarray) -> np.ndarray:
    result = x.copy()
    for step in range(result.shape[1]):
        missing = ~np.isfinite(result[:, step, :])
        replacement = np.zeros_like(result[:, step, :]) if step == 0 else result[:, step - 1, :]
        result[:, step, :] = np.where(missing, replacement, result[:, step, :])
    return result


def corrupt(x: np.ndarray, kind: str, seed: int) -> np.ndarray:
    if kind == "clean":
        return x.copy()
    rng = np.random.default_rng(seed)
    result = x.copy()
    if kind == "gaussian_0.10":
        return result + rng.normal(0.0, 0.10, size=result.shape).astype(np.float32)
    if kind == "random_missing_0.10":
        result[rng.random(result.shape) < 0.10] = np.nan
        return forward_fill_windows(result)
    if kind == "tail_missing_12":
        result[:, -12:, :] = np.nan
        return forward_fill_windows(result)
    if kind == "spikes_0.02":
        mask = rng.random(result.shape) < 0.02
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=result.shape)
        result[mask] += (3.0 * signs)[mask]
        return result
    raise ValueError(f"unknown corruption: {kind}")


def load_patch_model(path: Path, device: torch.device) -> PatchTSTTarget:
    payload = torch.load(path, map_location=device, weights_only=True)
    model = PatchTSTTarget(**payload["model_args"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model


def run(data_root: Path, model_root: Path, output: Path, device: torch.device) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        values, _, target_index, _ = load_ett(data_root / f"{dataset}.csv")
        for horizon in HORIZONS:
            x_train, y_train, _ = make_windows(
                values, target_index, INPUT_LEN, horizon,
                range(INPUT_LEN, TRAIN_END - horizon + 1),
            )
            x_val, y_val, _ = make_windows(
                values, target_index, INPUT_LEN, horizon,
                range(TRAIN_END, VALIDATION_END - horizon + 1),
            )
            x_test, y_test, _ = make_windows(
                values, target_index, INPUT_LEN, horizon,
                range(VALIDATION_END, TOTAL_END - horizon + 1),
            )
            x_train_u = x_train[:, :, target_index].numpy()
            x_val_u = x_val[:, :, target_index].numpy()
            y_train_np, y_val_np, y_test_np = y_train.numpy(), y_val.numpy(), y_test.numpy()
            alpha_scores = {
                alpha: metrics(y_val_np, predict_ridge(x_val_u, fit_ridge(x_train_u, y_train_np, alpha)))["mse"]
                for alpha in ALPHAS
            }
            best_alpha = min(alpha_scores, key=alpha_scores.get)
            ridge = fit_ridge(np.r_[x_train_u, x_val_u], np.r_[y_train_np, y_val_np], best_alpha)
            patch_models = {
                seed: load_patch_model(
                    model_root / "models" / f"{dataset}_h{horizon}_PatchTST-M_s{seed}.pt", device
                )
                for seed in MODEL_SEEDS
            }
            clean_cache: dict[tuple[str, int], float] = {}
            for corruption in CORRUPTIONS:
                seeds = (0,) if corruption == "clean" else CORRUPTION_SEEDS
                for corruption_seed in seeds:
                    x_corrupt = corrupt(x_test.numpy(), corruption, corruption_seed)
                    x_target = x_corrupt[:, :, target_index]
                    forecasts = {
                        "Last": np.repeat(x_target[:, -1:], horizon, axis=1),
                        "ARRidge": predict_ridge(x_target, ridge),
                    }
                    for model_seed, model in patch_models.items():
                        forecasts[f"PatchTST-M_s{model_seed}"] = predict(
                            model, torch.from_numpy(x_corrupt), device, batch_size=128
                        )
                    for model_name, forecast in forecasts.items():
                        result = metrics(y_test_np, forecast)
                        key = (model_name, corruption_seed)
                        if corruption == "clean":
                            clean_cache[(model_name, 0)] = result["mse"]
                        clean_mse = clean_cache[(model_name, 0)]
                        rows.append(
                            {
                                "dataset": dataset,
                                "horizon": horizon,
                                "model": model_name,
                                "corruption": corruption,
                                "corruption_seed": corruption_seed,
                                **result,
                                "clean_mse": clean_mse,
                                "relative_degradation_pct": 100.0 * (result["mse"] / clean_mse - 1.0),
                            }
                        )
            print(f"{dataset} H={horizon}: robustness complete", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "robustness_metrics.csv", index=False)
    summary = (
        frame.assign(
            model_family=frame["model"].str.replace(r"_s\d+$", "", regex=True)
        )
        .groupby(["dataset", "horizon", "model_family", "corruption"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            degradation_pct_mean=("relative_degradation_pct", "mean"),
            degradation_pct_std=("relative_degradation_pct", "std"),
            repeats=("mse", "size"),
        )
    )
    summary.to_csv(output / "robustness_summary.csv", index=False)
    (output / "robustness_protocol.json").write_text(
        json.dumps(
            {
                "datasets": DATASETS,
                "horizons": HORIZONS,
                "input_len": INPUT_LEN,
                "corruptions": CORRUPTIONS,
                "corruption_seeds": CORRUPTION_SEEDS,
                "model_seeds": MODEL_SEEDS,
                "missing_value_policy": "within-window forward fill; zero only if the first step is missing",
                "metric": "MSE and relative degradation against the same model on clean inputs",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def self_check() -> None:
    x = np.ones((5, INPUT_LEN, 7), dtype=np.float32)
    for kind in CORRUPTIONS:
        y = corrupt(x, kind, 31)
        assert y.shape == x.shape and np.isfinite(y).all()
    assert np.allclose(corrupt(x, "clean", 31), x)
    assert not np.allclose(corrupt(x, "spikes_0.02", 31), x)
    print("self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("references/external/ETDataset/ETT-small"))
    parser.add_argument("--model-root", type=Path, default=Path("outputs/ett_benchmark"))
    parser.add_argument("--output", type=Path, default=Path("outputs/robustness"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        self_check()
    else:
        print(run(args.data_root, args.model_root, args.output, torch.device(args.device)).to_string(index=False))
