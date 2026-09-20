"""Train a leakage-safe multivariate DLinear baseline on Time-MMD."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import load_time_ordered_frame
from run_baselines import CONFIGS, DomainConfig


SEEDS = (2026, 2027, 2028)
DATE_COLUMNS = {"date", "Date", "start_date", "end_date", "MapDate", "Month"}


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-6


class DLinearTarget(nn.Module):
    """DLinear decomposition with all numeric channels predicting target OT."""

    def __init__(self, input_len: int, channels: int, horizon: int, kernel_size: int):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        features = input_len * channels
        self.seasonal = nn.Linear(features, horizon)
        self.trend = nn.Linear(features, horizon)
        nn.init.constant_(self.seasonal.weight, 1.0 / features)
        nn.init.constant_(self.trend.weight, 1.0 / features)
        nn.init.zeros_(self.seasonal.bias)
        nn.init.zeros_(self.trend.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half = (self.kernel_size - 1) // 2
        channel_first = x.transpose(1, 2)
        padded = nn.functional.pad(channel_first, (half, half), mode="replicate")
        trend = nn.functional.avg_pool1d(padded, kernel_size=self.kernel_size, stride=1).transpose(1, 2)
        seasonal = x - trend
        return self.seasonal(seasonal.flatten(1)) + self.trend(trend.flatten(1))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_domain(
    path: Path, train_end: int, feature_mode: str
) -> tuple[np.ndarray, list[str], int, dict[str, object]]:
    frame, order_audit = load_time_ordered_frame(path)
    numeric = frame.drop(columns=[column for column in DATE_COLUMNS if column in frame], errors="ignore")
    numeric = numeric.apply(pd.to_numeric, errors="coerce")
    if "OT" not in numeric:
        raise ValueError(f"{path} has no numeric OT column")

    train = numeric.iloc[:train_end]
    usable = [
        column
        for column in numeric.columns
        if train[column].notna().any() and float(train[column].std(skipna=True) or 0.0) > 1e-8
    ]
    if "OT" not in usable:
        raise ValueError(f"{path} has unusable OT values in training segment")
    if feature_mode == "univariate":
        usable = ["OT"]
    numeric = numeric[usable].ffill()
    medians = numeric.iloc[:train_end].median()
    numeric = numeric.fillna(medians)
    if numeric.isna().any().any():
        raise ValueError(f"{path} still has missing values after causal fill and train median fill")

    mean = numeric.iloc[:train_end].mean()
    std = numeric.iloc[:train_end].std(ddof=0).replace(0.0, 1.0)
    scaled = ((numeric - mean) / std).to_numpy(np.float32)
    target_index = usable.index("OT")
    audit = {
        **order_audit,
        "features": usable,
        "dropped_columns": [column for column in frame.columns if column not in usable],
        "train_mean": {column: float(mean[column]) for column in usable},
        "train_std": {column: float(std[column]) for column in usable},
        "imputation": "historical forward fill, then training-segment median for leading missing values",
    }
    return scaled, usable, target_index, audit


def make_windows(
    values: np.ndarray,
    target_index: int,
    input_len: int,
    horizon: int,
    origins: range,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    valid = [i for i in origins if i >= input_len and i + horizon <= len(values)]
    x = np.stack([values[i - input_len : i] for i in valid])
    y = np.stack([values[i : i + horizon, target_index] for i in valid])
    return torch.from_numpy(x), torch.from_numpy(y), np.asarray(valid)


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


@torch.inference_mode()
def predict(model: nn.Module, x: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False, num_workers=0)
    return np.concatenate([model(batch[0].to(device)).cpu().numpy() for batch in loader])


def train_one(
    arrays: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    model_args: dict[str, int],
    config: TrainConfig,
    seed: int,
    device: torch.device,
    model_class: type[nn.Module] = DLinearTarget,
) -> tuple[nn.Module, list[dict[str, float]], int, float]:
    x_train, y_train, x_val, y_val, _, _ = arrays
    seed_everything(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = model_class(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        loss_sum = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_x)
        val_prediction = predict(model, x_val, device, config.batch_size)
        val_mse = score(y_val.numpy(), val_prediction)["mse"]
        history.append({"epoch": epoch, "train_mse": loss_sum / len(x_train), "validation_mse": val_mse})
        if val_mse < best_val - config.min_delta:
            best_val = val_mse
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    # Refit from scratch on train+validation for the selected number of epochs.
    # This matches the deterministic ridge protocol while keeping test data sealed.
    seed_everything(seed)
    model = model_class(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    refit_x = torch.cat([x_train, x_val])
    refit_y = torch.cat([y_train, y_val])
    generator = torch.Generator().manual_seed(seed)
    refit_loader = DataLoader(
        TensorDataset(refit_x, refit_y),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    for epoch in range(1, best_epoch + 1):
        model.train()
        loss_sum = 0.0
        for batch_x, batch_y in refit_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_x)
        history.append(
            {
                "epoch": epoch,
                "train_mse": loss_sum / len(refit_x),
                "validation_mse": np.nan,
                "phase": "refit_train_plus_validation",
            }
        )
    for row in history[: len(history) - best_epoch]:
        row["phase"] = "model_selection"
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model, history, best_epoch, time.perf_counter() - started


def run(
    root: Path,
    output: Path,
    train_config: TrainConfig,
    device: torch.device,
    feature_mode: str,
) -> pd.DataFrame:
    metrics_rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    audits: dict[str, object] = {}
    model_dir = output / "models"
    log_dir = output / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    for domain, domain_config in CONFIGS.items():
        path = root / "numerical" / domain / f"{domain}.csv"
        row_count = len(pd.read_csv(path, usecols=["OT"]))
        train_end, val_end = int(row_count * 0.7), int(row_count * 0.8)
        values, features, target_index, audit = load_domain(path, train_end, feature_mode)
        audits[domain] = audit
        kernel_size = 7 if domain_config.seasonal_period == 52 else 5

        for horizon in domain_config.horizons:
            x_train, y_train, _ = make_windows(
                values, target_index, domain_config.input_len, horizon,
                range(domain_config.input_len, train_end - horizon + 1),
            )
            x_val, y_val, _ = make_windows(
                values, target_index, domain_config.input_len, horizon,
                range(train_end, val_end - horizon + 1),
            )
            x_test, y_test, origins = make_windows(
                values, target_index, domain_config.input_len, horizon,
                range(val_end, row_count - horizon + 1),
            )
            arrays = (x_train, y_train, x_val, y_val, x_test, y_test)
            model_args = {
                "input_len": domain_config.input_len,
                "channels": len(features),
                "horizon": horizon,
                "kernel_size": kernel_size,
            }
            for seed in SEEDS:
                model, history, best_epoch, seconds = train_one(arrays, model_args, train_config, seed, device)
                forecast = predict(model, x_test, device, train_config.batch_size)
                result = score(y_test.numpy(), forecast)
                result.update(
                    {
                        "domain": domain,
                        "horizon": horizon,
                        "model": "DLinear-U" if feature_mode == "univariate" else "DLinear-M",
                        "seed": seed,
                        "channels": len(features),
                        "parameters": sum(parameter.numel() for parameter in model.parameters()),
                        "best_epoch": best_epoch,
                        "training_seconds": seconds,
                        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0,
                        "train_windows": len(x_train),
                        "validation_windows": len(x_val),
                        "test_windows": len(x_test),
                    }
                )
                metrics_rows.append(result)
                print(
                    f"{domain} {feature_mode} H={horizon} seed={seed} "
                    f"MSE={result['mse']:.6f} epoch={best_epoch} time={seconds:.2f}s",
                    flush=True,
                )
                predictions.append(
                    pd.DataFrame(
                        {
                            "domain": domain,
                            "horizon": horizon,
                            "model": "DLinear-U" if feature_mode == "univariate" else "DLinear-M",
                            "seed": seed,
                            "origin_index": np.repeat(origins, horizon),
                            "step": np.tile(np.arange(1, horizon + 1), len(origins)),
                            "actual_z": y_test.numpy().ravel(),
                            "prediction_z": forecast.ravel(),
                        }
                    )
                )
                torch.save(
                    {"state_dict": model.state_dict(), "model_args": model_args, "features": features},
                    model_dir / f"{domain}_h{horizon}_s{seed}.pt",
                )
                pd.DataFrame(history).to_csv(log_dir / f"{domain}_h{horizon}_s{seed}.csv", index=False)

    metrics_frame = pd.DataFrame(metrics_rows)
    summary = (
        metrics_frame.groupby(["domain", "horizon", "model"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"), mse_std=("mse", "std"),
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            training_seconds_mean=("training_seconds", "mean"),
            peak_gpu_mb=("peak_gpu_mb", "max"), parameters=("parameters", "first"),
        )
    )
    metrics_frame.to_csv(output / "dlinear_metrics_by_seed.csv", index=False)
    summary.to_csv(output / "dlinear_metrics_summary.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "dlinear_predictions.csv", index=False)
    (output / "dlinear_run.json").write_text(
        json.dumps(
            {
                "device": str(device),
                "feature_mode": feature_mode,
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "seeds": SEEDS,
                "train_config": asdict(train_config),
                "domain_configs": {name: asdict(value) for name, value in CONFIGS.items()},
                "feature_audit": audits,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def self_check() -> None:
    seed_everything(1)
    model = DLinearTarget(input_len=12, channels=3, horizon=4, kernel_size=5)
    output = model(torch.randn(8, 12, 3))
    assert output.shape == (8, 4)
    values = np.random.default_rng(1).normal(size=(100, 3)).astype(np.float32)
    x, y, origins = make_windows(values, 1, 12, 4, range(70, 80))
    assert x.shape == (10, 12, 3) and y.shape == (10, 4) and len(origins) == 10
    print("self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("references/external/Time-MMD"))
    parser.add_argument("--output", type=Path, default=Path("outputs/dlinear"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-mode", choices=("univariate", "multivariate"), default="multivariate")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        self_check()
    else:
        selected_device = torch.device(args.device)
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        print(run(args.root, args.output, TrainConfig(), selected_device, args.feature_mode).to_string(index=False))
