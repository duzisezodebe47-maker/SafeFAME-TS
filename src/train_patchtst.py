"""Train the official PatchTST backbone under the project's frozen protocol."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from run_baselines import CONFIGS
from train_dlinear import SEEDS, TrainConfig, load_domain, make_windows, predict, score, train_one


PATCHTST_ROOT = Path(__file__).resolve().parents[1] / "references" / "external" / "PatchTST" / "PatchTST_supervised"
sys.path.insert(0, str(PATCHTST_ROOT))
from models.PatchTST import Model as OfficialPatchTST  # noqa: E402


class PatchTSTTarget(nn.Module):
    def __init__(
        self,
        input_len: int,
        channels: int,
        horizon: int,
        patch_len: int,
        stride: int,
        d_model: int = 32,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        configs = SimpleNamespace(
            enc_in=channels,
            seq_len=input_len,
            pred_len=horizon,
            e_layers=e_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            fc_dropout=dropout,
            head_dropout=0.0,
            individual=False,
            patch_len=patch_len,
            stride=stride,
            padding_patch="end",
            revin=True,
            affine=True,
            subtract_last=False,
            decomposition=False,
            kernel_size=25,
        )
        self.model = OfficialPatchTST(configs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)[:, :, 0]


def run(root: Path, output: Path, config: TrainConfig, device: torch.device) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    model_dir, log_dir = output / "models", output / "logs"
    model_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    feature_audit: dict[str, object] = {}

    for domain, domain_config in CONFIGS.items():
        path = root / "numerical" / domain / f"{domain}.csv"
        row_count = len(pd.read_csv(path, usecols=["OT"]))
        train_end, val_end = int(row_count * 0.7), int(row_count * 0.8)
        values, features, target_index, audit = load_domain(path, train_end, "univariate")
        feature_audit[domain] = audit
        patch_len = 8 if domain_config.input_len == 52 else 4
        stride = patch_len // 2

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
                "channels": 1,
                "horizon": horizon,
                "patch_len": patch_len,
                "stride": stride,
                "d_model": 32,
                "n_heads": 4,
                "e_layers": 2,
                "d_ff": 64,
                "dropout": 0.1,
            }
            for seed in SEEDS:
                model, history, best_epoch, seconds = train_one(
                    arrays, model_args, config, seed, device, model_class=PatchTSTTarget
                )
                forecast = predict(model, x_test, device, config.batch_size)
                result = score(y_test.numpy(), forecast)
                result.update(
                    {
                        "domain": domain,
                        "horizon": horizon,
                        "model": "PatchTST",
                        "seed": seed,
                        "channels": 1,
                        "parameters": sum(parameter.numel() for parameter in model.parameters()),
                        "best_epoch": best_epoch,
                        "training_seconds": seconds,
                        "peak_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0,
                        "train_windows": len(x_train),
                        "validation_windows": len(x_val),
                        "test_windows": len(x_test),
                    }
                )
                rows.append(result)
                print(
                    f"{domain} PatchTST H={horizon} seed={seed} "
                    f"MSE={result['mse']:.6f} epoch={best_epoch} time={seconds:.2f}s",
                    flush=True,
                )
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "domain": domain,
                            "horizon": horizon,
                            "model": "PatchTST",
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

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby(["domain", "horizon", "model"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"), mse_std=("mse", "std"),
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            training_seconds_mean=("training_seconds", "mean"),
            peak_gpu_mb=("peak_gpu_mb", "max"), parameters=("parameters", "first"),
        )
    )
    metrics.to_csv(output / "patchtst_metrics_by_seed.csv", index=False)
    summary.to_csv(output / "patchtst_metrics_summary.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output / "patchtst_predictions.csv", index=False)
    (output / "patchtst_run.json").write_text(
        json.dumps(
            {
                "implementation": str(PATCHTST_ROOT),
                "license": "Apache-2.0",
                "device": str(device),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "seeds": SEEDS,
                "train_config": asdict(config),
                "domain_configs": {name: asdict(value) for name, value in CONFIGS.items()},
                "feature_audit": feature_audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def self_check() -> None:
    model = PatchTSTTarget(24, 1, 6, 4, 2)
    output = model(torch.randn(8, 24, 1))
    assert output.shape == (8, 6)
    print("self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("references/external/Time-MMD"))
    parser.add_argument("--output", type=Path, default=Path("outputs/patchtst"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        self_check()
    else:
        device = torch.device(args.device)
        training = TrainConfig(
            batch_size=64,
            learning_rate=3e-4,
            weight_decay=1e-4,
            max_epochs=100,
            patience=10,
            min_delta=1e-6,
        )
        print(run(args.root, args.output, training, device).to_string(index=False))
