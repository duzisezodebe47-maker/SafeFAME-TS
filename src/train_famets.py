"""Train FAME-TS: frequency-aware, quality-gated multimodal expert fusion."""

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
from run_baselines import CONFIGS, make_windows


SEEDS = (2026, 2027, 2028)
VARIANTS = ("full", "no_text", "no_frequency", "shuffled_text")


@dataclass(frozen=True)
class GroupSpec:
    name: str
    domains: tuple[str, str]
    input_len: int
    horizons: tuple[int, int, int]
    seasonal_period: int
    moving_average: int


GROUPS = {
    "weekly": GroupSpec("weekly", ("Climate", "Energy"), 52, (4, 12, 24), 52, 7),
    "monthly": GroupSpec("monthly", ("Economy", "Traffic"), 24, (3, 6, 12), 12, 5),
}


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 120
    patience: int = 15
    min_delta: float = 1e-5
    text_gate_penalty: float = 2e-3


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split_origins(n: int, input_len: int, horizon: int) -> dict[str, range]:
    train_end, validation_end = int(n * 0.7), int(n * 0.8)
    return {
        "train": range(input_len, train_end - horizon + 1),
        "validation": range(train_end, validation_end - horizon + 1),
        "test": range(validation_end, n - horizon + 1),
    }


def _semantic_rows(cache: np.lib.npyio.NpzFile, origins: np.ndarray) -> tuple[np.ndarray, ...]:
    positions = {int(origin): index for index, origin in enumerate(cache["origin_index"])}
    indices = [positions[int(origin)] for origin in origins]
    quality = cache["quality"][indices].astype(np.float32)
    availability = np.c_[1.0 - quality[:, 4], 1.0 - quality[:, 5]].astype(np.float32)
    return (
        cache["report_embedding"][indices].astype(np.float32),
        cache["search_embedding"][indices].astype(np.float32),
        quality,
        availability,
    )


def prepare_group(
    root: Path, semantic_root: Path, spec: GroupSpec, variant: str, seed: int
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, object]]:
    pieces: dict[str, list[np.ndarray]] = {split: [] for split in ("train", "validation", "test")}
    metadata: dict[str, list[dict[str, object]]] = {split: [] for split in pieces}
    target_scales: dict[str, dict[str, float]] = {}

    for domain_index, domain in enumerate(spec.domains):
        frame, order_audit = load_time_ordered_frame(root / "numerical" / domain / f"{domain}.csv")
        raw = pd.to_numeric(frame["OT"], errors="coerce").to_numpy(float)
        if np.isnan(raw).any():
            raise ValueError(f"{domain} OT contains missing values")
        train_end = int(len(raw) * 0.7)
        mean, std = float(raw[:train_end].mean()), float(raw[:train_end].std())
        values = (raw - mean) / std
        target_scales[domain] = {"mean": mean, "std": std, **order_audit}
        cache = np.load(semantic_root / f"{domain}.npz")

        for horizon_index, horizon in enumerate(spec.horizons):
            for split, origin_range in _split_origins(len(raw), spec.input_len, horizon).items():
                x, y, origins = make_windows(values, spec.input_len, horizon, origin_range)
                report, search, quality, availability = _semantic_rows(cache, origins)
                padded_y = np.zeros((len(y), max(spec.horizons)), dtype=np.float32)
                mask = np.zeros_like(padded_y)
                padded_y[:, :horizon] = y.astype(np.float32)
                mask[:, :horizon] = 1.0
                domain_ids = np.full((len(y), 1), domain_index, dtype=np.int64)
                horizon_ids = np.full((len(y), 1), horizon_index, dtype=np.int64)
                task_ids = np.full((len(y), 1), domain_index * len(spec.horizons) + horizon_index, dtype=np.int64)
                block = np.concatenate(
                    [
                        x.astype(np.float32),
                        report,
                        search,
                        quality,
                        availability,
                        padded_y,
                        mask,
                        domain_ids.astype(np.float32),
                        horizon_ids.astype(np.float32),
                        task_ids.astype(np.float32),
                        origins[:, None].astype(np.float32),
                    ],
                    axis=1,
                )
                pieces[split].append(block)
                metadata[split].extend(
                    {"domain": domain, "horizon": horizon, "origin_index": int(origin)} for origin in origins
                )

    widths = {
        "x": spec.input_len,
        "report": 384,
        "search": 384,
        "quality": 10,
        "availability": 2,
        "y": max(spec.horizons),
        "mask": max(spec.horizons),
        "domain": 1,
        "horizon": 1,
        "task": 1,
        "origin": 1,
    }
    offsets: dict[str, slice] = {}
    cursor = 0
    for name, width in widths.items():
        offsets[name] = slice(cursor, cursor + width)
        cursor += width

    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split in pieces:
        merged = np.concatenate(pieces[split], axis=0)
        arrays[split] = {name: merged[:, section] for name, section in offsets.items()}

    quality_mean = arrays["train"]["quality"].mean(axis=0)
    quality_std = arrays["train"]["quality"].std(axis=0)
    quality_std[quality_std < 1e-6] = 1.0
    for split in arrays:
        arrays[split]["quality"] = (arrays[split]["quality"] - quality_mean) / quality_std

    if variant == "no_text":
        for split in arrays:
            arrays[split]["report"].fill(0.0)
            arrays[split]["search"].fill(0.0)
            arrays[split]["availability"].fill(0.0)
            arrays[split]["quality"].fill(0.0)
    elif variant == "shuffled_text":
        for split_index, split in enumerate(arrays):
            order = np.random.default_rng(seed * 100 + split_index).permutation(len(arrays[split]["report"]))
            for key in ("report", "search", "quality", "availability"):
                arrays[split][key] = arrays[split][key][order]

    train_tasks = arrays["train"]["task"].astype(int).ravel()
    counts = np.bincount(train_tasks, minlength=len(spec.domains) * len(spec.horizons))
    task_weights = len(train_tasks) / (len(counts) * counts)
    for split in arrays:
        ids = arrays[split]["task"].astype(int).ravel()
        arrays[split]["sample_weight"] = task_weights[ids, None].astype(np.float32)

    for split in arrays:
        if any(not np.isfinite(array).all() for array in arrays[split].values()):
            raise AssertionError(f"Non-finite values in {spec.name}/{variant}/{split}")
    info = {
        "target_scales": target_scales,
        "quality_train_mean": quality_mean.tolist(),
        "quality_train_std": quality_std.tolist(),
        "split_sizes": {split: len(arrays[split]["x"]) for split in arrays},
        "metadata": metadata,
    }
    return arrays, info


class FAMETS(nn.Module):
    def __init__(self, spec: GroupSpec, variant: str, embedding_dim: int = 384, hidden: int = 48):
        super().__init__()
        self.spec = spec
        self.variant = variant
        self.max_horizon = max(spec.horizons)
        self.frequency_bins = spec.input_len // 2 + 1
        spectrum_dim = self.frequency_bins * 2
        self.domain_embedding = nn.Embedding(len(spec.domains), 8)
        self.horizon_embedding = nn.Embedding(len(spec.horizons), 8)

        self.time_seasonal = nn.Linear(spec.input_len, self.max_horizon)
        self.time_trend = nn.Linear(spec.input_len, self.max_horizon)
        self.frequency_encoder = nn.Sequential(
            nn.Linear(spectrum_dim, hidden), nn.GELU(), nn.Dropout(0.1)
        )
        self.frequency_head = nn.Linear(hidden, self.max_horizon)
        self.low_frequency_encoder = nn.Linear(spectrum_dim, 32)
        self.high_frequency_encoder = nn.Linear(spectrum_dim, 32)
        self.numeric_gate = nn.Sequential(nn.Linear(6 + 16, 32), nn.GELU(), nn.Linear(32, 4))

        self.report_projection = nn.Sequential(nn.Linear(embedding_dim, 32), nn.GELU())
        self.search_projection = nn.Sequential(nn.Linear(embedding_dim, 32), nn.GELU())
        text_context_dim = 32 * 6 + 10 + 2 + 16 + 6
        self.text_context = nn.Sequential(
            nn.Linear(text_context_dim, 96), nn.GELU(), nn.Dropout(0.15), nn.Linear(96, hidden), nn.GELU()
        )
        self.text_residual = nn.Linear(hidden, self.max_horizon)
        self.text_gate = nn.Sequential(nn.Linear(hidden + 10 + 2 + 16, 32), nn.GELU(), nn.Linear(32, 1))

        low_mask = torch.zeros(spectrum_dim)
        cutoff = max(2, self.frequency_bins // 4)
        low_mask[:cutoff] = 1.0
        low_mask[self.frequency_bins : self.frequency_bins + cutoff] = 1.0
        self.register_buffer("low_mask", low_mask)
        self.register_buffer("high_mask", 1.0 - low_mask)

    def _moving_average(self, x: torch.Tensor) -> torch.Tensor:
        half = (self.spec.moving_average - 1) // 2
        padded = nn.functional.pad(x.unsqueeze(1), (half, half), mode="replicate")
        return nn.functional.avg_pool1d(padded, self.spec.moving_average, stride=1).squeeze(1)

    def forward(
        self,
        x: torch.Tensor,
        report: torch.Tensor,
        search: torch.Tensor,
        quality: torch.Tensor,
        availability: torch.Tensor,
        domain_id: torch.Tensor,
        horizon_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        trend = self._moving_average(x)
        time_prediction = self.time_seasonal(x - trend) + self.time_trend(trend)
        spectrum_complex = torch.fft.rfft(x, dim=1)
        spectrum = torch.cat([spectrum_complex.real, spectrum_complex.imag], dim=1)
        frequency_representation = self.frequency_encoder(spectrum)
        frequency_prediction = self.frequency_head(frequency_representation)
        persistence = x[:, -1:].repeat(1, self.max_horizon)
        seasonal_offsets = torch.arange(self.max_horizon, device=x.device) % self.spec.seasonal_period
        seasonal_prediction = x[:, -self.spec.seasonal_period :][:, seasonal_offsets]

        magnitudes = spectrum_complex.abs()
        split = max(2, self.frequency_bins // 4)
        total_energy = magnitudes.square().sum(dim=1).clamp_min(1e-6)
        low_energy = magnitudes[:, :split].square().sum(dim=1) / total_energy
        high_energy = 1.0 - low_energy
        slope = x[:, -1] - x[:, 0]
        numeric_statistics = torch.stack(
            [x.mean(1), x.std(1, unbiased=False), x[:, -1], slope, low_energy, high_energy], dim=1
        )
        domain = self.domain_embedding(domain_id)
        horizon = self.horizon_embedding(horizon_id)
        condition = torch.cat([numeric_statistics, domain, horizon], dim=1)
        expert_logits = self.numeric_gate(condition)
        if self.variant == "no_frequency":
            expert_logits = expert_logits.clone()
            expert_logits[:, 1] = -1e4
            frequency_prediction = torch.zeros_like(frequency_prediction)
        expert_weights = torch.softmax(expert_logits, dim=1)
        experts = torch.stack(
            [time_prediction, frequency_prediction, persistence, seasonal_prediction], dim=1
        )
        base_prediction = (expert_weights.unsqueeze(-1) * experts).sum(dim=1)

        if self.variant == "no_text":
            return base_prediction, torch.zeros(len(x), 1, device=x.device), expert_weights
        report_projection = self.report_projection(report)
        search_projection = self.search_projection(search)
        low_frequency = torch.tanh(self.low_frequency_encoder(spectrum * self.low_mask))
        high_frequency = torch.tanh(self.high_frequency_encoder(spectrum * self.high_mask))
        if self.variant == "no_frequency":
            low_frequency = torch.zeros_like(low_frequency)
            high_frequency = torch.zeros_like(high_frequency)
        text_features = torch.cat(
            [
                report_projection,
                search_projection,
                report_projection * low_frequency,
                report_projection * high_frequency,
                search_projection * low_frequency,
                search_projection * high_frequency,
                quality,
                availability,
                domain,
                horizon,
                numeric_statistics,
            ],
            dim=1,
        )
        text_context = self.text_context(text_features)
        residual = self.text_residual(text_context)
        gate_features = torch.cat([text_context, quality, availability, domain, horizon], dim=1)
        has_text = availability.max(dim=1, keepdim=True).values
        gate = torch.sigmoid(self.text_gate(gate_features)) * has_text
        return base_prediction + gate * residual, gate, expert_weights


TENSOR_KEYS = (
    "x", "report", "search", "quality", "availability", "domain", "horizon", "y", "mask", "task", "sample_weight"
)


def to_dataset(arrays: dict[str, np.ndarray]) -> TensorDataset:
    tensors: list[torch.Tensor] = []
    for key in TENSOR_KEYS:
        array = arrays[key]
        if key in ("domain", "horizon", "task"):
            tensors.append(torch.from_numpy(array.astype(np.int64).ravel()))
        else:
            tensors.append(torch.from_numpy(array.astype(np.float32)))
    return TensorDataset(*tensors)


def forward_batch(model: FAMETS, batch: tuple[torch.Tensor, ...], device: torch.device):
    moved = [tensor.to(device) for tensor in batch]
    x, report, search, quality, availability, domain, horizon, y, mask, task, sample_weight = moved
    prediction, text_gate, expert_weights = model(
        x, report, search, quality, availability, domain, horizon
    )
    return prediction, text_gate, expert_weights, y, mask, task, sample_weight


def weighted_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    per_sample = ((prediction - target).square() * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    return (per_sample * sample_weight.ravel()).mean()


@torch.inference_mode()
def evaluate(model: FAMETS, dataset: TensorDataset, device: torch.device, batch_size: int):
    model.eval()
    outputs = []
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        prediction, text_gate, expert_weights, y, mask, task, _ = forward_batch(model, batch, device)
        outputs.append(
            (
                prediction.cpu().numpy(), y.cpu().numpy(), mask.cpu().numpy(),
                task.cpu().numpy(), text_gate.cpu().numpy(), expert_weights.cpu().numpy(),
            )
        )
    return tuple(np.concatenate([item[index] for item in outputs]) for index in range(6))


def macro_task_mse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray, task: np.ndarray) -> float:
    sample_error = ((prediction - target) ** 2 * mask).sum(1) / mask.sum(1)
    return float(np.mean([sample_error[task == value].mean() for value in np.unique(task)]))


def fit_model(
    spec: GroupSpec,
    variant: str,
    arrays: dict[str, dict[str, np.ndarray]],
    config: TrainConfig,
    seed: int,
    device: torch.device,
) -> tuple[FAMETS, list[dict[str, object]], int, float, float]:
    seed_everything(seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = FAMETS(spec, variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_dataset = to_dataset(arrays["train"])
    validation_dataset = to_dataset(arrays["validation"])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0
    )
    best_validation = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, gate, _, target, mask, _, sample_weight = forward_batch(model, batch, device)
            loss = weighted_loss(prediction, target, mask, sample_weight)
            if variant != "no_text":
                loss = loss + config.text_gate_penalty * gate.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch[0])
        val_prediction, val_target, val_mask, val_task, _, _ = evaluate(
            model, validation_dataset, device, config.batch_size
        )
        val_mse = macro_task_mse(val_prediction, val_target, val_mask, val_task)
        history.append(
            {"phase": "model_selection", "epoch": epoch, "train_loss": total / len(train_dataset), "validation_macro_mse": val_mse}
        )
        if val_mse < best_validation - config.min_delta:
            best_validation = val_mse
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    # Refit with the selected epoch count on train+validation; test remains sealed.
    seed_everything(seed)
    model = FAMETS(spec, variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_tensors = to_dataset(arrays["train"]).tensors
    validation_tensors = to_dataset(arrays["validation"]).tensors
    combined = TensorDataset(
        *[torch.cat([train_tensors[index], validation_tensors[index]]) for index in range(len(TENSOR_KEYS))]
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(combined, batch_size=config.batch_size, shuffle=True, generator=generator, num_workers=0)
    for epoch in range(1, best_epoch + 1):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction, gate, _, target, mask, _, sample_weight = forward_batch(model, batch, device)
            loss = weighted_loss(prediction, target, mask, sample_weight)
            if variant != "no_text":
                loss = loss + config.text_gate_penalty * gate.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(batch[0])
        history.append(
            {"phase": "refit_train_plus_validation", "epoch": epoch, "train_loss": total / len(combined), "validation_macro_mse": np.nan}
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    return model, history, best_epoch, elapsed, peak


def score_test(
    model: FAMETS,
    arrays: dict[str, np.ndarray],
    metadata: list[dict[str, object]],
    spec: GroupSpec,
    device: torch.device,
    batch_size: int,
) -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    prediction, target, mask, _, gates, expert_weights = evaluate(
        model, to_dataset(arrays), device, batch_size
    )
    meta = pd.DataFrame(metadata)
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    gate_rows: list[pd.DataFrame] = []
    for (domain, horizon), indices in meta.groupby(["domain", "horizon"]).groups.items():
        indices = np.asarray(list(indices))
        actual = target[indices, :horizon]
        forecast = prediction[indices, :horizon]
        error = forecast - actual
        metric_rows.append(
            {
                "domain": domain,
                "horizon": horizon,
                "mse": float(np.mean(error**2)),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "test_windows": len(indices),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "domain": domain,
                    "horizon": horizon,
                    "origin_index": np.repeat(meta.loc[indices, "origin_index"].to_numpy(), horizon),
                    "step": np.tile(np.arange(1, horizon + 1), len(indices)),
                    "actual_z": actual.ravel(),
                    "prediction_z": forecast.ravel(),
                }
            )
        )
        gate_rows.append(
            pd.DataFrame(
                {
                    "domain": domain,
                    "horizon": horizon,
                    "origin_index": meta.loc[indices, "origin_index"].to_numpy(),
                    "text_gate": gates[indices, 0],
                    "time_weight": expert_weights[indices, 0],
                    "frequency_weight": expert_weights[indices, 1],
                    "persistence_weight": expert_weights[indices, 2],
                    "seasonal_weight": expert_weights[indices, 3],
                }
            )
        )
    return metric_rows, pd.concat(prediction_rows, ignore_index=True), pd.concat(gate_rows, ignore_index=True)


def run(
    root: Path,
    semantic_root: Path,
    output: Path,
    groups: list[str],
    variants: list[str],
    seeds: tuple[int, ...],
    config: TrainConfig,
    device: torch.device,
) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    (output / "models").mkdir(exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    gates: list[pd.DataFrame] = []
    run_manifest: dict[str, object] = {}
    for group_name in groups:
        spec = GROUPS[group_name]
        for variant in variants:
            for seed in seeds:
                arrays, info = prepare_group(root, semantic_root, spec, variant, seed)
                model, history, best_epoch, seconds, peak = fit_model(
                    spec, variant, arrays, config, seed, device
                )
                metric_rows, prediction_frame, gate_frame = score_test(
                    model, arrays["test"], info["metadata"]["test"], spec, device, config.batch_size
                )
                parameters = sum(parameter.numel() for parameter in model.parameters())
                for row in metric_rows:
                    row.update(
                        {
                            "group": group_name,
                            "variant": variant,
                            "seed": seed,
                            "model": "FAME-TS",
                            "best_epoch": best_epoch,
                            "training_seconds": seconds,
                            "peak_gpu_mb": peak,
                            "parameters": parameters,
                        }
                    )
                    print(
                        f"{group_name}/{variant}/seed={seed} {row['domain']} H={row['horizon']} "
                        f"MSE={row['mse']:.6f} epoch={best_epoch}",
                        flush=True,
                    )
                prediction_frame["group"] = group_name
                prediction_frame["variant"] = variant
                prediction_frame["seed"] = seed
                gate_frame["group"] = group_name
                gate_frame["variant"] = variant
                gate_frame["seed"] = seed
                rows.extend(metric_rows)
                predictions.append(prediction_frame)
                gates.append(gate_frame)
                torch.save(
                    {"state_dict": model.state_dict(), "group": asdict(spec), "variant": variant},
                    output / "models" / f"{group_name}_{variant}_s{seed}.pt",
                )
                pd.DataFrame(history).to_csv(
                    output / "logs" / f"{group_name}_{variant}_s{seed}.csv", index=False
                )
                run_manifest[f"{group_name}/{variant}/{seed}"] = {
                    "best_epoch": best_epoch,
                    "training_seconds": seconds,
                    "peak_gpu_mb": peak,
                    "data": {key: value for key, value in info.items() if key != "metadata"},
                }

    by_seed = pd.DataFrame(rows)
    summary = (
        by_seed.groupby(["domain", "horizon", "group", "variant", "model"], as_index=False)
        .agg(
            mse_mean=("mse", "mean"), mse_std=("mse", "std"),
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            training_seconds_mean=("training_seconds", "mean"),
            peak_gpu_mb=("peak_gpu_mb", "max"), parameters=("parameters", "first"),
        )
    )
    by_seed.to_csv(output / "famets_metrics_by_seed.csv", index=False)
    summary.to_csv(output / "famets_metrics_summary.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "famets_predictions.csv", index=False)
    pd.concat(gates, ignore_index=True).to_csv(output / "famets_gates.csv", index=False)
    (output / "famets_run.json").write_text(
        json.dumps(
            {
                "groups": {name: asdict(GROUPS[name]) for name in groups},
                "variants": variants,
                "seeds": seeds,
                "train_config": asdict(config),
                "device": str(device),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "runs": run_manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def self_check() -> None:
    spec = GROUPS["monthly"]
    model = FAMETS(spec, "full")
    batch = 5
    prediction, gate, weights = model(
        torch.randn(batch, spec.input_len),
        torch.randn(batch, 384),
        torch.randn(batch, 384),
        torch.randn(batch, 10),
        torch.tensor([[1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        torch.tensor([0, 0, 1, 1, 0]),
        torch.tensor([0, 1, 2, 0, 1]),
    )
    assert prediction.shape == (batch, max(spec.horizons))
    assert gate.shape == (batch, 1) and gate[1].item() == 0.0
    assert torch.allclose(weights.sum(1), torch.ones(batch), atol=1e-6)
    mask = torch.ones_like(prediction)
    loss = weighted_loss(prediction, torch.zeros_like(prediction), mask, torch.ones(batch, 1))
    loss.backward()
    assert torch.isfinite(loss)
    print("self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("references/external/Time-MMD"))
    parser.add_argument("--semantic-root", type=Path, default=Path("data_processed/semantic_features"))
    parser.add_argument("--output", type=Path, default=Path("outputs/famets"))
    parser.add_argument("--groups", nargs="+", choices=tuple(GROUPS), default=list(GROUPS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        self_check()
    else:
        result = run(
            args.root,
            args.semantic_root,
            args.output,
            args.groups,
            args.variants,
            tuple(args.seeds),
            TrainConfig(),
            torch.device(args.device),
        )
        print(result.to_string(index=False))
