"""Falsification-first text fusion with a validation-selected numerical fallback.

The calibration half of validation selects hyperparameters and the numerical
expert. The later validation half is reserved for pathway eligibility. Test
data is used once, after all choices are fixed.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import load_time_ordered_frame
from run_baselines import ALPHAS, ALL_CONFIGS, fit_ridge, make_windows, metrics, predict_ridge, seasonal_naive
from run_famets_selective import frequency_statistics, semantic_rows
from train_dlinear import DLinearTarget, TrainConfig, load_domain, make_windows as make_torch_windows, predict, seed_everything
from train_patchtst import PatchTSTTarget


SEEDS = (2026, 2027, 2028)
PERMUTATIONS = 99
RIDGE_ALPHAS = (*ALPHAS, 1000.0, 10000.0)
VARIANTS = ("semantic_residual", "frequency_residual")


class ResidualBuilder:
    def __init__(self, variant: str, components: int = 24):
        self.variant = variant
        self.components = components
        self.numeric_scaler = StandardScaler()
        self.quality_scaler = StandardScaler()
        self.frequency_scaler = StandardScaler()
        self.interaction_scaler = StandardScaler()
        self.pca: PCA | None = None

    def fit(self, x: np.ndarray, embeddings: np.ndarray, quality: np.ndarray) -> "ResidualBuilder":
        self.numeric_scaler.fit(x)
        self.quality_scaler.fit(quality)
        count = min(self.components, len(x) - 1, embeddings.shape[1])
        self.pca = PCA(n_components=count, svd_solver="full").fit(embeddings)
        if self.variant == "frequency_residual":
            frequency = frequency_statistics(x)
            self.frequency_scaler.fit(frequency)
            reduced = self.pca.transform(embeddings)
            frequency_z = self.frequency_scaler.transform(frequency)
            interaction = (reduced[:, :, None] * frequency_z[:, None, :8]).reshape(len(x), -1)
            self.interaction_scaler.fit(interaction)
        return self

    def transform(self, x: np.ndarray, embeddings: np.ndarray, quality: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("ResidualBuilder must be fitted before transform")
        reduced = self.pca.transform(embeddings)
        base = np.c_[self.numeric_scaler.transform(x), reduced, self.quality_scaler.transform(quality)]
        if self.variant == "semantic_residual":
            return base
        frequency_z = self.frequency_scaler.transform(frequency_statistics(x))
        interaction = (reduced[:, :, None] * frequency_z[:, None, :8]).reshape(len(x), -1)
        return np.c_[base, frequency_z, self.interaction_scaler.transform(interaction)]


def deep_model(
    model_class: type[nn.Module], model_args: dict[str, int | float], x_train: torch.Tensor,
    y_train: torch.Tensor, x_cal: torch.Tensor, y_cal: torch.Tensor, x_dec: torch.Tensor,
    config: TrainConfig, seed: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    seed_everything(seed)
    model = model_class(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
        cal = predict(model, x_cal, device, config.batch_size)
        value = metrics(y_cal.numpy(), cal)["mse"]
        if value < best_loss - config.min_delta:
            best_loss, best_epoch, stale = value, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("deep numerical model did not produce a checkpoint")
    model.load_state_dict(best_state)
    return (
        predict(model, x_cal, device, config.batch_size),
        predict(model, x_dec, device, config.batch_size),
        best_epoch,
        sum(parameter.numel() for parameter in model.parameters()),
        time.perf_counter() - started,
    )


def deep_refit(
    model_class: type[nn.Module], model_args: dict[str, int | float], x_fit: torch.Tensor,
    y_fit: torch.Tensor, x_test: torch.Tensor, epochs: list[int], config: TrainConfig,
    device: torch.device,
) -> np.ndarray:
    forecasts = []
    for seed, epoch_count in zip(SEEDS, epochs, strict=True):
        seed_everything(seed)
        model = model_class(**model_args).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        loader = DataLoader(
            TensorDataset(x_fit, y_fit), batch_size=config.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(seed), num_workers=0,
        )
        for _ in range(epoch_count):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(batch_x.to(device)), batch_y.to(device))
                loss.backward()
                optimizer.step()
        forecasts.append(predict(model, x_test, device, config.batch_size))
    return np.mean(forecasts, axis=0)


def fit_residual(
    variant: str, train: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    calibration: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[ResidualBuilder, Ridge, float, np.ndarray]:
    x_train, embeddings_train, quality_train, y_train = train
    x_cal, embeddings_cal, quality_cal, y_cal = calibration
    builder = ResidualBuilder(variant).fit(x_train, embeddings_train, quality_train)
    design_train = builder.transform(x_train, embeddings_train, quality_train)
    design_cal = builder.transform(x_cal, embeddings_cal, quality_cal)
    residual_train = y_train - x_train[:, -1, None]
    choices = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, solver="lsqr").fit(design_train, residual_train)
        prediction = x_cal[:, -1, None] + model.predict(design_cal)
        choices.append((metrics(y_cal, prediction)["mse"], alpha, model, prediction))
    _, alpha, model, prediction = min(choices, key=lambda item: item[0])
    return builder, model, alpha, prediction


def refit_residual(
    variant: str, alpha: float, fit_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    x_fit, embeddings_fit, quality_fit, y_fit = fit_data
    builder = ResidualBuilder(variant).fit(x_fit, embeddings_fit, quality_fit)
    model = Ridge(alpha=alpha, solver="lsqr").fit(
        builder.transform(x_fit, embeddings_fit, quality_fit), y_fit - x_fit[:, -1, None]
    )
    return test[0][:, -1, None] + model.predict(builder.transform(test[0], test[1], test[2]))


def moving_block_interval(
    actual: np.ndarray, fallback: np.ndarray, candidate: np.ndarray, block_length: int,
    seed: int = 2026, repeats: int = 5000,
) -> tuple[float, float, float]:
    difference = np.mean((fallback - actual) ** 2 - (candidate - actual) ** 2, axis=1)
    n = len(difference)
    length = min(max(2, block_length), n)
    starts = np.arange(n - length + 1)
    blocks = int(np.ceil(n / length))
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats)
    for i in range(repeats):
        indices = np.concatenate([np.arange(start, start + length) for start in rng.choice(starts, blocks)])[:n]
        samples[i] = difference[indices].mean()
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(difference.mean()), float(low), float(high)


def permutation_scores(
    variant: str, builder: ResidualBuilder, alpha: float,
    train: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    calibration: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    decision: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], count: int,
) -> list[float]:
    scores = []
    for seed in range(1001, 1001 + count):
        rng = np.random.default_rng(seed)
        train_order = rng.permutation(len(train[0]))
        cal_order = rng.permutation(len(calibration[0]))
        dec_order = rng.permutation(len(decision[0]))
        train_design = builder.transform(train[0], train[1][train_order], train[2][train_order])
        cal_design = builder.transform(calibration[0], calibration[1][cal_order], calibration[2][cal_order])
        residual = train[3] - train[0][:, -1, None]
        choices = []
        for candidate_alpha in RIDGE_ALPHAS:
            model = Ridge(alpha=candidate_alpha, solver="lsqr").fit(train_design, residual)
            pred = calibration[0][:, -1, None] + model.predict(cal_design)
            choices.append((metrics(calibration[3], pred)["mse"], candidate_alpha, model))
        _, _, model = min(choices, key=lambda item: item[0])
        decision_design = builder.transform(decision[0], decision[1][dec_order], decision[2][dec_order])
        pred = decision[0][:, -1, None] + model.predict(decision_design)
        scores.append(metrics(decision[3], pred)["mse"])
    return scores


def select_numeric(
    calibration: dict[str, np.ndarray], actual: np.ndarray,
) -> tuple[str, dict[str, float]]:
    scores = {name: metrics(actual, prediction)["mse"] for name, prediction in calibration.items()}
    return min(scores, key=scores.get), scores


def run(
    root: Path, semantic_root: Path, output: Path, device: torch.device,
    permutations: int = PERMUTATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output.mkdir(parents=True, exist_ok=True)
    training = TrainConfig(batch_size=64, learning_rate=3e-4, weight_decay=1e-4, max_epochs=60, patience=8, min_delta=1e-6)
    metric_rows, audit_rows = [], []
    for domain, config in ALL_CONFIGS.items():
        path = root / "numerical" / domain / f"{domain}.csv"
        ordered, order_audit = load_time_ordered_frame(path)
        raw = pd.to_numeric(ordered["OT"], errors="coerce").to_numpy(float)
        n, train_end, validation_end = len(raw), int(len(raw) * 0.7), int(len(raw) * 0.8)
        target_mean, target_std = raw[:train_end].mean(), raw[:train_end].std(ddof=0)
        target = (raw - target_mean) / target_std
        multi_values, features, target_index, _ = load_domain(path, train_end, "multivariate")
        cache = np.load(semantic_root / f"{domain}.npz")
        for horizon in config.horizons:
            arrays, origins = {}, {}
            for split, origin_range in {
                "train": range(config.input_len, train_end - horizon + 1),
                "validation": range(train_end, validation_end - horizon + 1),
                "test": range(validation_end, n - horizon + 1),
            }.items():
                x, y, origin = make_windows(target, config.input_len, horizon, origin_range)
                embedding, quality = semantic_rows(cache, origin)
                arrays[split], origins[split] = (x, embedding, quality, y), origin
            midpoint = len(arrays["validation"][0]) // 2
            calibration = tuple(value[:midpoint] for value in arrays["validation"])
            decision = tuple(value[midpoint:] for value in arrays["validation"])

            x_multi_train, y_torch_train, _ = make_torch_windows(
                multi_values, target_index, config.input_len, horizon,
                range(config.input_len, train_end - horizon + 1),
            )
            x_multi_val, y_torch_val, _ = make_torch_windows(
                multi_values, target_index, config.input_len, horizon,
                range(train_end, validation_end - horizon + 1),
            )
            x_multi_test, y_torch_test, _ = make_torch_windows(
                multi_values, target_index, config.input_len, horizon,
                range(validation_end, n - horizon + 1),
            )
            x_multi_cal, x_multi_dec = x_multi_val[:midpoint], x_multi_val[midpoint:]
            y_cal_t, y_dec_t = y_torch_val[:midpoint], y_torch_val[midpoint:]
            x_uni_train = torch.from_numpy(arrays["train"][0][:, :, None].astype(np.float32))
            x_uni_cal = torch.from_numpy(calibration[0][:, :, None].astype(np.float32))
            x_uni_dec = torch.from_numpy(decision[0][:, :, None].astype(np.float32))
            x_uni_test = torch.from_numpy(arrays["test"][0][:, :, None].astype(np.float32))
            y_train_t = torch.from_numpy(arrays["train"][3].astype(np.float32))
            y_test = arrays["test"][3]

            numeric_cal = {
                "Last": np.repeat(calibration[0][:, -1:], horizon, axis=1),
                "SeasonalNaive": seasonal_naive(calibration[0], horizon, config.seasonal_period),
            }
            numeric_dec = {
                "Last": np.repeat(decision[0][:, -1:], horizon, axis=1),
                "SeasonalNaive": seasonal_naive(decision[0], horizon, config.seasonal_period),
            }
            numeric_test = {
                "Last": np.repeat(arrays["test"][0][:, -1:], horizon, axis=1),
                "SeasonalNaive": seasonal_naive(arrays["test"][0], horizon, config.seasonal_period),
            }
            ridge_choices = []
            for alpha in ALPHAS:
                ridge = fit_ridge(arrays["train"][0], arrays["train"][3], alpha)
                prediction = predict_ridge(calibration[0], ridge)
                ridge_choices.append((metrics(calibration[3], prediction)["mse"], alpha, ridge))
            _, ridge_alpha, ridge = min(ridge_choices, key=lambda item: item[0])
            numeric_cal["AR-Ridge"] = predict_ridge(calibration[0], ridge)
            numeric_dec["AR-Ridge"] = predict_ridge(decision[0], ridge)
            ridge_refit = fit_ridge(
                np.r_[arrays["train"][0], arrays["validation"][0]],
                np.r_[arrays["train"][3], arrays["validation"][3]], ridge_alpha,
            )
            numeric_test["AR-Ridge"] = predict_ridge(arrays["test"][0], ridge_refit)

            deep_specs = {
                "DLinear-M": (
                    DLinearTarget,
                    {"input_len": config.input_len, "channels": len(features), "horizon": horizon,
                     "kernel_size": 7 if config.seasonal_period == 52 else 5},
                    (x_multi_train, x_multi_cal, x_multi_dec, x_multi_test),
                ),
                "PatchTST": (
                    PatchTSTTarget,
                    {"input_len": config.input_len, "channels": 1, "horizon": horizon,
                     "patch_len": 8 if config.input_len == 52 else 4,
                     "stride": 4 if config.input_len == 52 else 2,
                     "d_model": 32, "n_heads": 4, "e_layers": 2, "d_ff": 64, "dropout": 0.1},
                    (x_uni_train, x_uni_cal, x_uni_dec, x_uni_test),
                ),
            }
            deep_epochs, deep_meta = {}, {}
            for name, (model_class, model_args, xs) in deep_specs.items():
                cal_forecasts, dec_forecasts, epochs, times, parameters = [], [], [], [], 0
                for seed in SEEDS:
                    cal_pred, dec_pred, epoch, parameters, seconds = deep_model(
                        model_class, model_args, xs[0], y_train_t, xs[1], y_cal_t,
                        xs[2], training, seed, device,
                    )
                    cal_forecasts.append(cal_pred)
                    dec_forecasts.append(dec_pred)
                    epochs.append(epoch)
                    times.append(seconds)
                numeric_cal[name] = np.mean(cal_forecasts, axis=0)
                numeric_dec[name] = np.mean(dec_forecasts, axis=0)
                deep_epochs[name] = epochs
                deep_meta[name] = {"parameters": parameters, "training_seconds": float(sum(times))}

            fallback, numeric_cal_scores = select_numeric(numeric_cal, calibration[3])
            if fallback in deep_specs:
                model_class, model_args, xs = deep_specs[fallback]
                x_fit = torch.cat([xs[0], x_multi_val if fallback == "DLinear-M" else torch.from_numpy(arrays["validation"][0][:, :, None].astype(np.float32))])
                y_fit = torch.cat([y_train_t, y_torch_val])
                numeric_test[fallback] = deep_refit(
                    model_class, model_args, x_fit, y_fit, xs[3], deep_epochs[fallback], training, device
                )
            fallback_cal, fallback_dec, fallback_test = numeric_cal[fallback], numeric_dec[fallback], numeric_test[fallback]

            candidates, candidate_audit = {}, {}
            for variant in VARIANTS:
                builder, _, alpha, _ = fit_residual(variant, arrays["train"], calibration)
                decision_prediction = decision[0][:, -1, None] + Ridge(alpha=alpha, solver="lsqr").fit(
                    builder.transform(arrays["train"][0], arrays["train"][1], arrays["train"][2]),
                    arrays["train"][3] - arrays["train"][0][:, -1, None],
                ).predict(builder.transform(decision[0], decision[1], decision[2]))
                shuffled = permutation_scores(variant, builder, alpha, arrays["train"], calibration, decision, permutations)
                aligned_mse = metrics(decision[3], decision_prediction)["mse"]
                p_value = (1 + sum(value <= aligned_mse for value in shuffled)) / (permutations + 1)
                half = len(decision_prediction) // 2
                segment_wins = all(
                    metrics(actual, candidate)["mse"] < metrics(actual, baseline)["mse"]
                    for actual, candidate, baseline in (
                        (decision[3][:half], decision_prediction[:half], fallback_dec[:half]),
                        (decision[3][half:], decision_prediction[half:], fallback_dec[half:]),
                    )
                )
                eligible = aligned_mse < metrics(decision[3], fallback_dec)["mse"] and segment_wins and p_value <= 0.025
                test_prediction = refit_residual(
                    variant, alpha,
                    tuple(np.concatenate([arrays["train"][i], arrays["validation"][i]]) for i in range(4)),
                    arrays["test"],
                )
                candidates[variant] = test_prediction
                candidate_audit[variant] = {
                    "alpha": alpha, "decision_mse": aligned_mse,
                    "permutation_mean_mse": float(np.mean(shuffled)), "permutation_p_value": p_value,
                    "segment_wins": segment_wins, "eligible": eligible,
                }

            eligible = [variant for variant in VARIANTS if candidate_audit[variant]["eligible"]]
            selected = min(eligible, key=lambda name: candidate_audit[name]["decision_mse"]) if eligible else "numeric_fallback"
            selected_test = fallback_test if selected == "numeric_fallback" else candidates[selected]
            block = max(horizon, min(config.seasonal_period, 24))
            for name, prediction in {fallback: fallback_test, **candidates, "SafeFAME-TS-v2": selected_test}.items():
                result = metrics(y_test, prediction)
                delta, low, high = moving_block_interval(y_test, fallback_test, prediction, block)
                metric_rows.append({
                    "domain": domain, "horizon": horizon, "model": name, **result,
                    "improvement_vs_fallback_pct": 100.0 * (1.0 - result["mse"] / metrics(y_test, fallback_test)["mse"]),
                    "loss_difference_vs_fallback": delta, "block_ci_low": low, "block_ci_high": high,
                    "test_windows": len(y_test),
                })
            audit_rows.append({
                "domain": domain, "horizon": horizon, "numeric_fallback": fallback,
                "numeric_calibration_mse": numeric_cal_scores[fallback],
                "numeric_decision_mse": metrics(decision[3], fallback_dec)["mse"],
                "selected_path": selected, "calibration_windows": len(calibration[0]),
                "decision_windows": len(decision[0]), "test_windows": len(y_test),
                "ridge_alpha": ridge_alpha, "rows_reordered": order_audit["rows_moved_by_stable_sort"],
                **{f"{variant}_{key}": value for variant, values in candidate_audit.items() for key, value in values.items()},
            })
            print(f"{domain} H={horizon}: fallback={fallback}, selected={selected}", flush=True)

    metrics_frame, audit_frame = pd.DataFrame(metric_rows), pd.DataFrame(audit_rows)
    metrics_frame.to_csv(output / "safefame_v2_metrics.csv", index=False)
    audit_frame.to_csv(output / "safefame_v2_selection_audit.csv", index=False)
    (output / "safefame_v2_protocol.json").write_text(json.dumps({
        "split": {"train": 0.7, "validation": 0.1, "test": 0.2,
                  "validation_use": "first half calibration; second half untouched pathway decision"},
        "numeric_experts": ["Last", "SeasonalNaive", "AR-Ridge", "DLinear-M", "PatchTST"],
        "text_candidates": list(VARIANTS), "permutations": permutations,
        "permutation_p_threshold": 0.025,
        "selection": "candidate must beat the validation-selected numerical fallback on both decision halves and beat 97.5% of text permutations",
        "test_interval": "5000-repeat moving-block bootstrap of paired origin losses",
        "seeds": list(SEEDS), "training": asdict(training), "device": str(device),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics_frame, audit_frame


def self_check() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 24))
    embeddings = rng.normal(size=(80, 40))
    quality = rng.normal(size=(80, 10))
    y = np.repeat(x[:, -1:], 3, axis=1) + rng.normal(0, 0.1, size=(80, 3))
    builder, model, alpha, pred = fit_residual(
        "semantic_residual", (x[:50], embeddings[:50], quality[:50], y[:50]),
        (x[50:65], embeddings[50:65], quality[50:65], y[50:65]),
    )
    assert pred.shape == (15, 3) and alpha > 0 and model.coef_.shape[0] == 3
    features = builder.transform(x[65:], embeddings[65:], quality[65:])
    assert features.shape[0] == 15 and np.isfinite(features).all()
    delta, low, high = moving_block_interval(y[65:], y[65:] + 0.2, y[65:], 4, repeats=200)
    assert delta > 0 and low <= delta <= high
    print("self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("references/external/Time-MMD"))
    parser.add_argument("--semantic-root", type=Path, default=Path("data_processed/semantic_features"))
    parser.add_argument("--output", type=Path, default=Path("outputs/safefame_v2"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        self_check()
    else:
        run(args.root, args.semantic_root, args.output, torch.device(args.device), args.permutations)
