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
import hashlib
import platform
from datetime import datetime, timezone
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
from validation_boundaries import validation_masks, decision_segments


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
    config: TrainConfig, seed: int, device: torch.device, artifact: Path | None = None,
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
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
        cal = predict(model, x_cal, device, config.batch_size)
        value = metrics(y_cal.numpy(), cal)["mse"]
        history.append({"epoch": epoch, "calibration_mse": value})
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
    if artifact is not None:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "model_args": model_args,
                    "seed": seed, "best_epoch": best_epoch}, artifact.with_suffix('.pt'))
        pd.DataFrame(history).to_csv(artifact.with_suffix('.csv'), index=False)
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
    device: torch.device, artifact: Path | None = None,
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
        if artifact is not None:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "model_args": model_args,
                        "seed": seed, "epochs": epoch_count},
                       artifact.parent / f'{artifact.name}_s{seed}.pt')
    if artifact is not None:
        np.savez_compressed(artifact.with_suffix('.npz'), predictions=np.stack(forecasts), seeds=SEEDS)
    return np.mean(forecasts, axis=0)


def fit_residual(
    variant: str, train: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    calibration: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    solver: str = "lsqr",
) -> tuple[ResidualBuilder, Ridge, float, np.ndarray]:
    x_train, embeddings_train, quality_train, y_train = train
    x_cal, embeddings_cal, quality_cal, y_cal = calibration
    builder = ResidualBuilder(variant).fit(x_train, embeddings_train, quality_train)
    design_train = builder.transform(x_train, embeddings_train, quality_train)
    design_cal = builder.transform(x_cal, embeddings_cal, quality_cal)
    residual_train = y_train - x_train[:, -1, None]
    choices = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, solver=solver).fit(design_train, residual_train)
        prediction = x_cal[:, -1, None] + model.predict(design_cal)
        choices.append((metrics(y_cal, prediction)["mse"], alpha, model, prediction))
    _, alpha, model, prediction = min(choices, key=lambda item: item[0])
    return builder, model, alpha, prediction


def refit_residual(
    variant: str, alpha: float, fit_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    solver: str = "lsqr",
) -> np.ndarray:
    x_fit, embeddings_fit, quality_fit, y_fit = fit_data
    builder = ResidualBuilder(variant).fit(x_fit, embeddings_fit, quality_fit)
    model = Ridge(alpha=alpha, solver=solver).fit(
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
    solver: str = "lsqr",
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
            model = Ridge(alpha=candidate_alpha, solver=solver).fit(train_design, residual)
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
    purged: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output.mkdir(parents=True, exist_ok=True)
    prefix = 'safefame_v3' if purged else 'safefame_v2'
    solver = 'cholesky' if purged else 'lsqr'
    if purged:
        from threadpoolctl import threadpool_info
        frozen = {
            'revision': 'v3_target_disjoint', 'role': 'post-audit correction; previously viewed test periods; not prospective confirmation',
            'started_utc': datetime.now(timezone.utc).isoformat(),
            'permutations': permutations, 'threshold': 0.025, 'bootstrap_repeats': 5000,
            'boundary': 'raw validation midpoint; discard H-1 crossing origins; same rule within decision subperiods',
            'insufficient_segments': 'ineligible when either target-disjoint decision subperiod has no complete window',
            'solver': solver, 'seeds': SEEDS, 'python': platform.python_version(),
            'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': str(device),
            'threads': threadpool_info(), 'torch_threads': torch.get_num_threads(),
            'sha256': {p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [*Path(__file__).resolve().parent.glob('*.py'),
                                 *semantic_root.glob('*.npz'),
                                 *[root/'numerical'/d/f'{d}.csv' for d in ALL_CONFIGS]]},
        }
        (output/'run_manifest.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding='utf-8')
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
            task_dir = output/'tasks'/f'{domain}_h{horizon}' if purged else None
            if task_dir is not None:
                task_dir.mkdir(parents=True, exist_ok=True)
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
            cal_mask = np.arange(len(origins['validation'])) < midpoint
            dec_mask = ~cal_mask
            boundary = int(origins['validation'][midpoint])
            if purged:
                cal_mask, dec_mask, boundary = validation_masks(origins['validation'], horizon, train_end, validation_end)
            calibration = tuple(value[cal_mask] for value in arrays["validation"])
            decision = tuple(value[dec_mask] for value in arrays["validation"])
            dec_origins = origins['validation'][dec_mask]
            segment_masks = (np.arange(len(dec_origins)) < len(dec_origins)//2,
                             np.arange(len(dec_origins)) >= len(dec_origins)//2)
            if purged:
                seg1, seg2, _ = decision_segments(dec_origins, horizon, boundary, validation_end)
                segment_masks = (seg1, seg2)
            task_evidence = {'calibration_origins': origins['validation'][cal_mask],
                             'decision_origins': dec_origins, 'test_origins': origins['test'],
                             'calibration_actual': calibration[3], 'decision_actual': decision[3],
                             'test_actual': arrays['test'][3],
                             'segment1_mask': segment_masks[0], 'segment2_mask': segment_masks[1]}

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
            x_multi_cal, x_multi_dec = x_multi_val[cal_mask], x_multi_val[dec_mask]
            y_cal_t, y_dec_t = y_torch_val[cal_mask], y_torch_val[dec_mask]
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
                        task_dir/'models'/f'{name}_cal_s{seed}' if purged else None,
                    )
                    cal_forecasts.append(cal_pred)
                    dec_forecasts.append(dec_pred)
                    epochs.append(epoch)
                    times.append(seconds)
                    if purged:
                        task_evidence[f'{name}_s{seed}_calibration'] = cal_pred
                        task_evidence[f'{name}_s{seed}_decision'] = dec_pred
                numeric_cal[name] = np.mean(cal_forecasts, axis=0)
                numeric_dec[name] = np.mean(dec_forecasts, axis=0)
                deep_epochs[name] = epochs
                deep_meta[name] = {"parameters": parameters, "training_seconds": float(sum(times))}

            fallback, numeric_cal_scores = select_numeric(numeric_cal, calibration[3])
            if fallback in deep_specs and not purged:
                model_class, model_args, xs = deep_specs[fallback]
                x_fit = torch.cat([xs[0], x_multi_val if fallback == "DLinear-M" else torch.from_numpy(arrays["validation"][0][:, :, None].astype(np.float32))])
                y_fit = torch.cat([y_train_t, y_torch_val])
                numeric_test[fallback] = deep_refit(
                    model_class, model_args, x_fit, y_fit, xs[3], deep_epochs[fallback], training, device,
                    task_dir/'models'/f'{fallback}_refit' if purged else None,
                )
            fallback_cal, fallback_dec = numeric_cal[fallback], numeric_dec[fallback]
            fallback_test = numeric_test.get(fallback)

            candidates, candidate_audit = {}, {}
            for variant in VARIANTS:
                builder, _, alpha, cal_prediction = fit_residual(variant, arrays["train"], calibration, solver=solver)
                decision_prediction = decision[0][:, -1, None] + Ridge(alpha=alpha, solver=solver).fit(
                    builder.transform(arrays["train"][0], arrays["train"][1], arrays["train"][2]),
                    arrays["train"][3] - arrays["train"][0][:, -1, None],
                ).predict(builder.transform(decision[0], decision[1], decision[2]))
                shuffled = permutation_scores(variant, builder, alpha, arrays["train"], calibration, decision, permutations, solver=solver)
                aligned_mse = metrics(decision[3], decision_prediction)["mse"]
                p_value = (1 + sum(value <= aligned_mse for value in shuffled)) / (permutations + 1)
                segment_wins = all(
                    mask.any() and metrics(decision[3][mask], decision_prediction[mask])["mse"] <
                    metrics(decision[3][mask], fallback_dec[mask])["mse"]
                    for mask in segment_masks
                )
                eligible = aligned_mse < metrics(decision[3], fallback_dec)["mse"] and segment_wins and p_value <= 0.025
                if not purged:
                    candidates[variant] = refit_residual(
                        variant, alpha,
                        tuple(np.concatenate([arrays["train"][i], arrays["validation"][i]]) for i in range(4)),
                        arrays["test"], solver=solver,
                    )
                candidate_audit[variant] = {
                    "alpha": alpha, "decision_mse": aligned_mse,
                    "permutation_mean_mse": float(np.mean(shuffled)), "permutation_p_value": p_value,
                    "segment_wins": segment_wins, "eligible": eligible,
                }
                if purged:
                    task_evidence.update({f'{variant}_calibration': cal_prediction,
                                          f'{variant}_decision': decision_prediction,
                                          f'{variant}_permutation_mse': np.asarray(shuffled)})

            eligible = [variant for variant in VARIANTS if candidate_audit[variant]["eligible"]]
            selected = min(eligible, key=lambda name: candidate_audit[name]["decision_mse"]) if eligible else "numeric_fallback"
            if purged:
                (task_dir/'frozen_selection.json').write_text(json.dumps({
                    'frozen_utc': datetime.now(timezone.utc).isoformat(), 'fallback': fallback,
                    'selected': selected, 'numeric_calibration_scores': numeric_cal_scores,
                    'candidates': candidate_audit, 'deep_epochs': deep_epochs,
                    'caveat': 'frozen before this run test evaluation; test period was seen in historical research',
                }, ensure_ascii=False, indent=2), encoding='utf-8')
                if fallback in deep_specs:
                    model_class, model_args, xs = deep_specs[fallback]
                    x_fit = torch.cat([xs[0], x_multi_val if fallback == 'DLinear-M' else
                                       torch.from_numpy(arrays['validation'][0][:, :, None].astype(np.float32))])
                    fallback_test = deep_refit(model_class, model_args, x_fit,
                        torch.cat([y_train_t, y_torch_val]), xs[3], deep_epochs[fallback], training,
                        device, task_dir/'models'/f'{fallback}_refit')
                for variant in VARIANTS:
                    candidates[variant] = refit_residual(variant, candidate_audit[variant]['alpha'],
                        tuple(np.concatenate([arrays['train'][i], arrays['validation'][i]]) for i in range(4)),
                        arrays['test'], solver=solver)
                    task_evidence[f'{variant}_test'] = candidates[variant]
            selected_test = fallback_test if selected == "numeric_fallback" else candidates[selected]
            block = max(horizon, min(config.seasonal_period, 24))
            for name, prediction in {fallback: fallback_test, **candidates, "SafeFAME-TS-v3" if purged else "SafeFAME-TS-v2": selected_test}.items():
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
            if purged:
                audit_rows[-1].update({'target_boundary': boundary,
                    'validation_crossing_windows_removed': int((~(cal_mask|dec_mask)).sum()),
                    'segment1_windows': int(segment_masks[0].sum()), 'segment2_windows': int(segment_masks[1].sum()),
                    'segment_status': 'sufficient' if all(mask.any() for mask in segment_masks) else 'insufficient_target_disjoint_subperiods'})
                for name in numeric_cal:
                    task_evidence[f'{name}_calibration'] = numeric_cal[name]
                    task_evidence[f'{name}_decision'] = numeric_dec[name]
                task_evidence.update({'fallback_test': fallback_test, 'selected_test': selected_test})
                np.savez_compressed(task_dir/'predictions.npz', **task_evidence)
                detail = {'audit': audit_rows[-1], 'numeric_calibration_scores': numeric_cal_scores,
                          'deep_epochs': deep_epochs, 'deep_meta': deep_meta,
                          'train_end': train_end, 'validation_end': validation_end, 'rows': n,
                          'features': features, 'target_mean': target_mean, 'target_std': target_std,
                          'block_length': block, 'bootstrap_seed': 2026, 'bootstrap_repeats': 5000}
                (task_dir/'audit.json').write_text(json.dumps(detail, ensure_ascii=False, indent=2), encoding='utf-8')
                pd.DataFrame(metric_rows).to_csv(output/f'{prefix}_metrics.csv', index=False)
                pd.DataFrame(audit_rows).to_csv(output/f'{prefix}_selection_audit.csv', index=False)
            print(f"{domain} H={horizon}: fallback={fallback}, selected={selected}", flush=True)

    metrics_frame, audit_frame = pd.DataFrame(metric_rows), pd.DataFrame(audit_rows)
    metrics_frame.to_csv(output / f"{prefix}_metrics.csv", index=False)
    audit_frame.to_csv(output / f"{prefix}_selection_audit.csv", index=False)
    (output / f"{prefix}_protocol.json").write_text(json.dumps({
        "split": {"train": 0.7, "validation": 0.1, "test": 0.2,
                  "validation_use": "target-disjoint raw-time halves with crossing windows removed" if purged else "origin-disjoint halves; H-step target values overlap across calibration/decision"},
        "revision": 'post-audit v3 correction; not a new untouched test set' if purged else 'historical v2',
        "solver": solver,
        "numeric_experts": ["Last", "SeasonalNaive", "AR-Ridge", "DLinear-M", "PatchTST"],
        "text_candidates": list(VARIANTS), "permutations": permutations,
        "permutation_p_threshold": 0.025,
        "permutation_role": "frozen eligibility gate; within-split joint row permutation of text and quality; alpha reselected on calibration for each permutation",
        "sensitivity_role": "999 circular shifts are post-review diagnostics only; frozen routing is unchanged",
        "bootstrap_role": "5000 moving-block repeats for test paired-loss intervals only; never model or route selection",
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--purged", action="store_true", help="Post-audit v3: target-disjoint validation and full evidence logs")
    parser.add_argument("--threads", type=int, default=1, help="Pinned BLAS/PyTorch threads for v3 only; historical v2 unchanged")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        self_check()
    else:
        output = args.output or Path('outputs/safefame_v3' if args.purged else 'outputs/safefame_v2')
        if args.purged and output.resolve() == Path('outputs/safefame_v2').resolve():
            raise ValueError('v3 must not overwrite historical v2')
        if args.purged:
            from threadpoolctl import threadpool_limits
            torch.set_num_threads(args.threads)
            with threadpool_limits(args.threads):
                run(args.root, args.semantic_root, output, torch.device(args.device), args.permutations, purged=True)
        else:
            run(args.root, args.semantic_root, output, torch.device(args.device), args.permutations)
