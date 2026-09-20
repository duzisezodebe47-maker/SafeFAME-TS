"""Leakage-safe selective FAME-TS with frequency-semantic interactions."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from data_utils import load_time_ordered_frame
from run_baselines import ALPHAS, CONFIGS, CONFIRMATION_CONFIGS, make_windows, metrics


VARIANTS = ("numeric", "no_frequency", "full")
PERMUTATION_SEEDS = (2026, 2027, 2028)
ALL_CONFIGS = {**CONFIGS, **CONFIRMATION_CONFIGS}


def semantic_rows(cache: np.lib.npyio.NpzFile, origins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = {int(origin): index for index, origin in enumerate(cache["origin_index"].ravel())}
    indices = [positions[int(origin)] for origin in origins]
    embeddings = np.c_[cache["report_embedding"][indices], cache["search_embedding"][indices]]
    return embeddings.astype(np.float32), cache["quality"][indices].astype(np.float32)


def frequency_statistics(x: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(x, axis=1))
    power = spectrum**2
    total = power.sum(1, keepdims=True).clip(1e-8)
    normalized = power / total
    bins = normalized.shape[1]
    edges = np.linspace(0, bins, 5, dtype=int)
    bands = np.column_stack([normalized[:, edges[i] : edges[i + 1]].sum(1) for i in range(4)])
    frequencies = np.linspace(0.0, 1.0, bins)
    centroid = (normalized * frequencies).sum(1)
    entropy = -(normalized * np.log(normalized.clip(1e-12))).sum(1) / np.log(max(2, bins))
    dominant = np.argmax(normalized[:, 1:], axis=1) + 1 if bins > 1 else np.zeros(len(x))
    dominant = dominant / max(1, bins - 1)
    concentration = normalized.max(1)
    slope = x[:, -1] - x[:, 0]
    volatility = np.diff(x, axis=1).std(1)
    return np.c_[bands, centroid, entropy, dominant, concentration, slope, volatility].astype(np.float32)


class FeatureBuilder:
    def __init__(self, variant: str, components: int = 16):
        self.variant = variant
        self.components = components
        self.quality_scaler = StandardScaler()
        self.frequency_scaler = StandardScaler()
        self.interaction_scaler = StandardScaler()
        self.pca: PCA | None = None

    def fit(self, x: np.ndarray, embeddings: np.ndarray, quality: np.ndarray) -> "FeatureBuilder":
        self.quality_scaler.fit(quality)
        if self.variant == "full":
            frequency = frequency_statistics(x)
            self.frequency_scaler.fit(frequency)
            count = min(self.components, len(x) - 1, embeddings.shape[1])
            self.pca = PCA(n_components=count, svd_solver="full").fit(embeddings)
            reduced = self.pca.transform(embeddings)
            frequency_z = self.frequency_scaler.transform(frequency)
            interaction = (reduced[:, :, None] * frequency_z[:, None, :8]).reshape(len(x), -1)
            self.interaction_scaler.fit(interaction)
        return self

    def transform(self, x: np.ndarray, embeddings: np.ndarray, quality: np.ndarray) -> np.ndarray:
        if self.variant == "numeric":
            return x
        quality_z = self.quality_scaler.transform(quality)
        base = np.c_[x, embeddings, quality_z]
        if self.variant == "no_frequency":
            return base
        if self.pca is None:
            raise RuntimeError("FeatureBuilder must be fitted before transform")
        frequency_z = self.frequency_scaler.transform(frequency_statistics(x))
        reduced = self.pca.transform(embeddings)
        interaction = (reduced[:, :, None] * frequency_z[:, None, :8]).reshape(len(x), -1)
        return np.c_[base, frequency_z, self.interaction_scaler.transform(interaction)]


def fit_candidate(
    variant: str,
    train: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[FeatureBuilder, Ridge, float, np.ndarray]:
    x_train, embedding_train, quality_train, y_train = train
    x_val, embedding_val, quality_val, _ = validation
    builder = FeatureBuilder(variant).fit(x_train, embedding_train, quality_train)
    train_features = builder.transform(x_train, embedding_train, quality_train)
    val_features = builder.transform(x_val, embedding_val, quality_val)
    scores: dict[float, float] = {}
    models: dict[float, Ridge] = {}
    for alpha in (*ALPHAS, 1000.0, 10000.0):
        model = Ridge(alpha=alpha, solver="lsqr").fit(train_features, y_train)
        models[alpha] = model
        scores[alpha] = metrics(validation[3], model.predict(val_features))["mse"]
    best_alpha = min(scores, key=scores.get)
    return builder, models[best_alpha], best_alpha, models[best_alpha].predict(val_features)


def refit_predict(
    variant: str,
    alpha: float,
    train_validation: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    x_fit, embedding_fit, quality_fit, y_fit = train_validation
    builder = FeatureBuilder(variant).fit(x_fit, embedding_fit, quality_fit)
    model = Ridge(alpha=alpha, solver="lsqr").fit(
        builder.transform(x_fit, embedding_fit, quality_fit), y_fit
    )
    return model.predict(builder.transform(test[0], test[1], test[2]))


def permute_semantics(
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.random.default_rng(seed).permutation(len(data[0]))
    return data[0], data[1][order], data[2][order], data[3]


def bootstrap_interval(actual: np.ndarray, prediction: np.ndarray, seed: int = 2026) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    per_origin = np.mean((prediction - actual) ** 2, axis=1)
    samples = rng.choice(per_origin, size=(1000, len(per_origin)), replace=True).mean(1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


def apply_disagreement_gate(
    numeric_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    threshold: float,
    mixing_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    accepted = np.abs(candidate_prediction - numeric_prediction) <= threshold
    gate = accepted.astype(float) * mixing_weight
    return numeric_prediction + gate * (candidate_prediction - numeric_prediction), gate


def calibrate_disagreement_gate(
    numeric_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    actual: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    disagreement = np.abs(candidate_prediction - numeric_prediction)
    thresholds = np.unique(np.r_[0.0, np.quantile(disagreement, [0.5, 0.7, 0.8, 0.9, 0.95, 1.0])])
    choices: list[tuple[float, float, float, np.ndarray]] = []
    for threshold in thresholds:
        for mixing_weight in (0.25, 0.5, 0.75, 1.0):
            prediction, _ = apply_disagreement_gate(
                numeric_prediction, candidate_prediction, float(threshold), mixing_weight
            )
            choices.append(
                (metrics(actual, prediction)["mse"], float(threshold), mixing_weight, prediction)
            )
    _, threshold, mixing_weight, prediction = min(choices, key=lambda item: item[0])
    return threshold, mixing_weight, prediction


def select_variant(
    aligned_validation: dict[str, float],
    shuffled_validation: dict[str, list[float]],
    segment_validation: dict[str, tuple[float, float]],
) -> tuple[str, str]:
    numeric = aligned_validation["numeric"]
    numeric_segments = segment_validation["numeric"]
    eligible = ["numeric"]
    for variant in ("no_frequency", "full"):
        placebo = float(np.mean(shuffled_validation[variant]))
        stable = all(
            candidate < baseline
            for candidate, baseline in zip(segment_validation[variant], numeric_segments, strict=True)
        )
        if aligned_validation[variant] < numeric and aligned_validation[variant] < placebo and stable:
            eligible.append(variant)
    selected = min(eligible, key=aligned_validation.get)
    reason = "numeric_fallback" if selected == "numeric" else "aligned_beats_numeric_and_permutation"
    return selected, reason


def run(root: Path, semantic_root: Path, output: Path, domains: list[str]) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for domain in domains:
        config = ALL_CONFIGS[domain]
        frame, order_audit = load_time_ordered_frame(root / "numerical" / domain / f"{domain}.csv")
        raw = pd.to_numeric(frame["OT"], errors="coerce").to_numpy(float)
        n = len(raw)
        train_end, validation_end = int(n * 0.7), int(n * 0.8)
        mean, std = raw[:train_end].mean(), raw[:train_end].std()
        values = (raw - mean) / std
        cache = np.load(semantic_root / f"{domain}.npz")
        for horizon in config.horizons:
            split_data = {}
            split_origins = {}
            ranges = {
                "train": range(config.input_len, train_end - horizon + 1),
                "validation": range(train_end, validation_end - horizon + 1),
                "test": range(validation_end, n - horizon + 1),
            }
            for split, origins_range in ranges.items():
                x, y, origins = make_windows(values, config.input_len, horizon, origins_range)
                embeddings, quality = semantic_rows(cache, origins)
                split_data[split] = (x, embeddings, quality, y)
                split_origins[split] = origins

            fitted: dict[str, tuple[float, np.ndarray]] = {}
            raw_validation: dict[str, float] = {}
            aligned_validation: dict[str, float] = {}
            segment_validation: dict[str, tuple[float, float]] = {}
            calibrations: dict[str, tuple[float, float]] = {"numeric": (0.0, 0.0)}
            shuffled_validation = {"no_frequency": [], "full": []}
            for variant in VARIANTS:
                _, _, alpha, prediction = fit_candidate(
                    variant, split_data["train"], split_data["validation"]
                )
                fitted[variant] = (alpha, prediction)
                raw_validation[variant] = metrics(split_data["validation"][3], prediction)["mse"]
                if variant != "numeric":
                    for permutation_seed in PERMUTATION_SEEDS:
                        shuffled_train = permute_semantics(split_data["train"], permutation_seed)
                        shuffled_val = permute_semantics(split_data["validation"], permutation_seed + 100)
                        _, _, _, shuffled_prediction = fit_candidate(variant, shuffled_train, shuffled_val)
                        _, _, shuffled_gated = calibrate_disagreement_gate(
                            fitted["numeric"][1], shuffled_prediction, split_data["validation"][3]
                        )
                        shuffled_validation[variant].append(
                            metrics(split_data["validation"][3], shuffled_gated)["mse"]
                        )

            numeric_validation_prediction = fitted["numeric"][1]
            calibrated_validation_predictions = {"numeric": numeric_validation_prediction}
            for variant in ("no_frequency", "full"):
                threshold, mixing_weight, prediction = calibrate_disagreement_gate(
                    numeric_validation_prediction,
                    fitted[variant][1],
                    split_data["validation"][3],
                )
                calibrations[variant] = (threshold, mixing_weight)
                calibrated_validation_predictions[variant] = prediction
            midpoint = len(numeric_validation_prediction) // 2
            for variant, prediction in calibrated_validation_predictions.items():
                aligned_validation[variant] = metrics(split_data["validation"][3], prediction)["mse"]
                segment_validation[variant] = (
                    metrics(split_data["validation"][3][:midpoint], prediction[:midpoint])["mse"],
                    metrics(split_data["validation"][3][midpoint:], prediction[midpoint:])["mse"],
                )

            selected, reason = select_variant(
                aligned_validation, shuffled_validation, segment_validation
            )
            combined = tuple(
                np.concatenate([split_data["train"][i], split_data["validation"][i]])
                for i in range(4)
            )
            candidate_test_predictions: dict[str, np.ndarray] = {}
            for variant in VARIANTS:
                prediction = refit_predict(variant, fitted[variant][0], combined, split_data["test"])
                candidate_test_predictions[variant] = prediction
                score = metrics(split_data["test"][3], prediction)
                low, high = bootstrap_interval(split_data["test"][3], prediction)
                metric_rows.append(
                    {
                        "domain": domain,
                        "horizon": horizon,
                        "model": f"FAME-TS-{variant}",
                        **score,
                        "mse_ci_low": low,
                        "mse_ci_high": high,
                        "validation_mse": raw_validation[variant],
                        "selected_alpha": fitted[variant][0],
                        "test_windows": len(split_data["test"][0]),
                    }
                )

            selected_prediction = candidate_test_predictions[selected]
            selected_gate = np.zeros_like(selected_prediction)
            if selected != "numeric":
                selected_prediction, selected_gate = apply_disagreement_gate(
                    candidate_test_predictions["numeric"],
                    candidate_test_predictions[selected],
                    *calibrations[selected],
                )
            score = metrics(split_data["test"][3], selected_prediction)
            low, high = bootstrap_interval(split_data["test"][3], selected_prediction)
            metric_rows.append(
                {
                    "domain": domain,
                    "horizon": horizon,
                    "model": "FAME-TS-Selective",
                    **score,
                    "mse_ci_low": low,
                    "mse_ci_high": high,
                    "validation_mse": aligned_validation[selected],
                    "selected_alpha": fitted[selected][0],
                    "test_windows": len(split_data["test"][0]),
                }
            )
            selection_rows.append(
                {
                    "domain": domain,
                    "horizon": horizon,
                    "selected_variant": selected,
                    "selection_reason": reason,
                    "selected_disagreement_threshold": calibrations[selected][0],
                    "selected_mixing_weight": calibrations[selected][1],
                    "selected_gate_fraction": float(np.mean(selected_gate > 0)),
                    **{f"validation_raw_{key}": value for key, value in raw_validation.items()},
                    **{f"validation_{key}": value for key, value in aligned_validation.items()},
                    **{
                        f"validation_{key}_{segment}": values[index]
                        for key, values in segment_validation.items()
                        for index, segment in enumerate(("early", "late"))
                    },
                    "validation_shuffled_no_frequency_mean": np.mean(shuffled_validation["no_frequency"]),
                    "validation_shuffled_full_mean": np.mean(shuffled_validation["full"]),
                    "original_rows_reordered": order_audit["rows_moved_by_stable_sort"],
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "domain": domain,
                        "horizon": horizon,
                        "selected_variant": selected,
                        "origin_index": np.repeat(split_origins["test"], horizon),
                        "step": np.tile(np.arange(1, horizon + 1), len(split_origins["test"])),
                        "actual_z": split_data["test"][3].ravel(),
                        "prediction_z": selected_prediction.ravel(),
                        "semantic_gate": selected_gate.ravel(),
                    }
                )
            )
            print(
                f"{domain} H={horizon}: selected={selected}, val={aligned_validation[selected]:.6f}, "
                f"test={score['mse']:.6f}",
                flush=True,
            )

    result = pd.DataFrame(metric_rows)
    result.to_csv(output / "famets_selective_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "famets_selection_audit.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        output / "famets_selective_predictions.csv", index=False
    )
    (output / "famets_selective_config.json").write_text(
        json.dumps(
            {
                "split": {"train": 0.7, "validation": 0.1, "test": 0.2},
                "selection": "validation-calibrated disagreement gate; aligned path must beat numeric overall and in both chronological halves, then beat mean of three within-split text permutations",
                "protocol_phase": "development; the stability rule was added after an exploratory Time-MMD test exposed regime-shift failure",
                "alpha_candidates": [*ALPHAS, 1000.0, 10000.0],
                "frequency_features": "four power bands, centroid, entropy, dominant bin, concentration, slope, volatility",
                "semantic_features": "strict point-in-time MiniLM report/search embeddings plus ten quality features",
                "interaction": "16-dimensional train-only PCA semantic projection times first eight frequency statistics",
                "bootstrap": "1000 origin-level resamples, seed 2026",
                "domains": {name: asdict(ALL_CONFIGS[name]) for name in domains},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def self_check() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 24))
    embeddings = rng.normal(size=(40, 32))
    quality = rng.normal(size=(40, 10))
    y = rng.normal(size=(40, 3))
    builder = FeatureBuilder("full", components=8).fit(x[:25], embeddings[:25], quality[:25])
    features = builder.transform(x[25:], embeddings[25:], quality[25:])
    assert features.shape[0] == 15 and np.isfinite(features).all()
    train = (x[:25], embeddings[:25], quality[:25], y[:25])
    validation = (x[25:33], embeddings[25:33], quality[25:33], y[25:33])
    _, _, alpha, prediction = fit_candidate("full", train, validation)
    assert prediction.shape == (8, 3) and alpha > 0
    assert select_variant(
        {"numeric": 1.0, "no_frequency": 0.9, "full": 0.8},
        {"no_frequency": [1.1], "full": [1.2]},
        {"numeric": (1.0, 1.0), "no_frequency": (0.9, 0.9), "full": (0.8, 0.8)},
    )[0] == "full"
    numeric = np.array([[0.0, 0.0], [0.0, 0.0]])
    candidate = np.array([[0.1, 5.0], [0.2, 6.0]])
    gated, gate = apply_disagreement_gate(numeric, candidate, 0.5, 1.0)
    assert gate[:, 0].all() and not gate[:, 1].any() and np.allclose(gated[:, 1], 0.0)
    print("self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("references/external/Time-MMD"))
    parser.add_argument("--semantic-root", type=Path, default=Path("data_processed/semantic_features"))
    parser.add_argument("--output", type=Path, default=Path("outputs/famets_selective"))
    parser.add_argument("--domains", nargs="+", choices=tuple(ALL_CONFIGS), default=list(CONFIGS))
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        self_check()
    else:
        print(run(args.root, args.semantic_root, args.output, args.domains).to_string(index=False))
