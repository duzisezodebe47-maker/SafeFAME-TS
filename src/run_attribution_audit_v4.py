"""Post-review attribution diagnostics for the ten text-enabled v4 routes.

This module does not change the frozen v4 routing decisions.  It separates
semantic content, text-quality metadata, numerical model capacity, and the
Last-anchor mismatch using the saved, untouched test segments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import run_safefame_v2 as base
from plot_style import configure_fonts, finish_fonts
from run_famets_selective import frequency_statistics
from run_reviewer_sensitivity import shifted_order
from run_safefame_v4 import arrays_for, interval, metrics


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "outputs" / "safefame_v4"
DEFAULT_OUTPUT = ROOT / "outputs" / "attribution_audit_v4"
MODES = ("numeric_only", "quality_only", "semantic_only", "full", "dimension_matched_numeric")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class AttributionBuilder:
    """Build one explicitly named feature set for a post-hoc residual head."""

    def __init__(self, variant: str, mode: str, components: int = 24, seed: int = 2026):
        if variant not in base.VARIANTS or mode not in MODES:
            raise ValueError((variant, mode))
        self.variant = variant
        self.mode = mode
        self.components = components
        self.seed = seed
        self.numeric_scaler = StandardScaler()
        self.frequency_scaler = StandardScaler()
        self.quality_scaler = StandardScaler()
        self.interaction_scaler = StandardScaler()
        self.rff_scaler = StandardScaler()
        self.pca: PCA | None = None
        self.rff_weights: np.ndarray | None = None
        self.rff_bias: np.ndarray | None = None
        self.output_dim: int | None = None

    def _numeric_base(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        numeric = self.numeric_scaler.transform(x)
        frequency = None
        if self.variant == "frequency_residual":
            frequency = self.frequency_scaler.transform(frequency_statistics(x))
        return numeric, frequency

    def fit(self, x: np.ndarray, embeddings: np.ndarray, quality: np.ndarray,
            target_dim: int | None = None) -> "AttributionBuilder":
        self.numeric_scaler.fit(x)
        if self.variant == "frequency_residual":
            self.frequency_scaler.fit(frequency_statistics(x))
        if self.mode in {"quality_only", "full"}:
            self.quality_scaler.fit(quality)
        if self.mode in {"semantic_only", "full"}:
            count = min(self.components, len(x) - 1, embeddings.shape[1])
            self.pca = PCA(n_components=count, svd_solver="full").fit(embeddings)
            if self.variant == "frequency_residual":
                frequency = self.frequency_scaler.transform(frequency_statistics(x))
                reduced = self.pca.transform(embeddings)
                interaction = (reduced[:, :, None] * frequency[:, None, :8]).reshape(len(x), -1)
                self.interaction_scaler.fit(interaction)
        if self.mode == "dimension_matched_numeric":
            if target_dim is None:
                raise ValueError("target_dim is required for the dimension-matched control")
            numeric, frequency = self._numeric_base(x)
            base_design = numeric if frequency is None else np.c_[numeric, frequency]
            extra = target_dim - base_design.shape[1]
            if extra < 0:
                raise ValueError((target_dim, base_design.shape[1]))
            rng = np.random.default_rng(self.seed)
            self.rff_weights = rng.normal(scale=1 / np.sqrt(base_design.shape[1]),
                                          size=(base_design.shape[1], extra))
            self.rff_bias = rng.uniform(0, 2 * np.pi, size=extra)
            if extra:
                self.rff_scaler.fit(np.sqrt(2.0 / extra) * np.cos(base_design @ self.rff_weights + self.rff_bias))
            self.output_dim = target_dim
        else:
            self.output_dim = self.transform(x, embeddings, quality).shape[1]
        return self

    def transform(self, x: np.ndarray, embeddings: np.ndarray, quality: np.ndarray) -> np.ndarray:
        numeric, frequency = self._numeric_base(x)
        chunks = [numeric]
        if self.mode in {"semantic_only", "full"}:
            if self.pca is None:
                raise RuntimeError("PCA is not fitted")
            reduced = self.pca.transform(embeddings)
            chunks.append(reduced)
        else:
            reduced = None
        if self.mode in {"quality_only", "full"}:
            chunks.append(self.quality_scaler.transform(quality))
        if frequency is not None:
            chunks.append(frequency)
            if reduced is not None:
                interaction = (reduced[:, :, None] * frequency[:, None, :8]).reshape(len(x), -1)
                chunks.append(self.interaction_scaler.transform(interaction))
        design = np.c_[tuple(chunks)]
        if self.mode == "dimension_matched_numeric":
            if self.rff_weights is None or self.rff_bias is None:
                raise RuntimeError("random Fourier control is not fitted")
            extra = self.rff_weights.shape[1]
            if extra:
                random_features = np.sqrt(2.0 / extra) * np.cos(design @ self.rff_weights + self.rff_bias)
                design = np.c_[design, self.rff_scaler.transform(random_features)]
        if self.output_dim is not None and design.shape[1] != self.output_dim:
            raise AssertionError((design.shape, self.output_dim))
        return design


def fit_head(builder: AttributionBuilder, train, train_anchor: np.ndarray,
             validation, validation_anchor: np.ndarray) -> tuple[float, np.ndarray]:
    x_train = builder.transform(*train[:3])
    x_validation = builder.transform(*validation[:3])
    residual = train[3] - train_anchor
    choices = []
    for alpha in base.RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, solver="cholesky").fit(x_train, residual)
        prediction = validation_anchor + model.predict(x_validation).reshape(len(x_validation), -1)
        choices.append((metrics(validation[3], prediction)["mse"], alpha, prediction))
    _, alpha, prediction = min(choices, key=lambda item: item[0])
    return float(alpha), prediction


def refit_head(variant: str, mode: str, components: int, seed: int, target_dim: int,
               alpha: float, fit, fit_anchor: np.ndarray, test, test_anchor: np.ndarray) -> tuple[np.ndarray, int]:
    builder = AttributionBuilder(variant, mode, components, seed).fit(
        *fit[:3], target_dim=target_dim if mode == "dimension_matched_numeric" else None
    )
    design_fit = builder.transform(*fit[:3])
    design_test = builder.transform(*test[:3])
    model = Ridge(alpha=alpha, solver="cholesky").fit(design_fit, fit[3] - fit_anchor)
    prediction = test_anchor + model.predict(design_test).reshape(len(test[0]), -1)
    return prediction, design_fit.shape[1]


def adjusted_pvalues(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Holm family-wise and Benjamini-Hochberg adjusted p-values."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    m = len(values)
    holm_sorted = np.maximum.accumulate((m - np.arange(m)) * ranked).clip(max=1.0)
    bh_sorted = np.minimum.accumulate((m / np.arange(m, 0, -1)) * ranked[::-1])[::-1].clip(max=1.0)
    holm = np.empty_like(values)
    bh = np.empty_like(values)
    holm[order] = holm_sorted
    bh[order] = bh_sorted
    return holm, bh


def dual_prediction(x: np.ndarray, y: np.ndarray, z: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge prediction with an unpenalized intercept, solved in sample space."""
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    centered_x, centered_y = x - x_mean, y - y_mean
    kernel = centered_x @ centered_x.T
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    dual = eigenvectors @ ((eigenvectors.T @ centered_y) / (eigenvalues[:, None] + alpha))
    return y_mean + (z - x_mean) @ centered_x.T @ dual


def frequency_designs(builder: base.ResidualBuilder, datasets: tuple, orders: tuple[np.ndarray, ...]):
    """Refit the pairing-dependent interaction scale for one frequency permutation."""
    prepared = []
    train_interaction = None
    for index, (data, order) in enumerate(zip(datasets, orders, strict=True)):
        x, embeddings, quality, _ = data
        numeric = builder.numeric_scaler.transform(x)
        reduced = builder.pca.transform(embeddings[order])
        quality_z = builder.quality_scaler.transform(quality[order])
        frequency = builder.frequency_scaler.transform(frequency_statistics(x))
        interaction = (reduced[:, :, None] * frequency[:, None, :8]).reshape(len(x), -1)
        if index == 0:
            train_interaction = StandardScaler().fit(interaction)
        prepared.append((numeric, reduced, quality_z, frequency, interaction))
    assert train_interaction is not None
    return [np.c_[numeric, reduced, quality_z, frequency, train_interaction.transform(interaction)]
            for numeric, reduced, quality_z, frequency, interaction in prepared]


def corrected_frequency_null(train, cal, dec, alpha: float, count: int, block: int,
                             row_seed: int, circular_seed: int) -> tuple[np.ndarray, np.ndarray]:
    builder = base.ResidualBuilder("frequency_residual").fit(*train[:3])
    residual = train[3] - train[0][:, -1, None]
    row_scores, circular_scores = [], []
    for k in range(count):
        local = np.random.default_rng(row_seed + k)
        row_orders = tuple(local.permutation(len(data[0])) for data in (train, cal, dec))
        xt, xc, xd = frequency_designs(builder, (train, cal, dec), row_orders)
        choices = []
        for candidate_alpha in base.RIDGE_ALPHAS:
            prediction = cal[0][:, -1, None] + dual_prediction(xt, residual, xc, candidate_alpha)
            choices.append((metrics(cal[3], prediction)["mse"], candidate_alpha))
        selected_alpha = min(choices, key=lambda item: item[0])[1]
        prediction = dec[0][:, -1, None] + dual_prediction(xt, residual, xd, selected_alpha)
        row_scores.append(metrics(dec[3], prediction)["mse"])

    rng = np.random.default_rng(circular_seed)
    for _ in range(count):
        orders = (shifted_order(len(train[0]), rng, block), np.arange(len(cal[0])),
                  shifted_order(len(dec[0]), rng, block))
        xt, _, xd = frequency_designs(builder, (train, cal, dec), orders)
        prediction = dec[0][:, -1, None] + dual_prediction(xt, residual, xd, alpha)
        circular_scores.append(metrics(dec[3], prediction)["mse"])
    return np.asarray(row_scores), np.asarray(circular_scores)


def self_check() -> None:
    rng = np.random.default_rng(71)
    x = rng.normal(size=(23, 11))
    y = rng.normal(size=(23, 3))
    z = rng.normal(size=(7, 11))
    for alpha in (0.01, 1.0, 100.0):
        expected = Ridge(alpha=alpha, solver="cholesky").fit(x, y).predict(z)
        np.testing.assert_allclose(dual_prediction(x, y, z, alpha), expected, rtol=1e-8, atol=1e-9)
    p = np.array([0.01, 0.02, 0.03])
    holm, bh = adjusted_pvalues(p)
    np.testing.assert_allclose(holm, [0.03, 0.04, 0.04])
    np.testing.assert_allclose(bh, [0.03, 0.03, 0.03])
    x = rng.normal(size=(40, 12)); e = rng.normal(size=(40, 32)); q = rng.normal(size=(40, 10))
    full = AttributionBuilder("frequency_residual", "full").fit(x, e, q)
    target_dim = full.transform(x, e, q).shape[1]
    matched = AttributionBuilder("frequency_residual", "dimension_matched_numeric").fit(
        x, e, q, target_dim=target_dim
    )
    assert matched.transform(x, e, q).shape[1] == target_dim


def make_figure(results: pd.DataFrame, output: Path) -> None:
    configure_fonts()
    labels = [f"{r.domain} H{r.horizon} F{r.fold}" for r in results.itertuples()]
    y = np.arange(len(results))
    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    ax.axvline(0, color="#666666", linewidth=1)
    ax.scatter(results["original_gain_vs_fallback_pct"], y - 0.18, color="#6B7280", marker="s",
               label="冻结Last锚点候选 对 实际回退")
    ax.scatter(results["fallback_head_gain_vs_fallback_pct"], y, color="#2878B5", marker="o",
               label="回退锚定完整特征 对 实际回退")
    ax.scatter(results["full_gain_vs_dimension_matched_numeric_pct"], y + 0.18, color="#D95F02", marker="^",
               label="完整特征 对 等维纯数值对照")
    ax.set_yticks(y, labels)
    ax.set_xlabel("测试MSE相对改善（%）")
    ax.set_title("实际启用路线的审查后归因诊断")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False, loc="best")
    finish_fonts(fig)
    fig.tight_layout()
    fig.savefig(output / "fig_selected_route_attribution.png", dpi=240)
    plt.close(fig)


def run(output: Path, permutations: int = 999) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    evidence_dir = output / "evidence"
    evidence_dir.mkdir(exist_ok=True)
    config = json.loads((ROOT / "configs" / "safefame_v4.json").read_text(encoding="utf-8"))
    tasks = pd.read_csv(V4 / "task_results.csv")
    candidates = pd.read_csv(V4 / "candidate_results.csv")
    selected = tasks[tasks.selected.ne("numeric_fallback")].copy()
    if len(selected) != 10:
        raise AssertionError(f"expected 10 selected routes, found {len(selected)}")

    all_p = candidates.p_value.to_numpy(float)
    holm, bh = adjusted_pvalues(all_p)
    multiplicity = {
        "candidate_paths": int(len(all_p)),
        "raw_p_le_0_025": int((all_p <= 0.025).sum()),
        "holm_p_le_0_05": int((holm <= 0.05).sum()),
        "bh_q_le_0_05": int((bh <= 0.05).sum()),
        "role": "post-hoc family-level description; not part of the frozen gate",
    }

    result_rows, comparison_rows, block_rows, corrected_rows = [], [], [], []
    for row_index, selected_row in enumerate(selected.itertuples(index=False)):
        domain, horizon, fold = selected_row.domain, int(selected_row.horizon), int(selected_row.fold)
        variant, fallback = selected_row.selected, selected_row.fallback
        settings = config["domains"][domain]
        n = len(pd.read_csv(ROOT / "data_processed" / "v4" / "numerical" / domain / f"{domain}.csv"))
        bounds = [int(n * p) for p in config["folds"][fold - 1]]
        arrays, _, _, _ = arrays_for(domain, settings, horizon, bounds)
        cal, dec, test = arrays["cal"], arrays["dec"], arrays["test"]
        fit = tuple(np.concatenate([cal[i], dec[i]]) for i in range(4))
        task_dir = V4 / "tasks" / f"{domain}_h{horizon}_f{fold}"
        saved = np.load(task_dir / "predictions.npz")
        fallback_cal, fallback_dec = saved[f"{fallback}_cal"], saved[f"{fallback}_dec"]
        fallback_test = saved["fallback_test"]
        fit_anchor = np.concatenate([fallback_cal, fallback_dec])
        original_candidate = saved[f"{variant}_test"]
        old_numeric_control = saved[f"{variant}_control_test"]
        candidate_row = candidates[(candidates.domain == domain) & (candidates.horizon == horizon) &
                                   (candidates.fold == fold) & (candidates.variant == variant)].iloc[0]

        component_count = min(24, len(cal[0]) - 1, cal[1].shape[1])
        seed = 7300 + 100 * row_index + horizon
        full_probe = AttributionBuilder(variant, "full", component_count, seed).fit(*cal[:3])
        target_dim = full_probe.transform(*cal[:3]).shape[1]
        predictions, dimensions, alphas = {}, {}, {}
        for mode in MODES:
            builder = AttributionBuilder(variant, mode, component_count, seed).fit(
                *cal[:3], target_dim=target_dim if mode == "dimension_matched_numeric" else None
            )
            alpha, _ = fit_head(builder, cal, fallback_cal, dec, fallback_dec)
            prediction, dimension = refit_head(variant, mode, component_count, seed, target_dim, alpha,
                                                fit, fit_anchor, test, fallback_test)
            predictions[mode], dimensions[mode], alphas[mode] = prediction, dimension, alpha

        block = int(candidate_row.block_length)
        base_mse = metrics(test[3], fallback_test)["mse"]
        full_mse = metrics(test[3], predictions["full"])["mse"]
        original_mse = metrics(test[3], original_candidate)["mse"]
        dim_mse = metrics(test[3], predictions["dimension_matched_numeric"])["mse"]
        fallback_seed = 12000 + row_index
        dimension_seed = 13000 + row_index
        full_vs_fallback = interval(test[3], fallback_test, predictions["full"], block,
                                    seed=fallback_seed, repeats=config["bootstrap_repeats"])
        full_vs_dim = interval(test[3], predictions["dimension_matched_numeric"], predictions["full"], block,
                               seed=dimension_seed, repeats=config["bootstrap_repeats"])
        result_rows.append({
            "domain": domain, "horizon": horizon, "fold": fold, "variant": variant, "fallback": fallback,
            "test_windows": len(test[0]), "block_length": block, "full_dimension": target_dim,
            "original_gain_vs_fallback_pct": 100 * (1 - original_mse / base_mse),
            "frozen_selected_ci_low_pct": 100 * float(selected_row.selected_ci_low) / base_mse,
            "frozen_selected_ci_high_pct": 100 * float(selected_row.selected_ci_high) / base_mse,
            "legacy_circular_p": float(candidate_row.circular_p_value),
            "legacy_numeric_ablation_ci_low_pct": 100 * float(candidate_row.matched_ci_low) /
                metrics(test[3], old_numeric_control)["mse"],
            "legacy_numeric_ablation_ci_high_pct": 100 * float(candidate_row.matched_ci_high) /
                metrics(test[3], old_numeric_control)["mse"],
            "fallback_head_gain_vs_fallback_pct": 100 * (1 - full_mse / base_mse),
            "fallback_head_ci_low_pct": 100 * full_vs_fallback[1] / base_mse,
            "fallback_head_ci_high_pct": 100 * full_vs_fallback[2] / base_mse,
            "full_gain_vs_dimension_matched_numeric_pct": 100 * (1 - full_mse / dim_mse),
            "full_vs_dimension_ci_low_pct": 100 * full_vs_dim[1] / dim_mse,
            "full_vs_dimension_ci_high_pct": 100 * full_vs_dim[2] / dim_mse,
            "fallback_bootstrap_seed": fallback_seed,
            "dimension_bootstrap_seed": dimension_seed,
            "legacy_numeric_ablation_gain_pct": 100 * (1 - original_mse / metrics(test[3], old_numeric_control)["mse"]),
            **{f"{mode}_dimension": dimensions[mode] for mode in MODES},
            **{f"{mode}_alpha": alphas[mode] for mode in MODES},
        })

        comparisons = [
            ("semantic_only_vs_numeric", "numeric_only", "semantic_only"),
            ("quality_only_vs_numeric", "numeric_only", "quality_only"),
            ("full_vs_quality_only", "quality_only", "full"),
            ("full_vs_semantic_only", "semantic_only", "full"),
            ("full_vs_dimension_matched_numeric", "dimension_matched_numeric", "full"),
        ]
        for comparison_index, (label, control_name, candidate_name) in enumerate(comparisons):
            control_prediction, candidate_prediction = predictions[control_name], predictions[candidate_name]
            control_mse = metrics(test[3], control_prediction)["mse"]
            candidate_mse = metrics(test[3], candidate_prediction)["mse"]
            comparison_seed = 14000 + 100 * row_index + comparison_index
            delta, low, high = interval(test[3], control_prediction, candidate_prediction, block,
                                        seed=comparison_seed,
                                        repeats=config["bootstrap_repeats"])
            comparison_rows.append({
                "domain": domain, "horizon": horizon, "fold": fold, "variant": variant,
                "comparison": label, "control": control_name, "candidate": candidate_name,
                "control_mse": control_mse, "candidate_mse": candidate_mse,
                "gain_pct": 100 * (1 - candidate_mse / control_mse),
                "loss_delta": delta, "ci_low": low, "ci_high": high,
                "ci_low_pct": 100 * low / control_mse, "ci_high_pct": 100 * high / control_mse,
                "bootstrap_seed": comparison_seed,
            })

        for candidate_block in sorted({max(2, horizon), block, min(len(test[0]), 2 * block)}):
            block_seed = 15000 + 100 * row_index + candidate_block
            delta, low, high = interval(test[3], fallback_test, original_candidate, candidate_block,
                                        seed=block_seed,
                                        repeats=config["bootstrap_repeats"])
            block_rows.append({
                "domain": domain, "horizon": horizon, "fold": fold, "variant": variant,
                "block_length": candidate_block, "is_primary": candidate_block == block,
                "gain_pct": 100 * (1 - original_mse / base_mse),
                "ci_low_pct": 100 * low / base_mse, "ci_high_pct": 100 * high / base_mse,
                "loss_delta": delta,
                "bootstrap_seed": block_seed,
            })

        evidence = {"actual": test[3], "fallback": fallback_test, "original_candidate": original_candidate,
                    "legacy_numeric_ablation": old_numeric_control,
                    **{name: value for name, value in predictions.items()}}
        np.savez_compressed(evidence_dir / f"{domain}_h{horizon}_f{fold}_{variant}.npz", **evidence)

        if variant == "frequency_residual":
            frozen = json.loads((task_dir / "frozen_selection.json").read_text(encoding="utf-8"))
            aligned_score = float(frozen["candidates"][variant]["decision_mse"])
            alpha = float(frozen["candidates"][variant]["alpha"])
            corrected_row, corrected_circular = corrected_frequency_null(
                arrays["train"], cal, dec, alpha, permutations, block, 1001,
                4100 + 100 * (fold - 1) + 10 * horizon + list(base.VARIANTS).index(variant),
            )
            row_p = (1 + int((corrected_row <= aligned_score).sum())) / (permutations + 1)
            circular_p = (1 + int((corrected_circular <= aligned_score).sum())) / (permutations + 1)
            corrected_rows.append({
                "domain": domain, "horizon": horizon, "fold": fold, "variant": variant,
                "permutations": permutations, "aligned_decision_mse": aligned_score,
                "legacy_row_p": float(candidate_row.p_value), "corrected_row_p": row_p,
                "legacy_circular_p": float(candidate_row.circular_p_value),
                "corrected_circular_p": circular_p,
                "role": "post-hoc sensitivity; frozen routing unchanged",
            })
            np.savez_compressed(evidence_dir / f"{domain}_h{horizon}_f{fold}_corrected_frequency_nulls.npz",
                                row_null=corrected_row, circular_null=corrected_circular,
                                aligned_decision_mse=aligned_score)

    results = pd.DataFrame(result_rows)
    comparisons = pd.DataFrame(comparison_rows)
    blocks = pd.DataFrame(block_rows)
    corrected = pd.DataFrame(corrected_rows)
    corrected_lookup = {(row.domain, int(row.horizon), int(row.fold)): float(row.corrected_circular_p)
                        for row in corrected.itertuples(index=False)}
    results["diagnostic_circular_p"] = [
        corrected_lookup.get((row.domain, int(row.horizon), int(row.fold)), float(row.legacy_circular_p))
        for row in results.itertuples(index=False)
    ]
    results["joint_posthoc_support"] = (
        (results.original_gain_vs_fallback_pct > 0) &
        (results.frozen_selected_ci_low_pct > 0) &
        (results.diagnostic_circular_p <= 0.025) &
        (results.fallback_head_ci_low_pct > 0) &
        (results.full_vs_dimension_ci_low_pct > 0)
    )
    results.to_csv(output / "selected_route_attribution.csv", index=False)
    comparisons.to_csv(output / "selected_route_comparisons.csv", index=False)
    blocks.to_csv(output / "block_length_sensitivity.csv", index=False)
    corrected.to_csv(output / "corrected_frequency_permutation.csv", index=False)
    make_figure(results, output)

    family_rows = []
    for (domain, horizon), group in tasks.groupby(["domain", "horizon"]):
        enabled = group[group.selected.ne("numeric_fallback")]
        counts = enabled.selected.value_counts()
        variant = counts.index[0] if len(counts) else "none"
        same_variant_count = int(counts.iloc[0]) if len(counts) else 0
        positive_count = int(((enabled.selected == variant) & (enabled.selected_gain_pct > 0)).sum())
        family_rows.append({"domain": domain, "horizon": int(horizon), "variant": variant,
                            "same_variant_selected_folds": same_variant_count,
                            "positive_selected_folds": positive_count,
                            "exploratory_replicated": same_variant_count >= 2 and positive_count >= 2})
    families = pd.DataFrame(family_rows)
    families.to_csv(output / "cross_fold_replication.csv", index=False)

    gains = selected.selected_gain_pct.to_numpy(float)
    comparison_summary = {}
    for name, group in comparisons.groupby("comparison"):
        comparison_summary[name] = {
            "routes": int(len(group)),
            "point_positive": int((group.gain_pct > 0).sum()),
            "ci_positive": int((group.ci_low > 0).sum()),
            "ci_negative": int((group.ci_high < 0).sum()),
            "median_gain_pct": float(group.gain_pct.median()),
        }
    summary = {
        "analysis_role": "post-review attribution diagnostics; no change to frozen v4 routing",
        "selected_routes": int(len(selected)),
        "selected_semantic_routes": int((selected.selected == "semantic_residual").sum()),
        "selected_frequency_routes": int((selected.selected == "frequency_residual").sum()),
        "original_selected_point_positive": int((gains > 0).sum()),
        "original_selected_point_negative": int((gains < 0).sum()),
        "original_selected_ci_positive": int((results.frozen_selected_ci_low_pct > 0).sum()),
        "original_selected_ci_negative": int((results.frozen_selected_ci_high_pct < 0).sum()),
        "selected_circular_pass_after_frequency_correction": int((results.diagnostic_circular_p <= 0.025).sum()),
        "legacy_numeric_ablation_selected_ci_positive": int((results.legacy_numeric_ablation_ci_low_pct > 0).sum()),
        "legacy_numeric_ablation_selected_ci_negative": int((results.legacy_numeric_ablation_ci_high_pct < 0).sum()),
        "joint_posthoc_supported_routes": int(results.joint_posthoc_support.sum()),
        "joint_posthoc_supported_list": [
            f"{row.domain} H{row.horizon} F{row.fold} {row.variant}"
            for row in results.itertuples(index=False) if row.joint_posthoc_support
        ],
        "original_selected_mean_gain_pct": float(gains.mean()),
        "original_selected_median_gain_pct": float(np.median(gains)),
        "all_120_mean_gain_pct": float(tasks.selected_gain_pct.mean()),
        "all_120_median_gain_pct": float(tasks.selected_gain_pct.median()),
        "all_120_mean_without_largest_gain_pct": float(
            tasks.drop(index=tasks.selected_gain_pct.idxmax()).selected_gain_pct.mean()
        ),
        "largest_route_share_of_total_net_gain_pct": float(
            100 * tasks.selected_gain_pct.max() / tasks.selected_gain_pct.sum()
        ),
        "largest_route_share_of_positive_gain_pct": float(100 * gains.max() / gains[gains > 0].sum()),
        "fallback_anchored_full_point_positive": int((results.fallback_head_gain_vs_fallback_pct > 0).sum()),
        "fallback_anchored_full_ci_positive": int((results.fallback_head_ci_low_pct > 0).sum()),
        "fallback_anchored_full_ci_negative": int((results.fallback_head_ci_high_pct < 0).sum()),
        "full_vs_dimension_matched_point_positive": int((results.full_gain_vs_dimension_matched_numeric_pct > 0).sum()),
        "full_vs_dimension_matched_ci_positive": int((results.full_vs_dimension_ci_low_pct > 0).sum()),
        "full_vs_dimension_matched_ci_negative": int((results.full_vs_dimension_ci_high_pct < 0).sum()),
        "exploratory_cross_fold_replicated_families": int(families.exploratory_replicated.sum()),
        "exploratory_cross_fold_replicated_list": [f"{r.domain} H{r.horizon} {r.variant}" for r in families.itertuples() if r.exploratory_replicated],
        "multiplicity": multiplicity,
        "comparisons": comparison_summary,
        "corrected_frequency_permutation": corrected.to_dict(orient="records"),
        "bootstrap_repeats": int(config["bootstrap_repeats"]),
        "random_fourier_control": "fixed task-specific seed; same design dimension as the full candidate; pure numerical inputs",
        "limitations": [
            "All analyses are post-hoc and use the already inspected historical v4 test blocks.",
            "The fallback-anchored head trains on calibration and decision residuals because train-segment predictions from every frozen fallback were not saved.",
            "Equal input dimension does not imply identical effective rank or inductive bias.",
            "The frequency permutation correction is limited to the one frequency route actually enabled by frozen v4.",
        ],
        "sha256": {
            "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "v4_summary": hashlib.sha256((V4 / "summary.json").read_bytes()).hexdigest(),
            "v4_tasks": hashlib.sha256((V4 / "task_results.csv").read_bytes()).hexdigest(),
            "v4_candidates": hashlib.sha256((V4 / "candidate_results.csv").read_bytes()).hexdigest(),
        },
    }
    write_json(output / "summary.json", summary)
    write_json(output / "protocol.json", {
        "scope": "ten routes enabled by the frozen v4 gate",
        "route_effect": "original frozen Last-anchored candidate versus frozen numerical fallback",
        "legacy_control_name": "pure numerical residual ablation; not capacity matched",
        "factorial_controls": list(MODES),
        "fallback_anchored_training": "fit on calibration, select Ridge alpha on decision, refit on calibration+decision, evaluate once on test",
        "dimension_control": "standardized numerical history/frequency plus fixed random Fourier features to exactly match the full design width",
        "permutation_correction": "refit frequency interaction scaler within every selected-route row or circular permutation",
        "multiplicity": "Holm FWER and BH FDR across all 240 saved main-gate candidate p-values; descriptive only",
        "bootstrap": f"{config['bootstrap_repeats']} moving-block resamples; diagnostic only",
        "routing_effect": "none",
    })
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if args.self_check:
        print("PASS: attribution audit numerical self-check")
        return
    summary = run(args.output, args.permutations)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
