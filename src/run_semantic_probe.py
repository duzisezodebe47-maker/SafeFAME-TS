"""Linear probe for cached MiniLM sample semantics under the frozen protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from data_utils import load_time_ordered_frame
from run_baselines import CONFIGS, make_windows, metrics


ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)


def semantic_for_origins(cache: np.lib.npyio.NpzFile, origins: np.ndarray) -> np.ndarray:
    positions = {int(origin): index for index, origin in enumerate(cache["origin_index"])}
    indices = [positions[int(origin)] for origin in origins]
    return np.concatenate(
        [cache["report_embedding"][indices], cache["search_embedding"][indices], cache["quality"][indices]],
        axis=1,
    ).astype(np.float32)


def fit_probe(
    numeric_train: np.ndarray,
    semantic_train: np.ndarray,
    y_train: np.ndarray,
    numeric_val: np.ndarray,
    semantic_val: np.ndarray,
    y_val: np.ndarray,
    numeric_test: np.ndarray,
    semantic_test: np.ndarray,
    model_kind: str,
) -> tuple[np.ndarray, float]:
    quality_start = semantic_train.shape[1] - 10
    scaler = StandardScaler().fit(semantic_train[:, quality_start:])
    semantic_train_z = np.c_[
        semantic_train[:, :quality_start], scaler.transform(semantic_train[:, quality_start:])
    ]
    semantic_val_z = np.c_[
        semantic_val[:, :quality_start], scaler.transform(semantic_val[:, quality_start:])
    ]
    x_train = semantic_train_z if model_kind == "TextOnly" else np.c_[numeric_train, semantic_train_z]
    x_val = semantic_val_z if model_kind == "TextOnly" else np.c_[numeric_val, semantic_val_z]
    scores: dict[float, float] = {}
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, solver="lsqr").fit(x_train, y_train)
        scores[alpha] = metrics(y_val, model.predict(x_val))["mse"]
    best_alpha = min(scores, key=scores.get)

    semantic_train_val = np.concatenate([semantic_train, semantic_val])
    final_scaler = StandardScaler().fit(semantic_train_val[:, quality_start:])
    semantic_train_val_z = np.c_[
        semantic_train_val[:, :quality_start],
        final_scaler.transform(semantic_train_val[:, quality_start:]),
    ]
    semantic_test_z = np.c_[
        semantic_test[:, :quality_start], final_scaler.transform(semantic_test[:, quality_start:])
    ]
    numeric_train_val = np.concatenate([numeric_train, numeric_val])
    x_train_val = (
        semantic_train_val_z
        if model_kind == "TextOnly"
        else np.c_[numeric_train_val, semantic_train_val_z]
    )
    x_test = semantic_test_z if model_kind == "TextOnly" else np.c_[numeric_test, semantic_test_z]
    model = Ridge(alpha=best_alpha, solver="lsqr").fit(x_train_val, np.concatenate([y_train, y_val]))
    return model.predict(x_test), best_alpha


def run(root: Path, semantic_root: Path, output: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for domain, config in CONFIGS.items():
        frame, _ = load_time_ordered_frame(root / "numerical" / domain / f"{domain}.csv")
        raw = pd.to_numeric(frame["OT"], errors="coerce").to_numpy(float)
        n = len(raw)
        train_end, val_end = int(n * 0.7), int(n * 0.8)
        values = (raw - raw[:train_end].mean()) / raw[:train_end].std()
        cache = np.load(semantic_root / f"{domain}.npz")
        for horizon in config.horizons:
            x_train, y_train, train_origins = make_windows(
                values, config.input_len, horizon, range(config.input_len, train_end - horizon + 1)
            )
            x_val, y_val, val_origins = make_windows(
                values, config.input_len, horizon, range(train_end, val_end - horizon + 1)
            )
            x_test, y_test, test_origins = make_windows(
                values, config.input_len, horizon, range(val_end, n - horizon + 1)
            )
            semantics = (
                semantic_for_origins(cache, train_origins),
                semantic_for_origins(cache, val_origins),
                semantic_for_origins(cache, test_origins),
            )
            for alignment in ("aligned", "shuffled"):
                current = semantics
                if alignment == "shuffled":
                    current = tuple(
                        array[np.random.default_rng(202601 + index).permutation(len(array))]
                        for index, array in enumerate(semantics)
                    )
                for model_kind in ("TextOnly", "EarlyFusion"):
                    forecast, alpha = fit_probe(
                        x_train, current[0], y_train,
                        x_val, current[1], y_val,
                        x_test, current[2], model_kind,
                    )
                    result = metrics(y_test, forecast)
                    result.update(
                        {
                            "domain": domain,
                            "horizon": horizon,
                            "model": f"MiniLM-{model_kind}",
                            "alignment": alignment,
                            "selected_alpha": alpha,
                            "semantic_dimension": current[0].shape[1],
                            "test_windows": len(x_test),
                        }
                    )
                    rows.append(result)
                    prediction_frames.append(
                        pd.DataFrame(
                            {
                                "domain": domain,
                                "horizon": horizon,
                                "model": f"MiniLM-{model_kind}",
                                "alignment": alignment,
                                "origin_index": np.repeat(test_origins, horizon),
                                "step": np.tile(np.arange(1, horizon + 1), len(test_origins)),
                                "actual_z": y_test.ravel(),
                                "prediction_z": forecast.ravel(),
                            }
                        )
                    )
                    print(
                        f"{domain} H={horizon} {alignment} {model_kind} MSE={result['mse']:.6f}",
                        flush=True,
                    )
    result = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "semantic_probe_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(output / "semantic_probe_predictions.csv", index=False)
    (output / "semantic_probe_config.json").write_text(
        json.dumps(
            {
                "alphas": ALPHAS,
                "semantic_scaling": "unit-normalized embeddings unchanged; quality features standardized within training scope",
                "shuffle": "independent deterministic permutation within each split",
                "encoder": "sentence-transformers/all-MiniLM-L6-v2, frozen cached embeddings",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def self_check() -> None:
    rng = np.random.default_rng(1)
    numeric = rng.normal(size=(20, 6))
    semantic = rng.normal(size=(20, 10))
    target = rng.normal(size=(20, 3))
    prediction, alpha = fit_probe(
        numeric[:12], semantic[:12], target[:12],
        numeric[12:16], semantic[12:16], target[12:16],
        numeric[16:], semantic[16:], "EarlyFusion",
    )
    assert prediction.shape == (4, 3) and alpha in ALPHAS
    print("self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("references/external/Time-MMD"))
    parser.add_argument("--semantic-root", type=Path, default=Path("data_processed/semantic_features"))
    parser.add_argument("--output", type=Path, default=Path("outputs/semantic_probe"))
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        self_check()
    else:
        print(run(args.root, args.semantic_root, args.output).to_string(index=False))
