"""Aggregate fact embeddings into point-in-time sample semantic features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


QUALITY_NAMES = [
    "report_log_available",
    "search_log_available",
    "report_log_selected",
    "search_log_selected",
    "report_missing",
    "search_missing",
    "latest_age_ratio",
    "mean_age_ratio",
    "report_future_language_fraction",
    "search_future_language_fraction",
]


def normalized_weighted_mean(vectors: np.ndarray, weights: np.ndarray, dimension: int) -> np.ndarray:
    if len(vectors) == 0:
        return np.zeros(dimension, dtype=np.float32)
    mean = np.average(vectors, axis=0, weights=weights)
    norm = np.linalg.norm(mean)
    return (mean / norm if norm > 1e-8 else mean).astype(np.float32)


def run(text_root: Path, embedding_root: Path, output: Path) -> dict[str, dict[str, object]]:
    corpus = pd.read_csv(text_root / "fact_corpus.csv")
    corpus["end_date"] = pd.to_datetime(corpus["end_date"])
    corpus = corpus.set_index("text_id")
    embedding_index = pd.read_csv(embedding_root / "fact_embedding_index.csv")
    embeddings = np.load(embedding_root / "fact_embeddings.npy", mmap_mode="r")
    row_map = embedding_index.set_index("text_id")["row"].to_dict()
    sample_index = pd.read_csv(text_root / "sample_text_index.csv")
    dimension = embeddings.shape[1]
    manifest: dict[str, dict[str, object]] = {}
    output.mkdir(parents=True, exist_ok=True)

    for domain, rows in sample_index.groupby("domain", sort=True):
        rows = rows.sort_values("origin_index")
        report_features: list[np.ndarray] = []
        search_features: list[np.ndarray] = []
        quality_features: list[list[float]] = []
        for row in rows.itertuples(index=False):
            forecast_start = pd.Timestamp(row.forecast_start)
            history_start = pd.Timestamp(row.history_start)
            history_days = max(float((forecast_start - history_start).days), 1.0)
            half_life = max(history_days / 4.0, 1.0)
            source_vectors: dict[str, np.ndarray] = {}
            future_fractions: dict[str, float] = {}
            for source in ("report", "search"):
                ids = json.loads(getattr(row, f"{source}_text_ids"))
                if ids:
                    indices = [row_map[text_id] for text_id in ids]
                    ages = np.asarray(
                        [(forecast_start - corpus.loc[text_id, "end_date"]).days for text_id in ids], dtype=float
                    )
                    if np.any(ages <= 0):
                        raise AssertionError("Point-in-time index contains text at or after forecast origin")
                    weights = np.exp(-np.log(2.0) * ages / half_life)
                    source_vectors[source] = normalized_weighted_mean(embeddings[indices], weights, dimension)
                    future_fractions[source] = float(corpus.loc[ids, "future_language_flag"].astype(float).mean())
                else:
                    source_vectors[source] = np.zeros(dimension, dtype=np.float32)
                    future_fractions[source] = 0.0
            report_features.append(source_vectors["report"])
            search_features.append(source_vectors["search"])
            latest_ratio = float(row.latest_age_days) / history_days if pd.notna(row.latest_age_days) else 1.0
            mean_ratio = float(row.mean_age_days) / history_days if pd.notna(row.mean_age_days) else 1.0
            quality_features.append(
                [
                    np.log1p(row.report_available_count),
                    np.log1p(row.search_available_count),
                    np.log1p(row.report_selected_count),
                    np.log1p(row.search_selected_count),
                    float(row.report_selected_count == 0),
                    float(row.search_selected_count == 0),
                    latest_ratio,
                    mean_ratio,
                    future_fractions["report"],
                    future_fractions["search"],
                ]
            )
        arrays = {
            "origin_index": rows["origin_index"].to_numpy(np.int32),
            "report_embedding": np.stack(report_features),
            "search_embedding": np.stack(search_features),
            "quality": np.asarray(quality_features, dtype=np.float32),
        }
        if any(not np.isfinite(array).all() for array in arrays.values()):
            raise AssertionError(f"{domain} semantic feature cache contains non-finite values")
        np.savez_compressed(output / f"{domain}.npz", **arrays)
        manifest[domain] = {
            "samples": len(rows),
            "origin_min": int(rows["origin_index"].min()),
            "origin_max": int(rows["origin_index"].max()),
            "embedding_dimension": dimension,
            "quality_dimension": len(QUALITY_NAMES),
            "report_missing_pct": 100.0 * float(np.mean(arrays["quality"][:, 4])),
            "search_missing_pct": 100.0 * float(np.mean(arrays["quality"][:, 5])),
        }
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "aggregation": "L2-normalized exponentially recency-weighted mean",
                "half_life": "one quarter of each sample's numerical history duration",
                "quality_names": QUALITY_NAMES,
                "domains": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def verify(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for domain, expected in manifest["domains"].items():
        cache = np.load(output / f"{domain}.npz")
        assert len(cache["origin_index"]) == expected["samples"]
        assert cache["report_embedding"].shape[1] == expected["embedding_dimension"]
        assert cache["search_embedding"].shape[1] == expected["embedding_dimension"]
        assert cache["quality"].shape[1] == expected["quality_dimension"]
        assert np.all(np.diff(cache["origin_index"]) > 0)
    print(f"verified semantic caches for {len(manifest['domains'])} domains")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-root", type=Path, default=Path("data_processed/text"))
    parser.add_argument(
        "--embedding-root", type=Path, default=Path("data_processed/embeddings/all_minilm_l6_v2")
    )
    parser.add_argument("--output", type=Path, default=Path("data_processed/semantic_features"))
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not args.verify_only:
        print(json.dumps(run(args.text_root, args.embedding_root, args.output), ensure_ascii=False, indent=2))
    verify(args.output)
