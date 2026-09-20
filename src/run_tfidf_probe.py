"""Probe point-in-time text signal with leakage-safe TF-IDF ridge models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from data_utils import load_time_ordered_frame
from run_baselines import ALPHAS, CONFIGS, make_windows, metrics


SOURCE_MODES = ("report", "search", "all")


def assemble_documents(
    sample_index: pd.DataFrame,
    fact_map: dict[str, str],
    domain: str,
    origins: np.ndarray,
    source_mode: str,
) -> list[str]:
    rows = sample_index[sample_index["domain"] == domain].set_index("origin_index")
    documents: list[str] = []
    for origin in origins:
        row = rows.loc[int(origin)]
        ids: list[str] = []
        if source_mode in ("report", "all"):
            ids.extend(json.loads(row["report_text_ids"]))
        if source_mode in ("search", "all"):
            ids.extend(json.loads(row["search_text_ids"]))
        documents.append(" [SEP] ".join(fact_map[text_id] for text_id in ids))
    return documents


def shuffled(documents: list[str], seed: int) -> list[str]:
    order = np.random.default_rng(seed).permutation(len(documents))
    return [documents[index] for index in order]


def vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_features=1024,
        sublinear_tf=True,
        dtype=np.float32,
    )


def design_matrix(numeric: np.ndarray, text: sparse.spmatrix, model_kind: str) -> sparse.csr_matrix:
    if model_kind == "TextOnly":
        return text.tocsr()
    if model_kind == "EarlyFusion":
        return sparse.hstack([sparse.csr_matrix(numeric), text], format="csr")
    raise ValueError(model_kind)


def fit_variant(
    numeric_train: np.ndarray,
    y_train: np.ndarray,
    docs_train: list[str],
    numeric_val: np.ndarray,
    y_val: np.ndarray,
    docs_val: list[str],
    numeric_test: np.ndarray,
    docs_test: list[str],
    model_kind: str,
) -> tuple[np.ndarray, float, int]:
    selection_vectorizer = vectorizer()
    text_train = selection_vectorizer.fit_transform(docs_train)
    text_val = selection_vectorizer.transform(docs_val)
    x_train = design_matrix(numeric_train, text_train, model_kind)
    x_val = design_matrix(numeric_val, text_val, model_kind)
    validation_scores: dict[float, float] = {}
    for alpha in ALPHAS:
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(x_train, y_train)
        validation_scores[alpha] = metrics(y_val, model.predict(x_val))["mse"]
    best_alpha = min(validation_scores, key=validation_scores.get)

    final_vectorizer = vectorizer()
    docs_train_val = docs_train + docs_val
    text_train_val = final_vectorizer.fit_transform(docs_train_val)
    text_test = final_vectorizer.transform(docs_test)
    numeric_train_val = np.concatenate([numeric_train, numeric_val])
    y_train_val = np.concatenate([y_train, y_val])
    final_model = Ridge(alpha=best_alpha, solver="lsqr")
    final_model.fit(design_matrix(numeric_train_val, text_train_val, model_kind), y_train_val)
    prediction = final_model.predict(design_matrix(numeric_test, text_test, model_kind))
    return prediction, best_alpha, len(final_vectorizer.vocabulary_)


def run(root: Path, text_root: Path, output: Path, domains: list[str]) -> pd.DataFrame:
    corpus = pd.read_csv(text_root / "fact_corpus.csv")
    sample_index = pd.read_csv(text_root / "sample_text_index.csv")
    fact_map = corpus.set_index("text_id")["fact"].to_dict()
    rows: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []

    for domain in domains:
        config = CONFIGS[domain]
        path = root / "numerical" / domain / f"{domain}.csv"
        frame, _ = load_time_ordered_frame(path)
        raw = pd.to_numeric(frame["OT"], errors="coerce").to_numpy(float)
        if np.isnan(raw).any():
            raise ValueError(f"{domain} OT contains missing values")
        n = len(raw)
        train_end, val_end = int(n * 0.7), int(n * 0.8)
        train_mean, train_std = raw[:train_end].mean(), raw[:train_end].std()
        values = (raw - train_mean) / train_std

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
            for source_mode in SOURCE_MODES:
                train_docs = assemble_documents(sample_index, fact_map, domain, train_origins, source_mode)
                val_docs = assemble_documents(sample_index, fact_map, domain, val_origins, source_mode)
                test_docs = assemble_documents(sample_index, fact_map, domain, test_origins, source_mode)
                for alignment in ("aligned", "shuffled") if source_mode == "all" else ("aligned",):
                    docs = (train_docs, val_docs, test_docs)
                    if alignment == "shuffled":
                        docs = (
                            shuffled(train_docs, 202601),
                            shuffled(val_docs, 202602),
                            shuffled(test_docs, 202603),
                        )
                    for model_kind in ("TextOnly", "EarlyFusion"):
                        forecast, alpha, vocabulary_size = fit_variant(
                            x_train, y_train, docs[0], x_val, y_val, docs[1], x_test, docs[2], model_kind
                        )
                        result = metrics(y_test, forecast)
                        result.update(
                            {
                                "domain": domain,
                                "horizon": horizon,
                                "model": f"TFIDF-{model_kind}",
                                "source_mode": source_mode,
                                "alignment": alignment,
                                "selected_alpha": alpha,
                                "vocabulary_size": vocabulary_size,
                                "train_windows": len(x_train),
                                "validation_windows": len(x_val),
                                "test_windows": len(x_test),
                            }
                        )
                        rows.append(result)
                        predictions.append(
                            pd.DataFrame(
                                {
                                    "domain": domain,
                                    "horizon": horizon,
                                    "model": f"TFIDF-{model_kind}",
                                    "source_mode": source_mode,
                                    "alignment": alignment,
                                    "origin_index": np.repeat(test_origins, horizon),
                                    "step": np.tile(np.arange(1, horizon + 1), len(test_origins)),
                                    "actual_z": y_test.ravel(),
                                    "prediction_z": forecast.ravel(),
                                }
                            )
                        )
                        print(
                            f"{domain} H={horizon} {source_mode}/{alignment} {model_kind} "
                            f"MSE={result['mse']:.6f}",
                            flush=True,
                        )
    result_frame = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(output / "tfidf_probe_metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "tfidf_probe_predictions.csv", index=False)
    (output / "tfidf_probe_config.json").write_text(
        json.dumps(
            {
                "vectorizer": {
                    "ngram_range": [1, 2],
                    "min_df": 2,
                    "max_features": 1024,
                    "fit_scope": "training only for selection; train+validation for final refit",
                },
                "alpha_candidates": ALPHAS,
                "shuffle": "independent deterministic permutation within each split",
                "domains": domains,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result_frame


def self_check() -> None:
    docs = ["alpha beta", "beta gamma", "gamma delta", "alpha delta"]
    y = np.arange(8, dtype=float).reshape(4, 2)
    numeric = np.arange(12, dtype=float).reshape(4, 3)
    forecast, alpha, vocabulary_size = fit_variant(
        numeric[:2], y[:2], docs[:2], numeric[2:3], y[2:3], docs[2:3], numeric[3:], docs[3:], "EarlyFusion"
    )
    assert forecast.shape == (1, 2) and alpha in ALPHAS and vocabulary_size > 0
    assert sorted(shuffled(docs, 1)) == sorted(docs)
    print("self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("references/external/Time-MMD"))
    parser.add_argument("--text-root", type=Path, default=Path("data_processed/text"))
    parser.add_argument("--output", type=Path, default=Path("outputs/tfidf_probe"))
    parser.add_argument("--domains", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        self_check()
    else:
        print(run(args.root, args.text_root, args.output, args.domains).to_string(index=False))
