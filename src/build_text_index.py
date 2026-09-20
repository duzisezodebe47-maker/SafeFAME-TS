"""Build a point-in-time-safe fact corpus and sample-to-text index for Time-MMD."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from audit_timemmd import _valid_text
from data_utils import load_time_ordered_frame, numerical_intervals
from run_baselines import CONFIGS, CONFIRMATION_CONFIGS


FUTURE_LANGUAGE = re.compile(
    r"\b(?:forecast|predict(?:ed|ion)?|project(?:ed|ion)?|expect(?:ed|s)?|outlook|likely|will|could|may)\b",
    re.IGNORECASE,
)
MAX_FACTS_PER_SOURCE = 32
ALL_CONFIGS = {**CONFIGS, **CONFIRMATION_CONFIGS}


def normalize_fact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def stable_id(domain: str, source: str, start: pd.Timestamp, end: pd.Timestamp, fact: str) -> str:
    payload = f"{domain}|{source}|{start.isoformat()}|{end.isoformat()}|{fact}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def load_corpus(root: Path, domains: list[str]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for domain in domains:
        for path in sorted((root / "textual" / domain).glob("*.csv")):
            source = path.stem.rsplit("_", 1)[-1]
            raw = pd.read_csv(path)
            required = {"start_date", "end_date", "fact"}
            if not required.issubset(raw.columns):
                raise ValueError(f"{path} lacks {sorted(required - set(raw.columns))}")
            valid = _valid_text(raw["fact"])
            frame = pd.DataFrame(
                {
                    "domain": domain,
                    "source": source,
                    "start_date": pd.to_datetime(raw.loc[valid, "start_date"], errors="coerce"),
                    "end_date": pd.to_datetime(raw.loc[valid, "end_date"], errors="coerce"),
                    "fact": raw.loc[valid, "fact"].map(normalize_fact).astype("string"),
                }
            ).dropna(subset=["start_date", "end_date"])
            frame = frame[frame["end_date"] >= frame["start_date"]]
            before = len(frame)
            # Exact repeated facts are retained at their earliest known availability.
            frame = frame.sort_values("end_date").drop_duplicates(["source", "fact"], keep="first")
            frame["text_id"] = [
                stable_id(domain, source, start, end, fact)
                for start, end, fact in zip(frame["start_date"], frame["end_date"], frame["fact"])
            ]
            frame["characters"] = frame["fact"].str.len()
            frame["future_language_flag"] = frame["fact"].str.contains(FUTURE_LANGUAGE)
            frames.append(frame)
            audit.append(
                {
                    "domain": domain,
                    "source": source,
                    "raw_rows": len(raw),
                    "valid_dated_facts": before,
                    "deduplicated_facts": len(frame),
                    "exact_duplicates_removed": before - len(frame),
                    "future_language_flag_pct": (
                        round(float(frame["future_language_flag"].mean() * 100), 3)
                        if len(frame)
                        else None
                    ),
                }
            )
    corpus = pd.concat(frames, ignore_index=True).sort_values(["domain", "source", "end_date", "text_id"])
    if corpus["text_id"].duplicated().any():
        raise ValueError("Stable text IDs collided")
    return corpus, audit


def numerical_dates(path: Path) -> tuple[pd.Series, pd.Series]:
    frame, _ = load_time_ordered_frame(path)
    start, end = numerical_intervals(frame, path)
    if start.isna().any() or end.isna().any():
        raise ValueError(f"{path} contains invalid numerical intervals")
    return start, end


def build_domain_index(
    domain: str,
    starts: pd.Series,
    ends: pd.Series,
    facts: pd.DataFrame,
    input_len: int,
    max_per_source: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for origin in range(input_len, len(starts)):
        history_start = starts.iloc[origin - input_len]
        history_end = ends.iloc[origin - 1]
        cutoff = starts.iloc[origin]
        # Strict point-in-time rule: facts become available only after their end date,
        # and that availability must be strictly before the forecast origin.
        eligible = facts[(facts["end_date"] >= history_start) & (facts["end_date"] < cutoff)]
        row: dict[str, object] = {
            "domain": domain,
            "origin_index": origin,
            "history_start": history_start.date().isoformat(),
            "history_end": history_end.date().isoformat(),
            "forecast_start": cutoff.date().isoformat(),
        }
        all_selected: list[str] = []
        ages: list[float] = []
        for source in ("report", "search"):
            source_facts = eligible[eligible["source"] == source].sort_values(
                ["end_date", "text_id"], ascending=[False, True]
            )
            selected = source_facts.head(max_per_source)
            ids = selected["text_id"].tolist()
            row[f"{source}_available_count"] = len(source_facts)
            row[f"{source}_selected_count"] = len(ids)
            row[f"{source}_text_ids"] = json.dumps(ids)
            all_selected.extend(ids)
            ages.extend((cutoff - selected["end_date"]).dt.total_seconds().div(86400).tolist())
        row["total_selected_count"] = len(all_selected)
        row["latest_age_days"] = min(ages) if ages else np.nan
        row["mean_age_days"] = float(np.mean(ages)) if ages else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def coverage_summary(index: pd.DataFrame, row_counts: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for domain, config in ALL_CONFIGS.items():
        n = row_counts[domain]
        train_end, val_end = int(n * 0.7), int(n * 0.8)
        domain_index = index[index["domain"] == domain]
        for horizon in config.horizons:
            valid = domain_index[domain_index["origin_index"] + horizon <= n].copy()
            valid["split"] = np.select(
                [
                    valid["origin_index"] + horizon <= train_end,
                    (valid["origin_index"] >= train_end) & (valid["origin_index"] + horizon <= val_end),
                    valid["origin_index"] >= val_end,
                ],
                ["train", "validation", "test"],
                default="cross_boundary_excluded",
            )
            valid = valid[valid["split"] != "cross_boundary_excluded"]
            for split, part in valid.groupby("split"):
                nonempty_ages = part["latest_age_days"].dropna()
                rows.append(
                    {
                        "domain": domain,
                        "horizon": horizon,
                        "split": split,
                        "samples": len(part),
                        "any_text_coverage_pct": 100.0 * float((part["total_selected_count"] > 0).mean()),
                        "report_coverage_pct": 100.0 * float((part["report_selected_count"] > 0).mean()),
                        "search_coverage_pct": 100.0 * float((part["search_selected_count"] > 0).mean()),
                        "median_selected_facts": float(part["total_selected_count"].median()),
                        "median_latest_age_days": float(nonempty_ages.median()) if len(nonempty_ages) else np.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(["domain", "horizon", "split"])


def run(root: Path, output: Path, max_per_source: int = MAX_FACTS_PER_SOURCE) -> tuple[pd.DataFrame, pd.DataFrame]:
    domains = list(ALL_CONFIGS)
    corpus, corpus_audit = load_corpus(root, domains)
    index_frames: list[pd.DataFrame] = []
    row_counts: dict[str, int] = {}
    for domain, config in ALL_CONFIGS.items():
        path = root / "numerical" / domain / f"{domain}.csv"
        starts, ends = numerical_dates(path)
        row_counts[domain] = len(starts)
        index_frames.append(
            build_domain_index(
                domain,
                starts,
                ends,
                corpus[corpus["domain"] == domain],
                config.input_len,
                max_per_source,
            )
        )
    index = pd.concat(index_frames, ignore_index=True)
    summary = coverage_summary(index, row_counts)
    output.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(output / "fact_corpus.csv", index=False)
    index.to_csv(output / "sample_text_index.csv", index=False)
    summary.to_csv(output / "point_in_time_coverage.csv", index=False)
    (output / "text_index_audit.json").write_text(
        json.dumps(
            {
                "rules": {
                    "used_field": "fact",
                    "excluded_fields": ["preds", "pred"],
                    "availability": "fact.end_date < numerical forecast start",
                    "lookback": "same interval as numerical input window",
                    "deduplication": "same source and normalized exact fact; earliest availability retained",
                    "max_facts_per_source_per_sample": max_per_source,
                    "future_language_flag": "audit flag only; not automatically removed",
                },
                "corpus_audit": corpus_audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return corpus, summary


def self_check() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "textual" / "Demo").mkdir(parents=True)
        pd.DataFrame(
            {
                "start_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "end_date": ["2020-01-01", "2020-01-02", "2020-01-03"],
                "fact": ["past", "at cutoff", "future"],
                "preds": ["forbidden", "forbidden", "forbidden"],
            }
        ).to_csv(root / "textual" / "Demo" / "Demo_report.csv", index=False)
        corpus, _ = load_corpus(root, ["Demo"])
        starts = pd.Series(pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]))
        index = build_domain_index("Demo", starts, starts, corpus, input_len=1, max_per_source=32)
        cutoff_row = index[index["forecast_start"] == "2020-01-02"].iloc[0]
        ids = json.loads(cutoff_row["report_text_ids"])
        selected_facts = corpus.set_index("text_id").loc[ids, "fact"].tolist()
        assert selected_facts == ["past"]
        assert not corpus["fact"].str.contains("forbidden").any()
    print("self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("references/external/Time-MMD"))
    parser.add_argument("--output", type=Path, default=Path("data_processed/text"))
    parser.add_argument("--max-per-source", type=int, default=MAX_FACTS_PER_SOURCE)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        self_check()
    else:
        corpus_result, summary_result = run(args.root, args.output, args.max_per_source)
        print(f"corpus_rows={len(corpus_result)}")
        print(summary_result.to_string(index=False))
