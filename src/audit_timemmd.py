"""Audit raw Time-MMD numerical and textual CSV files.

The script never modifies source data. It writes reproducible audit tables that
can be cited by the course report and checked before model training.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import load_time_ordered_frame, numerical_intervals


MISSING_TEXT = r"^\s*(?:NA\b.*)?$"


def _valid_text(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    return text.ne("") & ~text.str.match(MISSING_TEXT, case=False)


_numerical_intervals = numerical_intervals


def audit_numerical(path: Path) -> dict[str, object]:
    frame = pd.read_csv(path)
    if "OT" not in frame:
        raise ValueError(f"{path} must contain OT")

    dates, _ = _numerical_intervals(frame, path)
    _, order_audit = load_time_ordered_frame(path)
    unique_dates = dates.dropna().sort_values().drop_duplicates()
    steps = unique_dates.diff().dropna().dt.total_seconds().div(86400)
    mode_step = float(steps.mode().iloc[0]) if not steps.empty else None

    return {
        "domain": path.parent.name,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "numeric_columns": int(len(frame.select_dtypes(include=np.number).columns)),
        "start": dates.min().date().isoformat() if dates.notna().any() else None,
        "end": dates.max().date().isoformat() if dates.notna().any() else None,
        "invalid_dates": int(dates.isna().sum()),
        "duplicate_dates": int(dates.duplicated().sum()),
        **order_audit,
        "modal_step_days": mode_step,
        "ot_missing": int(frame["OT"].isna().sum()),
        "all_missing_cells": int(frame.isna().sum().sum()),
    }


def audit_text(path: Path, numerical_path: Path) -> dict[str, object]:
    text = pd.read_csv(path)
    numerical = pd.read_csv(numerical_path)
    required = {"start_date", "end_date", "fact"}
    if not required.issubset(text.columns):
        raise ValueError(f"{path} is missing {sorted(required - set(text.columns))}")

    text_start = pd.to_datetime(text["start_date"], errors="coerce")
    text_end = pd.to_datetime(text["end_date"], errors="coerce")
    num_start, num_end = _numerical_intervals(numerical, numerical_path)
    valid_fact = _valid_text(text["fact"])
    usable_start = text_start[valid_fact].reset_index(drop=True)
    usable_end = text_end[valid_fact].reset_index(drop=True)

    # ponytail: vectorized per numerical interval is O(N*M); current public
    # Time-MMD files are small enough. Replace with an interval index if the
    # corpus grows by roughly an order of magnitude.
    overlap_counts = [
        int(((usable_start <= end) & (usable_end >= start)).sum())
        for start, end in zip(num_start, num_end)
    ]
    prediction_col = "preds" if "preds" in text else "pred" if "pred" in text else None
    valid_prediction = _valid_text(text[prediction_col]) if prediction_col else pd.Series(False, index=text.index)

    return {
        "domain": path.parent.name,
        "source": path.stem.rsplit("_", 1)[-1],
        "rows": int(len(text)),
        "start": text_start.min().date().isoformat() if text_start.notna().any() else None,
        "end": text_end.max().date().isoformat() if text_end.notna().any() else None,
        "invalid_intervals": int(pd.concat([text_start, text_end], axis=1).isna().any(axis=1).sum()),
        "usable_fact_rows": int(valid_fact.sum()),
        "unique_usable_facts": int(text.loc[valid_fact, "fact"].nunique()),
        "fact_missing_pct": round(float((~valid_fact).mean() * 100), 3),
        "prediction_present_pct": round(float(valid_prediction.mean() * 100), 3),
        "duplicate_interval_pct": round(float(text.duplicated(["start_date", "end_date"]).mean() * 100), 3),
        "numerical_interval_coverage_pct": round(float(np.mean(np.asarray(overlap_counts) > 0) * 100), 3),
        "median_texts_per_interval": float(np.median(overlap_counts)),
        "max_texts_per_interval": int(max(overlap_counts, default=0)),
        "model_use": "fact_only; predictions excluded",
    }


def run_audit(root: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    numerical_rows = []
    text_rows = []
    for numerical_path in sorted((root / "numerical").glob("*/*.csv")):
        numerical_rows.append(audit_numerical(numerical_path))
        domain = numerical_path.parent.name
        for text_path in sorted((root / "textual" / domain).glob("*.csv")):
            text_rows.append(audit_text(text_path, numerical_path))

    if not numerical_rows or not text_rows:
        raise ValueError(f"No Time-MMD CSV files found under {root}")

    numerical_audit = pd.DataFrame(numerical_rows)
    text_audit = pd.DataFrame(text_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    numerical_audit.to_csv(output_dir / "timemmd_numerical_audit.csv", index=False)
    text_audit.to_csv(output_dir / "timemmd_text_audit.csv", index=False)
    summary = {
        "source_root": str(root.resolve()),
        "numerical": numerical_rows,
        "textual": text_rows,
        "rules": {
            "raw_data_modified": False,
            "forecast_text_field": "fact",
            "excluded_field": "preds/pred",
            "reason": "prediction text is future-looking by construction and risks target leakage",
        },
    }
    (output_dir / "timemmd_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return numerical_audit, text_audit


def self_check() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "numerical" / "Demo").mkdir(parents=True)
        (root / "textual" / "Demo").mkdir(parents=True)
        pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "OT": [1.0, 2.0],
            }
        ).to_csv(root / "numerical" / "Demo" / "Demo.csv", index=False)
        pd.DataFrame(
            {
                "start_date": ["2020-01-01", "2020-01-02"],
                "end_date": ["2020-01-01", "2020-01-02"],
                "fact": ["known event", "NA"],
                "preds": ["future guess", "NA"],
            }
        ).to_csv(root / "textual" / "Demo" / "Demo_report.csv", index=False)
        numerical, textual = run_audit(root, root / "out")
        assert numerical.loc[0, "rows"] == 2
        assert textual.loc[0, "usable_fact_rows"] == 1
        assert textual.loc[0, "numerical_interval_coverage_pct"] == 50.0
        assert (root / "out" / "timemmd_audit.json").exists()
    print("self-check passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("references/external/Time-MMD"))
    parser.add_argument("--output", type=Path, default=Path("outputs/audit"))
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.self_check:
        self_check()
    else:
        numerical_result, text_result = run_audit(args.root, args.output)
        print(numerical_result.to_string(index=False))
        print(text_result.to_string(index=False))
