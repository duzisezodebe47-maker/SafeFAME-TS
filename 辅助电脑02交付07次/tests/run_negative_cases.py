"""Nine independent boundary and package-integrity negative cases."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

import pandas as pd


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def expect(label: str, expected: str, call) -> dict:
    try:
        observed = call()
    except Exception as exc:  # negative cases intentionally exercise rejection paths
        observed = f"{type(exc).__name__}: {exc}"
    return {"case": label, "expected_contains": expected, "observed": str(observed), "pass": expected in str(observed)}


def run(root: Path) -> dict:
    replay = module(root / "code/replay.py", "replay_for_negative_tests")
    audit = module(root / "code/audit_source_time_boundaries.py", "audit_for_negative_tests")
    corpus = pd.read_csv(root / "inputs/fact_corpus.csv", nrows=1)
    base = corpus.iloc[0].copy()
    base.start_date = pd.Timestamp("2020-01-01")
    base.end_date = pd.Timestamp("2020-01-01")
    cutoff = pd.Timestamp("2020-02-01")
    history = pd.Timestamp("2019-12-01")
    cases = []

    row = base.copy(); row.start_date = cutoff; row.end_date = cutoff
    cases.append(expect("end_exactly_at_cutoff", "not_strictly_before_cutoff", lambda: audit.reason_for_fact(row, history, cutoff, "proxy", 7)))
    row = base.copy(); row.end_date = cutoff - pd.Timedelta(days=7)
    cases.append(expect("lag_exact_boundary", "lag_not_elapsed", lambda: audit.reason_for_fact(row, history, cutoff, "conservative_lag", 7)))
    row = base.copy(); row.start_date = cutoff; row.end_date = cutoff - pd.Timedelta(days=1)
    cases.append(expect("reversed_interval", "reversed_interval", lambda: audit.reason_for_fact(row, history, cutoff, "proxy", 7)))
    row = base.copy(); row.source_url = ""
    cases.append(expect("missing_original_url", "missing_original_url", lambda: audit.reason_for_fact(row, history, cutoff, "complete_source", 7)))
    duplicate = pd.concat([corpus, corpus], ignore_index=True)
    cases.append(expect("duplicate_fact", "duplicate text_id", lambda: replay.validate_corpus(duplicate)))

    with tempfile.TemporaryDirectory(prefix="safe_fame_negative_", dir=root.parent) as temp:
        fixture = Path(temp)
        (fixture / "inputs").mkdir()
        original = root / "inputs/fact_corpus.csv"
        original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
        manifest = {"files": [{"relative_path": "inputs/fact_corpus.csv", "bytes": original.stat().st_size, "sha256": original_sha}]}
        (fixture / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        cases.append(expect("release_missing_corpus", "Required Release input missing", lambda: replay.verify_inputs(fixture)))
        (fixture / "inputs/fact_corpus.csv").write_text("corrupted\n", encoding="utf-8")
        cases.append(expect("corpus_hash_changed", "hash/size mismatch", lambda: replay.verify_inputs(fixture)))
        cases.append(expect("required_column_missing", "missing required columns", lambda: replay.validate_corpus(corpus.drop(columns=["text_id"]))))
        outside = fixture.parent / "external_prohibited_input.csv"
        cases.append(expect("outside_release_path", "escapes Release", lambda: replay.inside(fixture, "../external_prohibited_input.csv")))

    result = {"status": "PASS" if all(x["pass"] for x in cases) else "FAIL", "cases": cases}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.root.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if result["status"] == "PASS" else 2)
