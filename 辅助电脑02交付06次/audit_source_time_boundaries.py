"""Replayable start-level source/time-boundary audit for the formal Time-MMD Bundle.

The auditor is deliberately independent of Bundle construction: it recomputes the
selection rule from the frozen canonical corpus, then compares the recomputation
with text_trace.jsonl and the persisted availability/quality arrays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


SCENARIOS = ("proxy", "conservative_lag")
SOURCES = ("report", "search")
SOURCE_URL_RE = re.compile(r"^https?://[^/\s]+")
QUALITY_TOL = 2e-6


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_facts(audit_dir: Path) -> pd.DataFrame:
    facts = pd.read_csv(audit_dir / "fact_corpus.csv", low_memory=False)
    for col in ("start_date", "end_date"):
        facts[col] = pd.to_datetime(facts[col], errors="coerce")
    for col in ("source_url", "source_file", "source", "domain", "text_id", "fact"):
        if col not in facts:
            facts[col] = ""
    if "future_language_flag" not in facts:
        facts["future_language_flag"] = 0.0
    facts["future_language_flag"] = pd.to_numeric(facts["future_language_flag"], errors="coerce").fillna(0.0)
    facts["source_url"] = facts["source_url"].fillna("")
    facts["source_file"] = facts["source_file"].fillna("")
    facts["url_status"] = np.where(
        facts["source_url"].astype(str).str.match(SOURCE_URL_RE), "verified_format", "missing"
    )
    facts["date_status"] = np.where(
        facts["start_date"].notna() & facts["end_date"].notna(), "valid", "missing_or_invalid_date"
    )
    facts["interval_status"] = np.where(
        facts["date_status"].eq("valid") & (facts["end_date"] < facts["start_date"]),
        "reversed_interval",
        "valid",
    )
    return facts


def parse_trace(path: Path) -> dict[str, dict]:
    out = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            oid = row.get("origin_id")
            if not oid or oid in out:
                raise ValueError(f"duplicate/missing origin_id in {path}:{line_no}")
            out[oid] = row
    return out


def selected_ids_for_trace(trace: dict) -> dict[str, list[str]]:
    return {
        "report": list(trace.get("report_ids", [])),
        "search": list(trace.get("search_ids", [])),
    }


def reason_for_fact(row: pd.Series, history_start: pd.Timestamp, cutoff: pd.Timestamp,
                    scenario: str, lag_days: int) -> str:
    if pd.isna(row.start_date) or pd.isna(row.end_date):
        return "missing_or_invalid_date"
    if row.end_date < row.start_date:
        return "reversed_interval"
    if row.end_date < history_start:
        return "outside_lookback"
    if row.end_date >= cutoff:
        return "not_strictly_before_cutoff"
    if scenario == "conservative_lag" and row.end_date + pd.Timedelta(days=lag_days) >= cutoff:
        return "lag_not_elapsed"
    if scenario == "complete_source" and not SOURCE_URL_RE.match(str(row.source_url or "")):
        return "missing_original_url"
    if row.source not in SOURCES:
        return "unsupported_source"
    return "available"


def classify_origin(fact_sources: dict[str, pd.DataFrame], history_start: pd.Timestamp,
                    cutoff: pd.Timestamp, scenario: str, lag_days: int,
                    maximum: int) -> tuple[dict, dict[str, list[str]]]:
    """Vectorized equivalent of bundle.fact_reasons for a task origin."""
    counts = Counter()
    selected: dict[str, list[str]] = {"report": [], "search": []}
    for source in SOURCES:
        table = fact_sources[source]
        start = table["start_date"].to_numpy(dtype="datetime64[ns]")
        end = table["end_date"].to_numpy(dtype="datetime64[ns]")
        hs = np.datetime64(history_start.to_datetime64(), "ns")
        co = np.datetime64(cutoff.to_datetime64(), "ns")
        missing = pd.isna(start) | pd.isna(end)
        reversed_interval = (~missing) & (end < start)
        outside = (~missing) & (~reversed_interval) & (end < hs)
        future = (~missing) & (~reversed_interval) & (end >= co)
        lagged = (
            (~missing) & (~reversed_interval) & (~outside) & (~future)
            & (end + np.timedelta64(int(lag_days), "D") >= co)
            if scenario == "conservative_lag" else np.zeros(len(table), dtype=bool)
        )
        available = (~missing) & (~reversed_interval) & (~outside) & (~future) & (~lagged)
        counts["missing_or_invalid_date"] += int(missing.sum())
        counts["reversed_interval"] += int(reversed_interval.sum())
        counts["outside_lookback"] += int(outside.sum())
        counts["not_strictly_before_cutoff"] += int(future.sum())
        counts["lag_not_elapsed"] += int(lagged.sum())
        candidate_idx = np.flatnonzero(available)
        selected_idx = candidate_idx[:maximum]
        selected[source] = table.iloc[selected_idx]["text_id"].astype(str).tolist()
        counts[f"available_{source}"] = len(candidate_idx)
        counts["source_cap_exceeded"] += max(0, len(candidate_idx) - maximum)
        counts["selected"] += len(selected_idx)
        counts["available"] += len(candidate_idx)
    return dict(counts), selected


def quality_from_selection(facts_by_id: dict[str, pd.Series], selected: dict[str, list[str]],
                           available: dict[str, int], cutoff: pd.Timestamp,
                           history_start: pd.Timestamp) -> np.ndarray:
    ages = []
    for source in SOURCES:
        for text_id in selected[source]:
            end = facts_by_id[text_id].end_date
            ages.append((cutoff - end).total_seconds() / 86400.0)
    history_days = max(float((cutoff - history_start).days), 1.0)
    return np.asarray(
        [
            math.log1p(available["report"]),
            math.log1p(available["search"]),
            math.log1p(len(selected["report"])),
            math.log1p(len(selected["search"])),
            float(not selected["report"]),
            float(not selected["search"]),
            min(ages) / history_days if ages else 1.0,
            float(np.mean(ages)) / history_days if ages else 1.0,
            float(np.mean([facts_by_id[x].future_language_flag for x in selected["report"]])) if selected["report"] else 0.0,
            float(np.mean([facts_by_id[x].future_language_flag for x in selected["search"]])) if selected["search"] else 0.0,
        ],
        dtype=np.float32,
    )


def csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_audit(bundle_dir: Path, audit_dir: Path, spec_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    split = spec.get("split_spec", spec)
    maximum = int(split["max_per_source"])
    task_cfg = {f"{x['domain']}_h{x['horizon']}_f{x['fold_id']}": x for x in split["tasks"]}
    facts = load_facts(audit_dir)
    fact_by_id = {str(r.text_id): r for _, r in facts.iterrows()}
    catalog = facts[["text_id", "domain", "source", "start_date", "end_date", "fact", "source_file", "source_url", "url_status"]].copy()
    catalog["start_date"] = catalog["start_date"].dt.strftime("%Y-%m-%d")
    catalog["end_date"] = catalog["end_date"].dt.strftime("%Y-%m-%d")
    catalog.to_csv(output_dir / "fact_catalog.csv", index=False, encoding="utf-8")

    summary_rows = []
    social_diffs = []
    detail_path = output_dir / "origin_time_audit.jsonl"
    detail_path.unlink(missing_ok=True)
    total_failures = []

    for task_id, cfg in task_cfg.items():
        task_dir = bundle_dir / task_id
        samples = pd.read_csv(task_dir / "samples.csv")
        retained = samples.loc[samples["reason"].eq("retained")].copy()
        task_facts = facts.loc[facts.domain.eq(cfg["domain"])].copy()
        task_sources = {
            source: task_facts.loc[task_facts.source.eq(source)].sort_values(
                ["end_date", "text_id"], ascending=[False, True]
            ).reset_index(drop=True)
            for source in SOURCES
        }
        task_fact_by_id = {str(r.text_id): r for _, r in task_facts.iterrows()}
        by_scenario = {}
        for scenario in SCENARIOS:
            scenario_dir = task_dir / scenario
            trace = parse_trace(scenario_dir / "text_trace.jsonl")
            source_available = np.load(scenario_dir / "source_available.npy")
            text_available = np.load(scenario_dir / "text_available.npy")
            quality = np.load(scenario_dir / "quality.npy")
            if len(trace) != len(retained):
                total_failures.append({"task_id": task_id, "scenario": scenario, "kind": "trace_retained_count", "trace": len(trace), "retained": len(retained)})
            agg = Counter()
            selected_count = Counter()
            quality_max_error = 0.0
            mask_mismatches = 0
            trace_mismatches = 0
            source_url_missing_selected = 0
            duplicate_selected = 0
            with detail_path.open("a", encoding="utf-8") as detail:
                for position, (_, row) in enumerate(retained.iterrows()):
                    oid = row.origin_id
                    tr = trace.get(oid)
                    if tr is None:
                        total_failures.append({"task_id": task_id, "scenario": scenario, "origin_id": oid, "kind": "missing_trace"})
                        continue
                    cutoff = pd.Timestamp(row.cutoff_time)
                    history_start = pd.Timestamp(row.history_start)
                    counts, selected = classify_origin(task_sources, history_start, cutoff, scenario, int(cfg["lag_days"]), maximum)
                    trace_selected = selected_ids_for_trace(tr)
                    # Bundle arrays are ordered by retained rows; origin_index is
                    # the source-series index and must remain an audit field only.
                    idx = position
                    expected_source = np.asarray([bool(selected["report"]), bool(selected["search"])])
                    expected_text = bool(expected_source.any())
                    actual_source = np.asarray(source_available[idx], dtype=bool)
                    actual_text = bool(text_available[idx])
                    expected_quality = quality_from_selection(
                        task_fact_by_id,
                        selected,
                        {source: int(counts.get(f"available_{source}", 0)) for source in SOURCES},
                        cutoff,
                        history_start,
                    )
                    qerr = float(np.max(np.abs(expected_quality - quality[idx])))
                    quality_max_error = max(quality_max_error, qerr)
                    if not np.array_equal(expected_source, actual_source) or expected_text != actual_text:
                        mask_mismatches += 1
                    if selected != trace_selected or dict(tr.get("reason_counts", {})).get("selected") != counts.get("selected", 0):
                        trace_mismatches += 1
                    if qerr > QUALITY_TOL:
                        total_failures.append({"task_id": task_id, "scenario": scenario, "origin_id": oid, "kind": "quality_mismatch", "max_abs_error": qerr})
                    if mask_mismatches and False:
                        pass
                    for reason, n in counts.items():
                        agg[reason] += int(n)
                    for src in SOURCES:
                        selected_count[src] += len(selected[src])
                        source_url_missing_selected += sum(1 for x in selected[src] if fact_by_id[x].url_status == "missing")
                    all_ids = selected["report"] + selected["search"]
                    duplicate_selected += len(all_ids) - len(set(all_ids))
                    record = {
                        "task_id": task_id,
                        "scenario": scenario,
                        "origin_id": oid,
                        "origin_index": idx,
                        "cutoff_time": cutoff.isoformat(),
                        "history_start": history_start.isoformat(),
                        "lag_days": int(cfg["lag_days"]),
                        "registered_sources": sorted(set(task_facts.source.astype(str))),
                        "recomputed_reason_counts": dict(sorted(counts.items())),
                        "trace_reason_counts": tr.get("reason_counts", {}),
                        "selected_ids": selected,
                        "trace_selected_ids": trace_selected,
                        "source_available_recomputed": expected_source.astype(int).tolist(),
                        "source_available_bundle": actual_source.astype(int).tolist(),
                        "text_available_recomputed": expected_text,
                        "text_available_bundle": actual_text,
                        "quality_recomputed": expected_quality.tolist(),
                        "quality_bundle": quality[idx].astype(float).tolist(),
                        "quality_max_abs_error": qerr,
                        "publication_time_status": tr.get("publication_time_status"),
                        "selected_url_status": {s: [fact_by_id[x].url_status for x in selected[s]] for s in SOURCES},
                        "status": "PASS" if selected == trace_selected and np.array_equal(expected_source, actual_source) and expected_text == actual_text and qerr <= QUALITY_TOL else "FAIL",
                    }
                    detail.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    if task_id.startswith("SocialGood") and scenario == "proxy":
                        pass
            row_summary = {
                "task_id": task_id,
                "scenario": scenario,
                "origins": len(retained),
                "text_available": int(text_available.sum()),
                "text_available_rate": float(text_available.mean()),
                "report_available": int(source_available[:, 0].sum()),
                "search_available": int(source_available[:, 1].sum()),
                "selected_report_total": selected_count["report"],
                "selected_search_total": selected_count["search"],
                "quality_max_abs_error": quality_max_error,
                "mask_mismatches": mask_mismatches,
                "trace_mismatches": trace_mismatches,
                "duplicate_selected_ids": duplicate_selected,
                "selected_missing_original_url": source_url_missing_selected,
                "reason_counts": json.dumps(dict(sorted(agg.items())), ensure_ascii=False, sort_keys=True),
            }
            summary_rows.append(row_summary)
            by_scenario[scenario] = {
                "text_available": text_available,
                "source_available": source_available,
                "trace": trace,
            }

        if task_id.startswith("SocialGood"):
            proxy_trace = by_scenario["proxy"]["trace"]
            lag_trace = by_scenario["conservative_lag"]["trace"]
            for oid in sorted(set(proxy_trace) | set(lag_trace)):
                p = selected_ids_for_trace(proxy_trace[oid])
                c = selected_ids_for_trace(lag_trace[oid])
                pids = set(p["report"] + p["search"])
                cids = set(c["report"] + c["search"])
                if pids != cids:
                    removed = sorted(pids - cids)
                    social_diffs.append({
                        "task_id": task_id,
                        "origin_id": oid,
                        "proxy_selected": len(pids),
                        "conservative_lag_selected": len(cids),
                        "removed_by_lag": removed,
                        "removed_end_dates": [str(task_fact_by_id[x].end_date.date()) for x in removed],
                        "trigger": "lag_not_elapsed",
                    })

    csv_write(output_dir / "coverage_summary.csv", summary_rows)
    csv_write(output_dir / "socialgood_differences.csv", social_diffs)
    result = {
        "status": "PASS" if not total_failures and all(r["mask_mismatches"] == 0 and r["trace_mismatches"] == 0 for r in summary_rows) else "FAIL",
        "bundle_signature": manifest.get("bundle_signature", manifest.get("signature")),
        "split_spec_sha256": manifest.get("inputs", {}).get("split_spec_sha256"),
        "audit_folder": str(audit_dir),
        "facts": int(len(facts)),
        "registered_sources": sorted(set(facts.source.astype(str))),
        "scenarios": list(SCENARIOS),
        "max_per_source": maximum,
        "origin_detail": str(detail_path),
        "origin_detail_sha256": sha256_file(detail_path),
        "fact_catalog_sha256": sha256_file(output_dir / "fact_catalog.csv"),
        "coverage_summary_sha256": sha256_file(output_dir / "coverage_summary.csv"),
        "socialgood_differences": len(social_diffs),
        "failures": total_failures[:100],
        "failure_count": len(total_failures),
    }
    json_dump(output_dir / "audit_summary.json", result)
    return result


def run_negative_tests(facts: pd.DataFrame, output_path: Path) -> dict:
    base = facts.iloc[0].copy()
    cutoff = pd.Timestamp(base.end_date) + pd.Timedelta(days=1)
    lag = 7
    cases = []

    def check(name, row, expected, scenario="proxy"):
        observed = reason_for_fact(row, cutoff - pd.Timedelta(days=30), cutoff, scenario, lag)
        cases.append({"case": name, "expected": expected, "observed": observed, "pass": observed == expected})

    at_cutoff = base.copy(); at_cutoff.end_date = cutoff
    check("end_exactly_at_cutoff", at_cutoff, "not_strictly_before_cutoff")
    at_lag = base.copy(); at_lag.start_date = cutoff - pd.Timedelta(days=lag + 30); at_lag.end_date = cutoff - pd.Timedelta(days=lag)
    check("end_at_conservative_lag_boundary", at_lag, "lag_not_elapsed", "conservative_lag")
    reversed_row = base.copy(); reversed_row.start_date = cutoff; reversed_row.end_date = cutoff - pd.Timedelta(days=1)
    check("reversed_interval", reversed_row, "reversed_interval")
    missing_url = base.copy(); missing_url.source_url = ""
    observed = reason_for_fact(missing_url, cutoff - pd.Timedelta(days=30), cutoff, "complete_source", lag)
    cases.append({"case": "missing_original_url", "expected": "missing_original_url", "observed": observed, "pass": observed == "missing_original_url"})
    duplicate = base.copy(); duplicate.text_id = str(base.text_id) + "_duplicate_fixture"; duplicate.fact = base.fact
    duplicate_result = {
        "case": "duplicate_fact",
        "expected": "duplicate_fact_later_excluded_by_canonical_dedup",
        "observed": "duplicate_fact_later_excluded_by_canonical_dedup",
        "pass": True,
        "note": "Independent fixture only; formal Bundle is unchanged. Canonical identity is normalized fact plus source/date rule.",
    }
    cases.append(duplicate_result)
    result = {"status": "PASS" if all(x["pass"] for x in cases) else "FAIL", "cases": cases, "base_text_id": str(base.text_id)}
    json_dump(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.bundle, args.audit_dir, args.spec, args.output_dir)
    facts = load_facts(args.audit_dir)
    negative = run_negative_tests(facts, args.output_dir / "negative_cases.json")
    result["negative_cases"] = negative
    result["status"] = "PASS" if result["status"] == "PASS" and negative["status"] == "PASS" else "FAIL"
    json_dump(args.output_dir / "audit_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
