"""Build compact, reproducible origin traces without copying large arrays to Git."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FIRST = ROOT / "辅助电脑02交付01次"
SECOND = ROOT / "辅助电脑02交付02次"
sys.path[:0] = [str(FIRST), str(SECOND)]

from audit import digest, write_json  # noqa: E402
from round3_pipeline import latest_audit, latest_preview  # noqa: E402


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def trace(bundle: Path, audit: Path, task_id: str, position: int, scenario: str) -> dict:
    task = bundle / task_id
    samples = pd.read_csv(task / "samples.csv")
    row = samples.iloc[position]
    domain = task_id.split("_h")[0]
    numerical = pd.read_csv(audit / f"{domain}_numerical.csv")
    lineage = pd.read_csv(audit / f"{domain}_numeric_lineage.csv")
    origin, horizon = int(row.origin_index), int(row.horizon)
    history_len = np.load(task / "numeric_history.npy", mmap_mode="r").shape[1]
    clean_indices = list(range(origin - history_len, origin + horizon))
    numeric_rows = numerical.iloc[clean_indices]
    lineage_rows = lineage[pd.to_numeric(lineage.clean_index, errors="coerce").isin(clean_indices)]
    text_trace = lines(task / scenario / "text_trace.jsonl")[position]
    semantic = np.load(task / scenario / "semantic.npy", mmap_mode="r")[position]
    quality = np.load(task / scenario / "quality.npy", mmap_mode="r")[position]
    frequency = np.load(task / "frequency.npy", mmap_mode="r")[position]
    text_available = bool(np.load(task / scenario / "text_available.npy", mmap_mode="r")[position])
    target_raw = np.load(task / "targets_raw.npy", mmap_mode="r")[position]
    target_std = np.load(task / "targets_standardized.npy", mmap_mode="r")[position]
    return {
        "task_id": task_id, "scenario": scenario, "origin_id": row.origin_id,
        "origin_index": origin, "segment": row.segment, "sample_reason": row.reason,
        "history_clean_index": [origin - history_len, origin - 1],
        "target_clean_index": [origin, origin + horizon - 1],
        "raw_source_rows": [int(value) for value in lineage_rows.raw_row.tolist()],
        "clean_dates": [str(numeric_rows.start_date.iloc[0]), str(numeric_rows.start_date.iloc[-1])],
        "history_first_last_ot": [float(numeric_rows.OT.iloc[0]), float(numeric_rows.OT.iloc[history_len - 1])],
        "target_raw": target_raw.astype(float).tolist(),
        "target_standardized": target_std.astype(float).tolist(),
        "selected_report_ids": text_trace["report_ids"],
        "selected_search_ids": text_trace["search_ids"],
        "text_available": text_available,
        "semantic_l2": float(np.linalg.norm(semantic)),
        "semantic_nonzero": int(np.count_nonzero(semantic)),
        "quality": quality.astype(float).tolist(),
        "frequency": frequency.astype(float).tolist(),
        "publication_time_status": text_trace["publication_time_status"],
        "source_clean_csv_sha256": digest(audit / f"{domain}_numerical.csv"),
        "sample_file_sha256": digest(task / "samples.csv"),
        "text_trace_file_sha256": digest(task / scenario / "text_trace.jsonl"),
    }


def main() -> None:
    bundle, audit = latest_preview(), latest_audit()
    agriculture = pd.read_csv(bundle / "Agriculture_h12_f1/samples.csv")
    positions = [int(agriculture.index[agriculture.segment.eq(segment)][0])
                 for segment in ("train", "calibration", "decision")]
    rows = [trace(bundle, audit, "Agriculture_h12_f1", position, "proxy") for position in positions]
    social = pd.read_csv(bundle / "SocialGood_h3_f1/samples.csv")
    available = np.load(bundle / "SocialGood_h3_f1/proxy/text_available.npy", allow_pickle=False)
    no_text = np.flatnonzero(social.segment.eq("train").to_numpy() & ~available)
    if not len(no_text):
        raise ValueError("Pre-registered SocialGood train has no no-text origin")
    rows.append(trace(bundle, audit, "SocialGood_h3_f1", int(no_text[0]), "proxy"))
    if rows[-1]["text_available"] or rows[-1]["semantic_nonzero"] != 0:
        raise ValueError("No-text control has a nonzero text signal")
    output = HERE / "trace_examples.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                              for row in rows), encoding="utf-8")
    write_json(HERE / "trace_summary.json", {
        "status": "PASS", "examples": len(rows), "agriculture_origins": len(positions),
        "socialgood_no_text_controls": 1, "trace_sha256": digest(output),
        "bundle_kind": "engineering_preview",
        "limitation": "Traces verify lineage and feature construction; they are not formal training evidence.",
    })
    print(json.dumps({"status": "PASS", "output": str(output), "examples": len(rows)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
