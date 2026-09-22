from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEGMENTS = ("train", "calibration", "decision", "test")
SCENARIOS = ("proxy", "conservative_lag")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def signature(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def segment_spans(bounds: list[int]) -> dict[str, tuple[int, int]]:
    points = [0, *map(int, bounds)]
    return dict(zip(SEGMENTS, zip(points[:-1], points[1:])))


def task_id(task: dict) -> str:
    return f"{task['domain']}_h{task['horizon']}_f{task['fold_id']}"


def inventory(folder: Path) -> dict[str, str]:
    return {
        path.relative_to(folder).as_posix(): digest(path)
        for path in folder.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }


def audit(bundle: Path, spec_path: Path, snapshots: Path) -> tuple[dict, list[dict]]:
    spec_raw = spec_path.read_bytes()
    spec = json.loads(spec_raw)
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    schema = json.loads((bundle / "schema.json").read_text(encoding="utf-8"))
    spec_sha = hashlib.sha256(spec_raw).hexdigest()

    require(spec["status"] == "frozen" and bool(spec["approved_by"]), "spec is not frozen")
    require(manifest["signature"] == signature(manifest["inputs"]), "manifest input signature differs")
    require(manifest.get("bundle_signature") == manifest["signature"], "bundle signature aliases differ")
    require(manifest["inputs"]["mode"] == "frozen", "manifest mode is not frozen")
    require(manifest["inputs"].get("test_fixture_only") is False, "test fixture cannot be formal")
    require(manifest["inputs"]["split_spec_sha256"] == spec_sha, "spec byte hash differs")
    require(manifest["inputs"]["split_spec"] == spec, "embedded spec differs")
    require(schema["status"] == "frozen", "schema status is not lowercase frozen")
    require(schema["split_spec_sha256"] == spec_sha, "schema spec hash differs")
    require(schema["scenarios"] == list(SCENARIOS), "schema scenarios differ")
    require(schema["complete_source"] == "audit only; no prediction arrays", "complete_source rule differs")
    require(inventory(bundle) == manifest["files"], "actual inventory or file SHA256 differs")

    coverage_rows: list[dict] = []
    task_reports: list[dict] = []
    for task in spec["tasks"]:
        name = task_id(task)
        folder = bundle / name
        source = snapshots / f"{task['domain']}_numerical.csv"
        frame = pd.read_csv(source)
        require(digest(source) == task["numerical_sha256"], f"{name}: snapshot hash differs")
        require(len(frame) == int(task["n_rows_expected"]), f"{name}: snapshot rows differ")
        values = frame["OT"].to_numpy(dtype=np.float64)
        require(np.isfinite(values).all(), f"{name}: nonfinite source target")

        samples = pd.read_csv(folder / "samples.csv")
        sample_audit = pd.read_csv(folder / "sample_audit.csv")
        origins = np.load(folder / "origin_index.npy", allow_pickle=False).astype(np.int64)
        history_raw = np.load(folder / "numeric_history_raw.npy", allow_pickle=False)
        history = np.load(folder / "numeric_history.npy", allow_pickle=False)
        targets = np.load(folder / "targets.npy", allow_pickle=False)
        targets_raw = np.load(folder / "targets_raw.npy", allow_pickle=False)
        standardized = np.load(folder / "targets_standardized.npy", allow_pickle=False)
        fit = json.loads((folder / "numeric_fit.json").read_text(encoding="utf-8"))
        horizon, input_len = int(task["horizon"]), int(task["input_len"])
        train_end = int(task["bounds"][0])
        mean, std = values[:train_end].mean(), values[:train_end].std(ddof=0)

        require((folder / "targets.npy").read_bytes() == (folder / "targets_raw.npy").read_bytes(),
                f"{name}: target aliases are not byte-identical")
        require(samples.origin_index.astype(int).tolist() == origins.tolist(), f"{name}: origin order differs")
        require(len(set(origins.tolist())) == len(origins), f"{name}: duplicate origins")
        require(history_raw.shape == (len(origins), input_len), f"{name}: history shape differs")
        require(targets.shape == (len(origins), horizon), f"{name}: target shape differs")
        require(float(fit["mean"]) == float(mean) and float(fit["std"]) == float(std),
                f"{name}: train-only fit differs")
        require(int(fit["train_end"]) == train_end and int(fit["ddof"]) == 0,
                f"{name}: fit metadata differs")

        expected_history = np.stack([values[o - input_len:o] for o in origins])
        expected_targets = np.stack([values[o:o + horizon] for o in origins])
        require(np.array_equal(history_raw, expected_history.astype(history_raw.dtype)),
                f"{name}: raw history differs from snapshot")
        require(np.array_equal(targets_raw, expected_targets.astype(targets_raw.dtype)),
                f"{name}: raw targets differ from snapshot")
        require(np.allclose(history, (expected_history - mean) / std, rtol=1e-6, atol=1e-6),
                f"{name}: history standardization differs")
        require(np.allclose(standardized, (expected_targets - mean) / std, rtol=1e-6, atol=1e-6),
                f"{name}: target standardization differs")

        spans = segment_spans(task["bounds"])
        expected_audit_origins: list[int] = []
        expected_retained: list[int] = []
        for segment, (lo, hi) in spans.items():
            start = max(lo, input_len)
            for origin in range(start, hi):
                expected_audit_origins.append(origin)
                if origin + horizon <= hi and np.isfinite(values[origin:origin + horizon]).all():
                    expected_retained.append(origin)
            selected = samples[samples.segment.eq(segment)].origin_index.astype(int).to_numpy()
            require(np.all((selected >= lo) & (selected + horizon <= hi)),
                    f"{name}/{segment}: crossing target window")
        require(sample_audit.origin_index.astype(int).tolist() == expected_audit_origins,
                f"{name}: sample audit candidate grid differs")
        require(origins.tolist() == expected_retained, f"{name}: retained grid differs")
        retained_from_audit = sample_audit[sample_audit.reason.eq("retained")].origin_index.astype(int).tolist()
        require(retained_from_audit == expected_retained, f"{name}: audit reasons differ")

        for scenario in SCENARIOS:
            scenario_dir = folder / scenario
            semantic = np.load(scenario_dir / "semantic.npy", allow_pickle=False)
            quality = np.load(scenario_dir / "quality.npy", allow_pickle=False)
            available = np.load(scenario_dir / "text_available.npy", allow_pickle=False).astype(bool)
            traces = [json.loads(line) for line in
                      (scenario_dir / "text_trace.jsonl").read_text(encoding="utf-8").splitlines()]
            require(semantic.shape == (len(origins), int(schema["semantic_dim"])),
                    f"{name}/{scenario}: semantic shape differs")
            require(quality.shape == (len(origins), int(schema["quality_dim"])),
                    f"{name}/{scenario}: quality shape differs")
            require(available.shape == (len(origins),), f"{name}/{scenario}: text mask shape differs")
            require(len(traces) == len(origins), f"{name}/{scenario}: trace rows differ")
            for segment in SEGMENTS:
                mask = samples.segment.eq(segment).to_numpy()
                audited = sample_audit[sample_audit.segment.eq(segment)]
                reasons = audited.reason.value_counts().to_dict()
                selected_ids = {
                    text_id
                    for index in np.flatnonzero(mask)
                    for source_key in ("report_ids", "search_ids")
                    for text_id in traces[index][source_key]
                }
                coverage_rows.append({
                    "task_id": name,
                    "scenario": scenario,
                    "segment": segment,
                    "candidate_origins": len(audited),
                    "retained_origins": int(mask.sum()),
                    "excluded_origins": int((audited.reason != "retained").sum()),
                    "target_crossing_excluded": int(sum(v for k, v in reasons.items() if "crosses" in k)),
                    "nonfinite_target_excluded": int(reasons.get("nonfinite_target", 0)),
                    "text_available_origins": int(available[mask].sum()),
                    "unique_selected_texts": len(selected_ids),
                })
        require(not (folder / "complete_source").exists(), f"{name}: complete_source arrays must not exist")
        task_reports.append({
            "task_id": name,
            "snapshot_sha256": digest(source),
            "snapshot_rows": len(frame),
            "retained_origins": len(origins),
            "candidate_origins": len(sample_audit),
            "excluded_origins": int((sample_audit.reason != "retained").sum()),
            "targets_alias_byte_identical": True,
            "standardization_max_abs_error": float(np.max(np.abs(
                standardized - ((expected_targets - mean) / std)))),
            "status": "PASS",
        })

    existing_coverage = pd.read_csv(bundle / "coverage.csv")
    require(set(existing_coverage.scenario) == {*SCENARIOS, "complete_source_audit_only"},
            "coverage scenarios differ")
    require(len(existing_coverage) == len(spec["tasks"]) * len(SEGMENTS) * 3,
            "coverage row count differs")
    report = {
        "status": "PASS",
        "bundle_signature": manifest["signature"],
        "split_spec_sha256": spec_sha,
        "manifest_signature_recomputed": signature(manifest["inputs"]),
        "manifest_sha256": digest(bundle / "manifest.json"),
        "files_excluding_manifest": len(manifest["files"]),
        "bytes_including_manifest": sum(p.stat().st_size for p in bundle.rglob("*") if p.is_file()),
        "tasks": task_reports,
        "checks": [
            "manifest signature(inputs)", "frozen spec bytes and embedded object",
            "exact inventory and every file SHA256", "snapshot hashes and row counts",
            "raw history and targets against source OT", "train-only standardization",
            "origin identity/order and disjoint half-open boundaries",
            "two scenario shapes, text coverage and trace row counts",
            "complete_source audit-only rule",
        ],
    }
    return report, coverage_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--snapshots", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--coverage", required=True, type=Path)
    args = parser.parse_args()
    report, rows = audit(args.bundle, args.spec, args.snapshots)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with args.coverage.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": report["status"], "tasks": len(report["tasks"]),
                      "coverage_rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
