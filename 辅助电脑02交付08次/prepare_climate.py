"""Build the Climate-only, strict-isolation Release from frozen local inputs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


BASE = "a06dfe1f2c34c604c6d2b7e374aac44e1783dde9"
BUNDLE = "a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38"
SPEC_SHA = "a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031"
TASK = "Climate_h4_f2"
SAFE_SHARED = ("origin_index.npy", "target_time.npy", "numeric_missing.npy")
SAFE_SCENARIO = ("semantic.npy", "quality.npy", "source_available.npy", "text_available.npy")
FIT_ONLY = ("numeric_history.npy", "numeric_history_raw.npy", "frequency.npy",
            "targets.npy", "targets_raw.npy", "targets_standardized.npy")
SCENARIOS = ("proxy", "conservative_lag")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def npy_digest(array: np.ndarray) -> str:
    stream = io.BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def build(args: argparse.Namespace) -> None:
    bundle = args.bundle.resolve(strict=True)
    task_dir = bundle / TASK
    release = args.release.resolve()
    if release.exists():
        raise ValueError(f"Release stage already exists: {release}")
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["bundle_signature"] != BUNDLE or digest(args.spec) != SPEC_SHA:
        raise ValueError("frozen Bundle or split spec anchor mismatch")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    task_spec = next(item for item in spec["tasks"] if item["domain"] == "Climate")
    if (task_spec["horizon"], task_spec["input_len"], task_spec["bounds"]) != (4, 52, [636, 763, 1017, 1144]):
        raise ValueError("Climate frozen task fields changed")
    snapshot = args.numerical.resolve(strict=True)
    if digest(snapshot) != task_spec["numerical_sha256"]:
        raise ValueError("clean Climate numeric snapshot SHA256 mismatch")
    source_files = manifest["files"]
    for relative, expected in source_files.items():
        if relative.startswith(TASK + "/") and digest(bundle / relative) != expected:
            raise ValueError(f"formal Bundle file SHA256 mismatch: {relative}")
    samples = pd.read_csv(task_dir / "samples.csv", keep_default_na=False)
    sample_audit = pd.read_csv(task_dir / "sample_audit.csv", keep_default_na=False)
    if len(samples) != 1080 or samples.segment.value_counts().to_dict() != {
        "train": 581, "calibration": 124, "decision": 251, "test": 124}:
        raise ValueError("Climate sample identity/count mismatch")
    if not samples.origin_id.is_unique or not samples.origin_index.is_unique:
        raise ValueError("duplicate source origin")
    fit_rows = np.flatnonzero(samples.segment.ne("test").to_numpy())
    test_rows = np.flatnonzero(samples.segment.eq("test").to_numpy())
    release.mkdir(parents=True)
    copy_file(bundle / "manifest.json", release / "anchors/formal_bundle_manifest.json")
    safe_root = release / "source_safe"
    fit_root = release / "selection_fit"
    test_root = release / "test_features"
    fit_root.mkdir()
    test_root.mkdir()
    source_feature_sha: dict[str, str] = {}
    for name in SAFE_SHARED:
        source_name = "horizon_time.npy" if name == "target_time.npy" else name
        source = task_dir / name
        copy_file(source, safe_root / source_name)
        source_feature_sha[source_name] = source_files[f"{TASK}/{name}"]
        whole = np.load(source, allow_pickle=False)
        np.save(fit_root / source_name, whole[fit_rows], allow_pickle=False)
        np.save(test_root / source_name, whole[test_rows], allow_pickle=False)
    for scenario in SCENARIOS:
        for name in SAFE_SCENARIO:
            source = task_dir / scenario / name
            relative = f"{scenario}/{name}"
            copy_file(source, safe_root / relative)
            source_feature_sha[relative] = source_files[f"{TASK}/{relative}"]
            whole = np.load(source, allow_pickle=False)
            (fit_root / scenario).mkdir(parents=True, exist_ok=True)
            (test_root / scenario).mkdir(parents=True, exist_ok=True)
            np.save(fit_root / relative, whole[fit_rows], allow_pickle=False)
            np.save(test_root / relative, whole[test_rows], allow_pickle=False)

    fit_commitments = {}
    for name in FIT_ONLY:
        whole = np.load(task_dir / name, allow_pickle=False)
        selected = whole[fit_rows]
        np.save(fit_root / name, selected, allow_pickle=False)
        fit_commitments[name] = {"full_bundle_file_sha256": source_files[f"{TASK}/{name}"],
                                 "selection_slice_npy_sha256": npy_digest(selected),
                                 "selection_shape": list(selected.shape), "dtype": str(selected.dtype)}
    write_json(release / "selection_fit_commitments.json", fit_commitments)

    # The test truth is read here ONLY to compute a byte commitment. No test truth array is written.
    truth = np.load(task_dir / "targets_raw.npy", allow_pickle=False)
    write_json(release / "truth_commitment.json", {
        "task": TASK, "bundle_signature": BUNDLE, "split_spec_sha256": SPEC_SHA,
        "source_file": f"{TASK}/targets_raw.npy",
        "full_bundle_file_sha256": source_files[f"{TASK}/targets_raw.npy"],
        "test_slice_npy_sha256": npy_digest(truth[test_rows]),
        "test_shape": list(truth[test_rows].shape), "test_dtype": str(truth.dtype),
        "test_origin_count": len(test_rows),
    })
    del truth

    # No per-origin test numerical history or frequency derivative is distributed.
    test_history = np.load(task_dir / "numeric_history_raw.npy", allow_pickle=False)[test_rows]
    valid = test_history[np.isfinite(test_history)]
    write_json(release / "test_history_aggregate.json", {
        "feature": "numeric_history_raw/OT", "unit": "source OT unit", "segment": "test",
        "origin_count": len(test_rows), "history_cells": int(test_history.size),
        "finite_cells": int(valid.size), "missing_cells": int(test_history.size - valid.size),
        "mean": float(valid.mean()), "std": float(valid.std()),
        "q25": float(np.quantile(valid, .25)), "median": float(np.median(valid)),
        "q75": float(np.quantile(valid, .75)),
        "method": "aggregate of historical input cells only; no test target array used",
    })

    columns = ["origin_id", "origin_index", "segment", "cutoff_time", "history_start",
               "segment_lo", "segment_hi"]
    mapping = samples[columns].copy()
    mapping.insert(0, "source_bundle_row", np.arange(len(samples)))
    mapping["package_row"] = -1
    mapping.loc[fit_rows, "package_row"] = np.arange(len(fit_rows))
    mapping.loc[test_rows, "package_row"] = np.arange(len(test_rows))
    mapping["package_partition"] = np.where(mapping.segment.eq("test"), "test_features", "selection_fit")
    mapping["horizon_end_index"] = mapping.origin_index + 4
    mapping.to_csv(release / "mapping.csv", index=False)
    mapping.iloc[fit_rows].drop(columns="source_bundle_row").to_csv(fit_root / "metadata.csv", index=False)
    mapping.iloc[test_rows].drop(columns="source_bundle_row").to_csv(test_root / "metadata.csv", index=False)
    sample_audit.to_csv(release / "sample_audit.csv", index=False)

    numeric = pd.read_csv(snapshot, usecols=["start_date", "end_date"])
    if len(numeric) != 1272:
        raise ValueError("numeric date axis row count mismatch")
    numeric.insert(0, "numerical_row", np.arange(len(numeric)))
    numeric.to_csv(release / "weekly_axis.csv", index=False)
    lineage = pd.read_csv(args.lineage, keep_default_na=False)
    slim_columns = ["raw_row", "domain", "source", "start_date", "end_date", "text_id",
                    "canonical_text_id", "reason", "source_file", "source_url", "publication_verified"]
    lineage.loc[lineage.domain.eq("Climate"), slim_columns].to_csv(release / "source_lineage_slim.csv", index=False)
    for scenario in SCENARIOS:
        copy_file(task_dir / scenario / "text_trace.jsonl", release / "text_trace" / f"{scenario}.jsonl")

    write_json(release / "source_anchors.json", {
        "base_commit": BASE, "task": TASK, "bundle_signature": BUNDLE,
        "split_spec_sha256": SPEC_SHA, "numerical_snapshot_sha256": digest(snapshot),
        "formal_bundle_manifest_sha256": digest(bundle / "manifest.json"),
        "source_safe_full_file_sha256": source_feature_sha,
        "text_trace_full_file_sha256": {scenario: source_files[f"{TASK}/{scenario}/text_trace.jsonl"]
                                        for scenario in SCENARIOS},
        "selection_fit_rows": len(fit_rows), "test_rows": len(test_rows),
        "test_policy": "strict_safe_subset; per-origin numeric history and frequency withheld",
    })
    delivery = args.delivery.resolve(strict=True)
    for source, target in (("verify_climate.py", "code/verify_climate.py"),
                           ("tests/run_negative_cases.py", "tests/run_negative_cases.py"),
                           ("requirements-replay.txt", "requirements-replay.txt"),
                           ("RUNBOOK.md", "RUNBOOK.md")):
        copy_file(delivery / source, release / target)
    print(json.dumps({"status": "BUILT", "release": str(release), "fit": len(fit_rows),
                      "test": len(test_rows), "truth_files_written": 0}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--numerical", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--delivery", type=Path, required=True)
    build(parser.parse_args())
