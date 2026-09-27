"""Build the SocialGood strict-isolation input Release from frozen inputs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


TASK = "SocialGood_h3_f1"
BUNDLE_SHA = "a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38"
SPEC_SHA = "a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031"
SCENARIOS = ("proxy", "conservative_lag")
SAFE_SHARED = ("origin_index.npy", "target_time.npy", "numeric_missing.npy")
SAFE_TEXT = ("semantic.npy", "quality.npy", "source_available.npy", "text_available.npy")
FIT_ONLY = ("numeric_history.npy", "numeric_history_raw.npy", "frequency.npy",
            "targets.npy", "targets_raw.npy", "targets_standardized.npy")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slice_sha(array: np.ndarray) -> str:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def put_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def build(bundle: Path, spec_path: Path, numeric: Path, lineage_path: Path,
          release: Path, delivery: Path) -> dict:
    bundle, spec_path, numeric, lineage_path = (p.resolve(strict=True) for p in
                                                (bundle, spec_path, numeric, lineage_path))
    if release.exists():
        raise ValueError(f"release stage already exists: {release}")
    formal_manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if formal_manifest["bundle_signature"] != BUNDLE_SHA or sha(spec_path) != SPEC_SHA:
        raise ValueError("frozen Bundle signature or spec SHA256 mismatch")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    task_spec = next(item for item in spec["tasks"] if item["domain"] == "SocialGood")
    if (task_spec["input_len"], task_spec["horizon"], task_spec["bounds"],
        spec["seasonal_periods"]["SocialGood"]) != (24, 3, [366, 458, 641, 732], 12):
        raise ValueError("SocialGood task parameters are not frozen values")
    if sha(numeric) != task_spec["numerical_sha256"]:
        raise ValueError("numeric snapshot SHA256 mismatch")
    files = formal_manifest["files"]
    task = bundle / TASK
    for relative, expected in files.items():
        if relative.startswith(f"{TASK}/") and sha(bundle / relative) != expected:
            raise ValueError(f"formal Bundle file changed: {relative}")
    samples = pd.read_csv(task / "samples.csv", keep_default_na=False)
    sample_audit = pd.read_csv(task / "sample_audit.csv", keep_default_na=False)
    bounds = [0, *task_spec["bounds"]]
    calculated = {}
    for segment, lo, hi in zip(("train", "calibration", "decision", "test"), bounds[:-1], bounds[1:]):
        # The frozen protocol keeps origins with complete history and targets in a half-open segment.
        calculated[segment] = len(range(max(lo, task_spec["input_len"]), hi - task_spec["horizon"] + 1))
    observed = samples.segment.value_counts().to_dict()
    if observed != calculated or len(samples) != sum(calculated.values()):
        raise ValueError(f"retained origin count disagrees with computed boundaries: {observed} != {calculated}")
    if not samples.origin_id.is_unique or not samples.origin_index.is_unique:
        raise ValueError("duplicate Bundle origin")
    fit = np.flatnonzero(samples.segment.ne("test").to_numpy())
    test = np.flatnonzero(samples.segment.eq("test").to_numpy())
    release.mkdir(parents=True)
    (release / "selection_fit").mkdir()
    (release / "test_features").mkdir()
    copy(bundle / "manifest.json", release / "anchors/formal_bundle_manifest.json")
    copy(spec_path, release / "anchors/split_spec_v2.json")
    anchors = {}
    for name in SAFE_SHARED:
        external = "horizon_time.npy" if name == "target_time.npy" else name
        source = task / name
        copy(source, release / "source_safe" / external)
        anchors[external] = files[f"{TASK}/{name}"]
        whole = np.load(source, allow_pickle=False)
        np.save(release / "selection_fit" / external, whole[fit], allow_pickle=False)
        np.save(release / "test_features" / external, whole[test], allow_pickle=False)
    for scenario in SCENARIOS:
        for name in SAFE_TEXT:
            relative = f"{scenario}/{name}"
            source = task / relative
            copy(source, release / "source_safe" / relative)
            anchors[relative] = files[f"{TASK}/{relative}"]
            for partition, rows in (("selection_fit", fit), ("test_features", test)):
                target = release / partition / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                np.save(target, np.load(source, allow_pickle=False)[rows], allow_pickle=False)
        copy(task / scenario / "text_trace.jsonl", release / "text_trace" / f"{scenario}.jsonl")
    commitments = {}
    for name in FIT_ONLY:
        source = task / name
        selected = np.load(source, allow_pickle=False)[fit]
        np.save(release / "selection_fit" / name, selected, allow_pickle=False)
        commitments[name] = {"formal_full_sha256": files[f"{TASK}/{name}"],
                             "selection_slice_sha256": slice_sha(selected),
                             "shape": list(selected.shape), "dtype": str(selected.dtype)}
    put_json(release / "selection_fit_commitments.json", commitments)
    # Only the digest of test truth is released; its values are never written to this stage.
    truth = np.load(task / "targets_raw.npy", allow_pickle=False)[test]
    put_json(release / "truth_commitment.json", {
        "task": TASK, "bundle_signature": BUNDLE_SHA, "split_spec_sha256": SPEC_SHA,
        "source_file": f"{TASK}/targets_raw.npy", "formal_full_sha256": files[f"{TASK}/targets_raw.npy"],
        "test_slice_sha256": slice_sha(truth), "test_shape": list(truth.shape),
        "test_dtype": str(truth.dtype), "test_origins": len(test)})
    del truth
    mapping = samples[["origin_id", "origin_index", "segment", "cutoff_time", "history_start",
                       "target_end_index", "segment_lo", "segment_hi"]].copy()
    mapping.insert(0, "source_bundle_row", np.arange(len(mapping)))
    mapping["package_partition"] = np.where(mapping.segment.eq("test"), "test_features", "selection_fit")
    mapping["package_row"] = -1
    mapping.loc[fit, "package_row"] = np.arange(len(fit))
    mapping.loc[test, "package_row"] = np.arange(len(test))
    mapping.to_csv(release / "mapping.csv", index=False)
    mapping.iloc[fit].drop(columns="source_bundle_row").to_csv(release / "selection_fit/metadata.csv", index=False)
    mapping.iloc[test].drop(columns="source_bundle_row").to_csv(release / "test_features/metadata.csv", index=False)
    sample_audit.to_csv(release / "sample_audit.csv", index=False)
    axis = pd.read_csv(numeric, usecols=["start_date", "end_date"])
    if len(axis) != task_spec["n_rows_expected"]:
        raise ValueError("numeric time-axis row count mismatch")
    axis.insert(0, "numerical_row", np.arange(len(axis)))
    axis.to_csv(release / "monthly_axis.csv", index=False)
    lineage = pd.read_csv(lineage_path, keep_default_na=False)
    columns = ["raw_row", "domain", "source", "start_date", "end_date", "text_id",
               "canonical_text_id", "reason", "source_file", "source_url", "publication_verified"]
    lineage.loc[lineage.domain.eq("SocialGood"), columns].to_csv(release / "source_lineage_slim.csv", index=False)
    put_json(release / "source_anchors.json", {
        "task": TASK, "bundle_signature": BUNDLE_SHA, "split_spec_sha256": SPEC_SHA,
        "formal_bundle_manifest_sha256": sha(bundle / "manifest.json"),
        "numeric_snapshot_sha256": sha(numeric), "lineage_sha256": sha(lineage_path),
        "source_safe_sha256": anchors,
        "text_trace_sha256": {s: files[f"{TASK}/{s}/text_trace.jsonl"] for s in SCENARIOS},
        "computed_retained_by_segment": calculated,
        "selection_origins": len(fit), "test_origins": len(test),
        "test_policy": "safe subset; test numeric history and frequency withheld"})
    for source, target in (("verify_socialgood.py", "code/verify_socialgood.py"),
                           ("tests/run_negative_cases.py", "tests/run_negative_cases.py"),
                           ("requirements-replay.txt", "requirements-replay.txt"),
                           ("README.md", "README.md"),
                           ("RUNBOOK.md", "RUNBOOK.md")):
        copy(delivery / source, release / target)
    return {"status": "BUILT", "computed_retained_by_segment": calculated,
            "selection_origins": len(fit), "test_origins": len(test), "test_truth_files_written": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for option in ("bundle", "spec", "numeric", "lineage", "release", "delivery"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.bundle, args.spec, args.numeric, args.lineage,
                           args.release, args.delivery), ensure_ascii=False))
