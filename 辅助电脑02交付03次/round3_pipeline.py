"""Third-round data handoff: snapshot transfer and frozen Bundle promotion."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FIRST = ROOT / "辅助电脑02交付01次"
SECOND = ROOT / "辅助电脑02交付02次"
sys.path[:0] = [str(SECOND), str(FIRST)]

from audit import digest, require, signature, write_json  # noqa: E402
from round2_pipeline import seal_local, validate_frozen_spec, verify_local  # noqa: E402

MASTER_REF = "origin/team/main-eval-20260921"
MASTER_SPEC = "team_work/main/round2/split_spec_v2.json"
DOMAINS = ("Agriculture", "Climate", "SocialGood", "Environment")
SCENARIOS = ("proxy", "conservative_lag")
SHARED = ROOT / "data_processed/team_transfer/辅助电脑02交付03次"


def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def current_master_spec() -> tuple[dict, bytes, str, str]:
    raw = git_bytes("show", f"{MASTER_REF}:{MASTER_SPEC}")
    return json.loads(raw), raw, sha_bytes(raw), git_bytes("rev-parse", MASTER_REF).decode().strip()


def latest_audit() -> Path:
    candidates = sorted((ROOT / "data_processed/team_data").glob("audit-*"),
                        key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for candidate in candidates:
        manifest = candidate / "manifest.json"
        if not manifest.is_file():
            continue
        value = json.loads(manifest.read_text(encoding="utf-8"))
        actual = {p.relative_to(candidate).as_posix(): digest(p) for p in candidate.rglob("*")
                  if p.is_file() and p.name != "manifest.json"}
        if actual == value.get("files"):
            return candidate
    raise ValueError("No byte-verified audit folder is available")


def latest_preview() -> Path:
    result = json.loads((SECOND / "evidence/preview_result.json").read_text(encoding="utf-8"))
    folder = Path(result["bundle_path"])
    verify_local(folder, result["bundle_signature"])
    return folder


def source_records(audit: Path) -> list[dict]:
    audit_manifest = json.loads((audit / "manifest.json").read_text(encoding="utf-8"))
    transfer = SHARED / "clean_snapshots"
    transfer.mkdir(parents=True, exist_ok=True)
    records = []
    for domain in DOMAINS:
        source = audit / f"{domain}_numerical.csv"
        destination = transfer / source.name
        if destination.exists():
            require(digest(destination) == digest(source),
                    f"Transferred snapshot changed; refusing overwrite: {destination}")
        else:
            shutil.copyfile(source, destination)
        require(digest(destination) == digest(source), f"Snapshot copy hash mismatch: {domain}")
        raw = f"references/external/Time-MMD/numerical/{domain}/{domain}.csv"
        clean_input = f"data_processed/v4/numerical/{domain}/{domain}.csv"
        rows = len(pd.read_csv(source))
        records.append({
            "domain": domain,
            "original_source_relative": raw,
            "original_source_sha256": audit_manifest["inputs"]["raw_and_cache"][raw],
            "cleaning_input_relative": clean_input,
            "cleaning_input_sha256": audit_manifest["inputs"]["raw_and_cache"][clean_input],
            "audit_output_relative": source.relative_to(ROOT).as_posix(),
            "transfer_relative": destination.relative_to(ROOT).as_posix(),
            "sha256": digest(source),
            "bytes": source.stat().st_size,
            "rows": rows,
            "columns": "|".join(pd.read_csv(source, nrows=0).columns),
            "generation_command": '.venv/Scripts/python.exe "辅助电脑02交付01次/run.py" audit',
            "copy_verified": "true",
        })
    return records


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"Refusing empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize_bundle(bundle: Path) -> tuple[list[dict], list[dict]]:
    coverage, audits = [], []
    for task_dir in sorted(path for path in bundle.iterdir() if path.is_dir()):
        samples = pd.read_csv(task_dir / "samples.csv")
        sample_audit = pd.read_csv(task_dir / "sample_audit.csv")
        raw = np.load(task_dir / "targets_raw.npy", allow_pickle=False)
        standardized = np.load(task_dir / "targets_standardized.npy", allow_pickle=False)
        fit = json.loads((task_dir / "numeric_fit.json").read_text(encoding="utf-8"))
        expected = (raw - float(fit["mean"])) / float(fit["std"])
        max_error = float(np.max(np.abs(expected - standardized)))
        for segment in ("train", "calibration", "decision", "test"):
            mask = samples.segment.eq(segment).to_numpy()
            segment_audit = sample_audit[sample_audit.segment.eq(segment)]
            reason_counts = segment_audit.reason.value_counts().to_dict()
            audits.extend({"task_id": task_dir.name, "segment": segment, "reason": reason,
                           "origins": int(count)} for reason, count in sorted(reason_counts.items()))
            for scenario in SCENARIOS:
                traces = [json.loads(line) for line in
                          (task_dir / scenario / "text_trace.jsonl").read_text(encoding="utf-8").splitlines()]
                selected = {text_id for index in np.flatnonzero(mask)
                            for source in ("report_ids", "search_ids") for text_id in traces[index][source]}
                text_available = np.load(task_dir / scenario / "text_available.npy", allow_pickle=False)
                coverage.append({
                    "task_id": task_dir.name, "scenario": scenario, "segment": segment,
                    "candidate_origins": int(len(segment_audit)), "retained_origins": int(mask.sum()),
                    "excluded_origins": int((segment_audit.reason != "retained").sum()),
                    "target_crossing_excluded": int(sum(v for k, v in reason_counts.items() if "crosses" in k)),
                    "nonfinite_target_excluded": int(reason_counts.get("nonfinite_target", 0)),
                    "text_available_origins": int(text_available[mask].sum()),
                    "unique_selected_texts": int(len(selected)),
                    "target_raw_mean": float(raw[mask].mean()), "target_raw_std": float(raw[mask].std()),
                    "target_standardized_mean": float(standardized[mask].mean()),
                    "target_standardized_std": float(standardized[mask].std()),
                    "standardization_max_abs_error": max_error,
                })
            coverage.append({
                "task_id": task_dir.name, "scenario": "complete_source_audit_only", "segment": segment,
                "candidate_origins": int(len(segment_audit)), "retained_origins": int(mask.sum()),
                "excluded_origins": int((segment_audit.reason != "retained").sum()),
                "target_crossing_excluded": int(sum(v for k, v in reason_counts.items() if "crosses" in k)),
                "nonfinite_target_excluded": int(reason_counts.get("nonfinite_target", 0)),
                "text_available_origins": 0, "unique_selected_texts": 0,
                "target_raw_mean": float(raw[mask].mean()), "target_raw_std": float(raw[mask].std()),
                "target_standardized_mean": float(standardized[mask].mean()),
                "target_standardized_std": float(standardized[mask].std()),
                "standardization_max_abs_error": max_error,
            })
    return coverage, audits


def prepare() -> dict:
    audit = latest_audit()
    preview = latest_preview()
    snapshots = source_records(audit)
    write_csv(HERE / "clean_snapshot_manifest.csv", snapshots)
    coverage, audits = summarize_bundle(preview)
    write_csv(HERE / "coverage.csv", coverage)
    write_csv(HERE / "sample_audit_summary.csv", audits)
    spec, raw, spec_sha, master_commit = current_master_spec()
    (HERE / "formal_split_spec.sha256").write_text(
        f"{spec_sha}  split_spec_v2.json\n"
        f"# observed_status={spec.get('status')}; approved_by={spec.get('approved_by')}; "
        "DATA_FORMAL_PENDING until master publishes status=frozen.\n", encoding="utf-8")
    pending = {
        "status": "DATA_FORMAL_PENDING", "formal_bundle": None,
        "reason": "Master split_spec_v2.json is not frozen or approved.",
        "observed_split_spec_status": spec.get("status"), "approved_by": spec.get("approved_by"),
        "observed_split_spec_sha256": spec_sha, "master_commit": master_commit,
        "preview_bundle_signature": json.loads((preview / "manifest.json").read_text(encoding="utf-8"))["signature"],
        "warning": "This file is a truthful pending record, not a formal Bundle manifest.",
    }
    write_json(HERE / "formal_bundle_manifest.json", pending)
    receipt = {
        "status": "DATA_FORMAL_PENDING", "receiver_status": "MASTER_ACK_PENDING",
        "clean_snapshots": {"status": "COPIED_AND_BYTE_VERIFIED_ON_SHARED_D",
                            "target": (SHARED / "clean_snapshots").relative_to(ROOT).as_posix(),
                            "files": len(snapshots), "bytes": sum(int(row["bytes"]) for row in snapshots)},
        "formal_bundle": {"status": "NOT_BUILT_OR_TRANSFERRED_WAITING_FOR_MASTER_FREEZE",
                          "bundle_signature": None, "files": 0, "bytes": 0, "target": None},
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "3218151885-creator"], text=True).strip(),
        "master_commit": master_commit, "split_spec_sha256": spec_sha,
        "python": platform.python_version(), "platform": platform.platform(),
        "dependencies": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    write_json(HERE / "transfer_receipt.json", receipt)
    return {"status": pending["status"], "snapshots": len(snapshots), "shared": str(SHARED),
            "split_spec_sha256": spec_sha}


def validate_spec_file(spec_path: Path, registry: pd.DataFrame,
                       test_only: bool = False) -> tuple[dict, str]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    validate_frozen_spec(spec, registry)
    require(spec.get("status") == "frozen" and bool(spec.get("approved_by")),
            "Master spec must have exact status=frozen and a non-empty approved_by")
    is_fixture = str(spec.get("approved_by", "")).startswith("TEST_FIXTURE_ONLY")
    require(test_only == is_fixture,
            "TEST_FIXTURE_ONLY specs are restricted to tests and cannot produce a formal Bundle")
    require(spec.get("alpha_grid") == [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
            "Frozen spec does not contain the master seven-alpha grid")
    observed_order = [f"{task['domain']}_h{task['horizon']}_f{task['fold_id']}" for task in spec["tasks"]]
    require(observed_order == ["Agriculture_h12_f1", "Climate_h4_f2",
                               "SocialGood_h3_f1", "Environment_h7_f2"],
            "Frozen task order differs from the master protocol")
    if not test_only:
        remote_spec, remote_raw, remote_sha, _remote_commit = current_master_spec()
        require(digest(spec_path) == remote_sha and spec_path.read_bytes() == remote_raw and spec == remote_spec,
                "Formal split spec is not byte-identical to the current master branch blob")
    return spec, digest(spec_path)


def promote(preview: Path, spec_path: Path, registry_path: Path,
            test_only: bool = False) -> tuple[Path, dict]:
    verify_local(preview, json.loads((preview / "manifest.json").read_text(encoding="utf-8"))["signature"])
    registry = pd.read_csv(registry_path)
    spec, spec_sha = validate_spec_file(spec_path, registry, test_only=test_only)
    inputs = {
        "mode": "frozen", "test_fixture_only": test_only,
        "split_spec_sha256": spec_sha, "split_spec": spec,
        "source_preview_manifest_sha256": digest(preview / "manifest.json"),
        "pipeline_sha256": digest(Path(__file__)),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "3218151885-creator"], text=True).strip(),
        "master_commit": git_bytes("rev-parse", MASTER_REF).decode().strip(),
    }
    bundle_signature = signature(inputs)
    prefix = "round3-test-fixture-" if test_only else "round3-bundle-"
    output = ROOT / "data_processed/team_data" / f"{prefix}{bundle_signature[:20]}"
    if output.exists():
        return output, verify_local(output, bundle_signature)
    parent = output.parent
    temporary = Path(tempfile.mkdtemp(prefix="round3-building-", dir=parent))
    try:
        shutil.copytree(preview, temporary, dirs_exist_ok=True)
        (temporary / "manifest.json").unlink()
        for task in spec["tasks"]:
            task_id = f"{task['domain']}_h{task['horizon']}_f{task['fold_id']}"
            task_dir = temporary / task_id
            require(digest(latest_audit() / f"{task['domain']}_numerical.csv") == task["numerical_sha256"],
                    f"{task_id}: source snapshot differs from frozen spec")
            shutil.copyfile(task_dir / "targets_raw.npy", task_dir / "targets.npy")
            require((task_dir / "targets.npy").read_bytes() == (task_dir / "targets_raw.npy").read_bytes(),
                    f"{task_id}: targets aliases are not byte-identical")
        schema = json.loads((temporary / "schema.json").read_text(encoding="utf-8"))
        schema["status"] = "frozen"
        schema["numeric_features"] = ["OT"]
        schema["split_spec_sha256"] = spec_sha
        write_json(temporary / "schema.json", schema)
        rows = [json.loads(line) for line in (temporary / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in rows:
            row["bundle_signature"] = bundle_signature
        (temporary / "samples.jsonl").write_text("".join(
            json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
        coverage, audits = summarize_bundle(temporary)
        write_csv(temporary / "coverage.csv", coverage)
        write_csv(temporary / "sample_audit_summary.csv", audits)
        manifest = seal_local(temporary, inputs, bundle_signature)
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "build", "verify"))
    parser.add_argument("--split-spec", type=Path)
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare()
    elif args.command == "build":
        require(args.split_spec is not None, "build requires --split-spec from master")
        bundle, manifest = promote(latest_preview(), args.split_spec,
                                   SECOND / "evidence/task_registry.csv")
        write_json(HERE / "formal_bundle_manifest.json", manifest)
        result = {"status": "FROZEN", "bundle": str(bundle), "signature": manifest["signature"]}
    else:
        require(args.bundle is not None, "verify requires --bundle")
        manifest = json.loads((args.bundle / "manifest.json").read_text(encoding="utf-8"))
        verify_local(args.bundle, manifest["signature"])
        result = {"status": "PASS", "bundle": str(args.bundle), "signature": manifest["signature"]}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
