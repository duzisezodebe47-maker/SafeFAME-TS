"""Independent byte and content audit before approving a numerical split spec."""
from __future__ import annotations

import csv
import json
import math
from datetime import date
from pathlib import Path

from .core import EvidenceError, sha256


BOUNDS = ((0.4, 0.5, 0.7, 0.8), (0.5, 0.6, 0.8, 0.9), (0.6, 0.7, 0.9, 1.0))


def _rows(path: Path) -> list[dict]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise EvidenceError(f"invalid CSV header: {path}")
            return list(reader)
    except OSError as exc:
        raise EvidenceError(f"cannot read CSV: {path}") from exc


def audit_one(task: dict, declared: dict, snapshot: Path) -> dict:
    """Inspect actual bytes; matching a self-reported hash is only one check."""
    domain = task["domain"]
    task_id = f"{domain}_h{task['horizon']}_f{task['fold_id']}"
    if declared.get("task_id") != task_id or declared.get("domain") != domain:
        raise EvidenceError(f"{domain}: wrong registry identity")
    try:
        digest = sha256(snapshot)
    except OSError as exc:
        raise EvidenceError(f"{domain}: clean CSV missing or unreadable: {snapshot}") from exc
    if digest != declared.get("numerical_sha256"):
        raise EvidenceError(f"{domain}: actual CSV SHA256 differs from registry")
    rows = _rows(snapshot)
    if len(rows) != int(task["n_rows_expected"]) or len(rows) != int(declared["n_rows"]):
        raise EvidenceError(f"{domain}: cleaned row count differs from both declarations")
    if not rows or set(rows[0]) != {"start_date", "end_date", "OT"}:
        raise EvidenceError(f"{domain}: expected exactly start_date,end_date,OT")
    previous = None
    for number, row in enumerate(rows, 2):
        try:
            start, end = date.fromisoformat(row["start_date"]), date.fromisoformat(row["end_date"])
            if row["start_date"] != start.isoformat() or row["end_date"] != end.isoformat():
                raise ValueError("noncanonical date")
            value = float(row["OT"])
        except (TypeError, ValueError) as exc:
            raise EvidenceError(f"{domain}: invalid date or OT on CSV line {number}") from exc
        if end < start or (previous is not None and start <= previous) or not math.isfinite(value):
            raise EvidenceError(f"{domain}: duplicate/unsorted interval or unknown OT on line {number}")
        previous = start
    fold = int(task["fold_id"])
    if fold not in (1, 2, 3):
        raise EvidenceError(f"{domain}: invalid fold")
    bounds = [int(len(rows) * p) for p in BOUNDS[fold - 1]]
    if bounds != task["bounds"] or bounds != [int(declared[k]) for k in
             ("train_end", "calibration_end", "decision_end", "test_end")]:
        raise EvidenceError(f"{domain}: fold boundaries do not match actual row count")
    for key in ("fold_id", "input_len", "horizon", "lag_days"):
        if int(declared[key]) != int(task[key]):
            raise EvidenceError(f"{domain}: registry/spec {key} mismatch")
    if int(declared["unknown_target_rows"]) != 0:
        raise EvidenceError(f"{domain}: registry reports unknown OT")
    return {"task_id": task_id, "path": str(snapshot.resolve()), "sha256": digest,
            "rows": len(rows), "first_start": rows[0]["start_date"],
            "last_end": rows[-1]["end_date"], "bounds": bounds,
            "ot_finite_rows": len(rows), "status": "BYTE_AND_CONTENT_CHECK_PASSED"}


def audit_snapshots(spec_path: Path, registry_path: Path, snapshot_dir: Path) -> dict:
    spec_path, registry_path, snapshot_dir = map(Path, (spec_path, registry_path, snapshot_dir))
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvidenceError("invalid draft split spec") from exc
    if spec.get("status") != "draft_not_for_training" or len(spec.get("tasks", [])) != 4:
        raise EvidenceError("expected four-task draft split spec")
    registry = _rows(registry_path)
    by_id = {row["task_id"]: row for row in registry}
    if len(registry) != 4 or len(by_id) != 4:
        raise EvidenceError("registry must contain exactly four distinct tasks")
    expected_ids = {f"{t['domain']}_h{t['horizon']}_f{t['fold_id']}" for t in spec["tasks"]}
    if set(by_id) != expected_ids:
        raise EvidenceError("registry task set differs from draft split spec")
    checks = []
    for task in spec["tasks"]:
        declared = by_id[f"{task['domain']}_h{task['horizon']}_f{task['fold_id']}"]
        path = snapshot_dir / f"{task['domain']}_numerical.csv"
        checks.append(audit_one(task, declared, path))
    return {"status": "READY_FOR_HUMAN_FREEZE_REVIEW", "split_spec_sha256": sha256(spec_path),
            "registry_sha256": sha256(registry_path), "snapshots": checks,
            "limitation": "This audits supplied clean bytes and registry consistency; it does not independently re-run cleaning from raw Time-MMD sources."}
