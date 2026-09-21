"""Strict, loss-preserving conversion of model row-null CSVs for route freezing."""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

from .core import EvidenceError, jsonl, mse, sha256, validate


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"expected JSON object: {path}")
    return value


def _close(actual: object, expected: float, label: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (float, int)) or not math.isfinite(actual):
        raise EvidenceError(f"{label}: missing finite number")
    if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-9):
        raise EvidenceError(f"{label}: {actual} differs from independently computed {expected}")


def verify_selection_stage(folder: Path) -> tuple[dict, dict, dict, dict, dict]:
    """Check the selection package has only cal/dec evidence and intact hashes."""
    folder = Path(folder)
    manifest = _read_json(folder / "manifest.json")
    if manifest.get("status") != "READY_FOR_FREEZE" or manifest.get("segments") != ["calibration", "decision"]:
        raise EvidenceError("selection stage is not a frozen-input-ready cal/decision package")
    expected = manifest.get("files_sha256")
    if not isinstance(expected, dict) or set(expected) != {"task.json", "spec.json", "samples.jsonl", "predictions.jsonl"}:
        raise EvidenceError("selection stage inventory incomplete")
    for name, recorded in expected.items():
        if sha256(folder / name) != recorded:
            raise EvidenceError(f"selection stage hash mismatch: {name}")
    task, spec = _read_json(folder / "task.json"), _read_json(folder / "spec.json")
    samples, predictions, _ = validate(task, jsonl(folder / "samples.jsonl"),
                                       jsonl(folder / "predictions.jsonl"), spec=spec)
    if set(k[0] for k in samples) != {"cal", "dec"} or set(k[0] for k in predictions) != {"cal", "dec"}:
        raise EvidenceError("selection package has test/train evidence or missing cal/dec")
    if task["feature_manifest_sha256"] != manifest.get("bundle_signature"):
        raise EvidenceError("selection Bundle signature mismatch")
    return manifest, task, spec, samples, predictions


def _observed_decision_mse(samples: dict, predictions: dict, candidate: str, seed: int) -> float:
    origins = sorted(o for segment, o in samples if segment == "dec")
    if not origins:
        raise EvidenceError("no decision origins")
    matching_seeds = {s for segment, _o, c, s in predictions if segment == "dec" and c == candidate}
    if matching_seeds != {seed}:
        raise EvidenceError(f"{candidate}: expected exactly one prediction seed {seed}, found {sorted(matching_seeds)}")
    truth = [samples[("dec", o)]["target"] for o in origins]
    forecast = [predictions[("dec", o, candidate, seed)] for o in origins]
    return mse(truth, forecast)


def convert_nulls(selection_folder: Path, null_dirs: list[Path], seed: int = 2026) -> tuple[list[dict], dict]:
    stage, task, spec, samples, predictions = verify_selection_stage(selection_folder)
    expected_candidates = list(spec["candidate_variants"])
    draws = int(spec["selection"]["row_null_draws"])
    if draws != 999 or len(null_dirs) != len(expected_candidates):
        raise EvidenceError("formal conversion requires exactly the registered candidates and 999 draws")
    by_candidate = {}
    source_hashes = {}
    for folder in map(Path, null_dirs):
        summary = _read_json(folder / "permutation_summary.json")
        candidate = summary.get("candidate")
        if candidate not in expected_candidates or candidate in by_candidate:
            raise EvidenceError(f"unregistered or duplicate null candidate: {candidate}")
        if (summary.get("task") != task["task_id"] or
                summary.get("scenario") != stage.get("scenario") or
                summary.get("bundle_signature") != stage["bundle_signature"]):
            raise EvidenceError(f"{candidate}: wrong task, scenario or Bundle signature")
        if summary.get("requested_permutations") != draws or summary.get("completed_permutations") != draws or summary.get("failures") != 0:
            raise EvidenceError(f"{candidate}: 999 successful refits not documented")
        if not re.fullmatch(r"[0-9a-f]{40}", str(summary.get("code_commit"))) or not re.fullmatch(r"[0-9a-f]{64}", str(summary.get("config_sha256"))):
            raise EvidenceError(f"{candidate}: missing code/config anchor")
        source = folder / "null_scores.csv"
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["candidate", "segment", "iteration", "seed", "loss"]:
                raise EvidenceError(f"{candidate}: unexpected null CSV columns")
            raw = list(reader)
        if len(raw) != draws:
            raise EvidenceError(f"{candidate}: null CSV has {len(raw)} rows, expected {draws}")
        losses = []
        for iteration, row in enumerate(raw):
            try:
                got_iteration, got_seed, loss = int(row["iteration"]), int(row["seed"]), float(row["loss"])
            except (TypeError, ValueError) as exc:
                raise EvidenceError(f"{candidate}: invalid null row {iteration}") from exc
            if (row["candidate"] != candidate or row["segment"] != "decision" or
                    got_iteration != iteration or got_seed != seed * 1000 + iteration or
                    not math.isfinite(loss) or loss < 0):
                raise EvidenceError(f"{candidate}: shifted/duplicate/nonfinite null row {iteration}")
            losses.append(loss)
        observed = _observed_decision_mse(samples, predictions, candidate, seed)
        _close(summary.get("observed_decision_loss"), observed, f"{candidate} observed MSE")
        expected_p = (1 + sum(value <= observed for value in losses)) / (draws + 1)
        _close(summary.get("p_value"), expected_p, f"{candidate} p value")
        csv_summary = _read_json(folder / "null_scores_summary.json")
        if (csv_summary.get("candidate") != candidate or csv_summary.get("segment") != "decision" or
                csv_summary.get("requested") != draws or csv_summary.get("completed") != draws or
                csv_summary.get("failures") != 0):
            raise EvidenceError(f"{candidate}: CSV summary count or identity mismatch")
        _close(csv_summary.get("observed_loss"), observed, f"{candidate} CSV observed MSE")
        _close(csv_summary.get("p_value"), expected_p, f"{candidate} CSV p value")
        source_hashes[candidate] = {"csv": sha256(source), "csv_summary": sha256(folder / "null_scores_summary.json"),
                                    "run_summary": sha256(folder / "permutation_summary.json")}
        by_candidate[candidate] = dict(task_id=task["task_id"], fold_id=task["fold_id"],
            candidate_id=candidate, decision_null_mse=losses, method="row-permutation-refit",
            seed=seed, source_sha256=source_hashes[candidate]["csv"],
            bundle_signature=stage["bundle_signature"], model_code_commit=summary["code_commit"],
            model_config_sha256=summary["config_sha256"])
    report = {"status": "NUMERICALLY_VERIFIED_REFIT_PROVENANCE_PENDING",
              "task_id": task["task_id"], "bundle_signature": stage["bundle_signature"],
              "selection_manifest_sha256": sha256(Path(selection_folder) / "manifest.json"),
              "source_hashes": source_hashes, "draws_per_candidate": draws,
              "limitation": "The converter verifies identities, row counts, hashes, observed MSE and p; it cannot prove that each CSV loss came from a separate refit without model execution logs and code audit."}
    return [by_candidate[c] for c in expected_candidates], report


def write_conversion(out: Path, rows: list[dict], report: dict) -> None:
    out = Path(out)
    if out.exists() or out.with_suffix(".manifest.json").exists():
        raise EvidenceError("conversion outputs already exist; refusing overwrite")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report = dict(report, converted_jsonl_sha256=sha256(out))
    with out.with_suffix(".manifest.json").open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
