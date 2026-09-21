"""Read-only bridge from the frozen data Bundle and step CSV to aligned arrays.

This module never chooses a route or reads a held-out test target during freeze.
It deliberately rejects the first-round v2 smoke prediction schema.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .core import EvidenceError


SCALE = "train_only_standardized_OT"
SEGMENTS = ("train", "calibration", "decision", "test")
PREDICTION_COLUMNS = frozenset({
    "task_id", "fold_id", "origin_id", "origin_index", "segment", "scenario",
    "candidate_id", "seed", "step", "y_pred", "target_scale",
    "bundle_signature", "config_sha256", "code_commit",
})


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def signature(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"expected JSON object: {path}")
    return value


def verify_bundle(folder: Path, expected_signature: str, spec_path: Path) -> dict:
    """Verify the data-side seal and master split, including every recorded byte."""
    folder = Path(folder)
    spec = _read_json(Path(spec_path))
    if spec.get("status") != "frozen" or not spec.get("approved_by"):
        raise EvidenceError("split spec is DRAFT; formal Bundle cannot be accepted")
    manifest = _read_json(folder / "manifest.json")
    if manifest.get("signature") != expected_signature or signature(manifest.get("inputs")) != expected_signature:
        raise EvidenceError("Bundle signature mismatch")
    inputs = manifest["inputs"]
    if inputs.get("mode") != "frozen" or inputs.get("split_spec_sha256") != digest(Path(spec_path)):
        raise EvidenceError("Bundle was built from another split spec or preview mode")
    if inputs.get("split_spec") != spec:
        raise EvidenceError("embedded Bundle split spec differs from master")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise EvidenceError("Bundle file inventory missing")
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p.name != "manifest.json"}
    if actual != set(files):
        raise EvidenceError(f"Bundle inventory mismatch: missing={sorted(set(files)-actual)}, extra={sorted(actual-set(files))}")
    for relative, expected_hash in files.items():
        candidate = (folder / relative).resolve()
        if not candidate.is_relative_to(folder.resolve()) or not re.fullmatch(r"[0-9a-f]{64}", str(expected_hash)):
            raise EvidenceError(f"unsafe Bundle manifest entry: {relative}")
        if digest(candidate) != expected_hash:
            raise EvidenceError(f"Bundle file hash mismatch: {relative}")
    return manifest


def _task_spec(spec: dict, task_id: str) -> dict:
    matching = [t for t in spec["tasks"] if f"{t['domain']}_h{t['horizon']}_f{t['fold_id']}" == task_id]
    if len(matching) != 1:
        raise EvidenceError(f"unregistered or duplicate task: {task_id}")
    task = matching[0]
    if not re.fullmatch(r"[0-9a-f]{64}", str(task.get("numerical_sha256"))):
        raise EvidenceError(f"unfrozen numerical snapshot: {task_id}")
    if not (0 < int(task["bounds"][0]) < int(task["bounds"][1]) <
            int(task["bounds"][2]) < int(task["bounds"][3]) <= int(task["n_rows_expected"])):
        raise EvidenceError(f"invalid frozen row boundaries: {task_id}")
    return task


def load_bundle(folder: Path, expected_signature: str, spec_path: Path,
                task_id: str, scenario: str) -> dict:
    """Return fully aligned samples and arrays; no implicit join is performed."""
    if scenario not in ("proxy", "conservative_lag"):
        raise EvidenceError("complete_source is metadata audit only")
    folder = Path(folder)
    manifest = verify_bundle(folder, expected_signature, spec_path)
    spec = _read_json(Path(spec_path))
    task = _task_spec(spec, task_id)
    schema = _read_json(folder / "schema.json")
    if schema.get("status") != "frozen" or schema.get("numeric_features") != ["OT"]:
        raise EvidenceError("wrong Bundle schema or non-frozen status")
    task_dir = folder / task_id
    with (task_dir / "samples.csv").open(encoding="utf-8-sig", newline="") as stream:
        samples = list(csv.DictReader(stream))
    if not samples:
        raise EvidenceError("empty Bundle samples")
    names = ("origin_index", "numeric_history", "targets", "targets_standardized")
    arrays = {name: np.load(task_dir / f"{name}.npy", allow_pickle=False) for name in names}
    arrays["text_available"] = np.load(task_dir / scenario / "text_available.npy", allow_pickle=False)
    n, h, length = len(samples), int(task["horizon"]), int(task["input_len"])
    if arrays["origin_index"].shape != (n,) or arrays["numeric_history"].shape != (n, length):
        raise EvidenceError("Bundle origin/history shape mismatch")
    if arrays["targets"].shape != (n, h) or arrays["targets_standardized"].shape != (n, h):
        raise EvidenceError("Bundle target shape mismatch")
    if arrays["text_available"].shape != (n,):
        raise EvidenceError("Bundle text mask shape mismatch")
    if not all(np.isfinite(arrays[k]).all() for k in ("numeric_history", "targets", "targets_standardized")):
        raise EvidenceError("Bundle has nonfinite model values")
    fit = _read_json(task_dir / "numeric_fit.json")
    mean, std = float(fit["mean"]), float(fit["std"])
    if int(fit["train_end"]) != int(task["bounds"][0]) or not math.isfinite(std) or std <= 0:
        raise EvidenceError("numeric fit is not anchored to the frozen train segment")
    if not np.allclose((arrays["targets"] - mean) / std, arrays["targets_standardized"], rtol=1e-6, atol=1e-6):
        raise EvidenceError("raw and standardized targets disagree")
    bounds = [0] + list(task["bounds"])
    spans = dict(zip(SEGMENTS, zip(bounds[:-1], bounds[1:])))
    seen, previous_segment, previous_origin = set(), -1, -1
    for i, row in enumerate(samples):
        if row.get("segment") not in spans:
            raise EvidenceError(f"sample {i}: unknown segment")
        seg_no = SEGMENTS.index(row["segment"])
        origin = int(row["origin_index"])
        expected_id = f"{task['domain']}:h{h}:f{task['fold_id']}:o{origin}"
        lo, hi = spans[row["segment"]]
        if row.get("origin_id") != expected_id or int(arrays["origin_index"][i]) != origin:
            raise EvidenceError(f"sample {i}: origin identity or array order mismatch")
        if not (lo <= origin and origin + h <= hi):
            raise EvidenceError(f"sample {i}: target crosses {row['segment']} boundary")
        if (seg_no, origin) <= (previous_segment, previous_origin) or expected_id in seen:
            raise EvidenceError("Bundle sample order or identity duplicated")
        seen.add(expected_id)
        previous_segment, previous_origin = seg_no, origin
    return {"task_id": task_id, "scenario": scenario, "signature": expected_signature,
            "manifest_sha256": digest(folder / "manifest.json"), "spec_sha256": digest(Path(spec_path)),
            "task": task, "samples": samples, "arrays": arrays, "manifest": manifest}


def load_prediction_grid(paths: list[Path], bundle: dict, segments: tuple[str, ...]) -> dict:
    """Reject every missing, extra, duplicate or reordered origin-step prediction."""
    if not paths or not segments or any(s not in SEGMENTS for s in segments):
        raise EvidenceError("prediction paths and valid segments are required")
    expected = [(r["segment"], r["origin_id"], int(r["origin_index"]))
                for r in bundle["samples"] if r["segment"] in segments]
    if not expected:
        raise EvidenceError("Bundle contains no requested segment samples")
    task = bundle["task"]
    h = int(task["horizon"])
    grids = defaultdict(list)
    values = defaultdict(dict)
    configs = {}
    for path in map(Path, paths):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or not PREDICTION_COLUMNS <= set(reader.fieldnames):
                raise EvidenceError(f"prediction CSV lacks v2 columns: {path}")
            for line, row in enumerate(reader, 2):
                label = f"{path}:{line}"
                try:
                    origin = int(row["origin_index"])
                    seed = int(row["seed"])
                    step = int(row["step"])
                    pred = float(row["y_pred"])
                except (TypeError, ValueError) as exc:
                    raise EvidenceError(f"{label}: malformed numeric field") from exc
                if row["task_id"] != bundle["task_id"] or int(row["fold_id"]) != int(task["fold_id"]):
                    raise EvidenceError(f"{label}: wrong task or fold")
                if row["scenario"] != bundle["scenario"] or row["bundle_signature"] != bundle["signature"]:
                    raise EvidenceError(f"{label}: scenario or Bundle signature mismatch")
                if row["target_scale"] != SCALE or not math.isfinite(pred) or not 1 <= step <= h:
                    raise EvidenceError(f"{label}: wrong target scale, step or nonfinite forecast")
                if not re.fullmatch(r"[0-9a-f]{64}", row["config_sha256"]) or not re.fullmatch(r"[0-9a-f]{40}", row["code_commit"]):
                    raise EvidenceError(f"{label}: missing model config or code anchor")
                candidate = row["candidate_id"]
                if not candidate or not row["origin_id"]:
                    raise EvidenceError(f"{label}: missing candidate or origin")
                group = (candidate, seed)
                configs.setdefault(group, row["config_sha256"])
                if configs[group] != row["config_sha256"]:
                    raise EvidenceError(f"{label}: mixed model config within candidate/seed")
                key = (row["segment"], row["origin_id"], origin, step)
                if key in values[group]:
                    raise EvidenceError(f"{label}: duplicate prediction key")
                values[group][key] = pred
                grids[group].append(key)
    expected_grid = [(segment, origin_id, origin, step)
                     for segment, origin_id, origin in expected for step in range(1, h + 1)]
    for group, grid in grids.items():
        if grid != expected_grid:
            missing = set(expected_grid) - set(grid)
            extra = set(grid) - set(expected_grid)
            raise EvidenceError(f"prediction grid misaligned for {group}: missing={len(missing)}, extra={len(extra)}, order_bad={not missing and not extra}")
    return {"values": dict(values), "configs": configs, "file_sha256": {str(p): digest(Path(p)) for p in paths},
            "segments": list(segments), "origins": len(expected), "horizon": h}


def numeric_baselines(bundle: dict, alpha_grid: list[float], seasonal_period: int) -> dict:
    """Fit OT-only baselines using training targets and calibration for Ridge alpha.

    The returned forecasts have the Bundle's row order and standardized target
    scale. No decision or test target is consulted for fitting or selection.
    """
    samples, arrays = bundle["samples"], bundle["arrays"]
    x = np.asarray(arrays["numeric_history"], dtype=float)
    y = np.asarray(arrays["targets_standardized"], dtype=float)
    train = np.array([r["segment"] == "train" for r in samples])
    cal = np.array([r["segment"] == "calibration" for r in samples])
    if train.sum() < 2 or not cal.any() or seasonal_period < 1 or seasonal_period > x.shape[1]:
        raise EvidenceError("numeric baseline lacks train/cal rows or usable seasonal history")
    if not alpha_grid or any(not math.isfinite(float(a)) or float(a) <= 0 for a in alpha_grid):
        raise EvidenceError("Ridge alpha grid must contain positive finite values")
    last = np.repeat(x[:, -1:], y.shape[1], axis=1)
    seasonal = np.stack([x[:, -seasonal_period + (step % seasonal_period)]
                         for step in range(y.shape[1])], axis=1)
    x_mean, y_mean = x[train].mean(axis=0), y[train].mean(axis=0)
    xc, yc = x[train] - x_mean, y[train] - y_mean
    gram, cross = xc.T @ xc, xc.T @ yc
    candidates = {}
    for alpha in sorted(set(map(float, alpha_grid))):
        weights = np.linalg.solve(gram + alpha * np.eye(x.shape[1]), cross)
        prediction = (x - x_mean) @ weights + y_mean
        loss = float(np.mean((prediction[cal] - y[cal]) ** 2))
        candidates[alpha] = (loss, prediction)
    chosen_alpha = min(candidates, key=lambda a: (candidates[a][0], a))
    predictions = {"Last": last, "SeasonalNaive": seasonal,
                   "AR-Ridge": candidates[chosen_alpha][1]}
    for candidate, prediction in predictions.items():
        if prediction.shape != y.shape or not np.isfinite(prediction).all():
            raise EvidenceError(f"invalid numeric baseline forecast: {candidate}")
    return {"predictions": predictions, "ridge_alpha": chosen_alpha,
            "ridge_calibration_mse": candidates[chosen_alpha][0],
            "fit_rows": int(train.sum()), "calibration_rows": int(cal.sum()),
            "alpha_grid": sorted(candidates), "target_scale": SCALE}


def write_baseline_predictions(path: Path, bundle: dict, baselines: dict,
                               scenario: str, code_commit: str, seed: int = 2026,
                               segments: tuple[str, ...] = ("calibration", "decision")) -> int:
    """Export numeric baselines in the same step CSV contract as model predictions."""
    if scenario != bundle["scenario"] or not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise EvidenceError("baseline scenario or code commit is not anchored")
    if not segments or any(s not in ("calibration", "decision", "test") for s in segments):
        raise EvidenceError("baseline export requires explicit non-training segments")
    fields = ["task_id", "fold_id", "origin_id", "origin_index", "segment", "scenario",
              "candidate_id", "seed", "step", "y_pred", "target_scale", "bundle_signature",
              "config_sha256", "code_commit"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for candidate, matrix in baselines["predictions"].items():
            config = {"method": candidate, "bundle_signature": bundle["signature"],
                      "alpha_grid": baselines["alpha_grid"] if candidate == "AR-Ridge" else None,
                      "ridge_alpha": baselines["ridge_alpha"] if candidate == "AR-Ridge" else None}
            config_hash = signature(config)
            for i, sample in enumerate(bundle["samples"]):
                if sample["segment"] not in segments:
                    continue
                for j, value in enumerate(matrix[i], 1):
                    writer.writerow(dict(task_id=bundle["task_id"], fold_id=bundle["task"]["fold_id"],
                        origin_id=sample["origin_id"], origin_index=sample["origin_index"],
                        segment=sample["segment"], scenario=scenario, candidate_id=candidate,
                        seed=seed, step=j, y_pred=float(value), target_scale=SCALE,
                        bundle_signature=bundle["signature"], config_sha256=config_hash,
                        code_commit=code_commit))
                    count += 1
    return count


def write_stage_inputs(folder: Path, bundle: dict, grid: dict, split_spec: dict,
                       segments: tuple[str, ...]) -> dict:
    """Materialize a stage-specific v1 evaluator package from aligned v2 evidence.

    The selection package contains only calibration and decision truth. Run the
    freeze process against that package in a directory without test artifacts.
    """
    allowed = (("calibration", "decision"), ("test",))
    if segments not in allowed or grid["segments"] != list(segments):
        raise EvidenceError("only separate selection or test stage exports are allowed")
    numeric = split_spec["numeric_fallback_candidates"]
    gates = split_spec["gate_candidates"]
    present = {candidate for candidate, _seed in grid["values"]}
    if not set(numeric + gates) <= present:
        raise EvidenceError(f"required registered candidates missing: {sorted(set(numeric + gates) - present)}")
    task = bundle["task"]
    bounds = list(task["bounds"])
    task_record = dict(task_id=bundle["task_id"], fold_id=task["fold_id"],
        domain=task["domain"], horizon=task["horizon"], input_len=task["input_len"],
        n_rows=task["n_rows_expected"], train_end=bounds[0], cal_end=bounds[1],
        dec_end=bounds[2], test_end=bounds[3],
        seasonal_period=split_spec["seasonal_periods"][task["domain"]],
        snapshot_sha256=task["numerical_sha256"],
        feature_manifest_sha256=bundle["signature"])
    evaluator_spec = dict(protocol_version="team-eval-2.0-bridge", status="FROZEN",
        original_split_spec_sha256=bundle["spec_sha256"],
        numeric_candidates=numeric, candidate_variants=gates,
        selection=dict(row_null_draws=split_spec["row_permutations"],
                       p_threshold=split_spec["p_threshold"]),
        uncertainty=dict(draws=5000, seed=split_spec["seed"]))
    segment_name = {"calibration": "cal", "decision": "dec", "test": "test"}
    selected = [(i, row) for i, row in enumerate(bundle["samples"]) if row["segment"] in segments]
    if len(selected) != grid["origins"]:
        raise EvidenceError("prediction and Bundle origin counts differ")
    sample_rows, prediction_rows = [], []
    h = int(task["horizon"])
    for i, sample in selected:
        origin = int(sample["origin_index"])
        segment = segment_name[sample["segment"]]
        sample_rows.append(dict(task_id=bundle["task_id"], fold_id=task["fold_id"],
            origin_id=origin, original_origin_id=sample["origin_id"], segment=segment,
            horizon=h, target=bundle["arrays"]["targets_standardized"][i].astype(float).tolist(),
            cutoff_index=origin - 1, target_start_index=origin,
            snapshot_sha256=task["numerical_sha256"],
            feature_manifest_sha256=bundle["signature"], preprocessing_fit_end=bounds[0]))
        for (candidate, seed), values in sorted(grid["values"].items()):
            forecast = [values[(sample["segment"], sample["origin_id"], origin, step)]
                        for step in range(1, h + 1)]
            prediction_rows.append(dict(task_id=bundle["task_id"], fold_id=task["fold_id"],
                origin_id=origin, original_origin_id=sample["origin_id"], segment=segment,
                candidate_id=candidate, seed=seed, prediction=forecast,
                config_sha256=grid["configs"][(candidate, seed)],
                feature_manifest_sha256=bundle["signature"]))
    folder = Path(folder)
    if folder.exists() and any(folder.iterdir()):
        raise EvidenceError("stage output must be a fresh empty folder")
    folder.mkdir(parents=True, exist_ok=True)
    for name, value in (("task.json", task_record), ("spec.json", evaluator_spec)):
        (folder / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("samples.jsonl", sample_rows), ("predictions.jsonl", prediction_rows)):
        (folder / name).write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                           for row in rows), encoding="utf-8")
    manifest = dict(status="READY_FOR_FREEZE" if segments[0] == "calibration" else "READY_FOR_SCORE",
        segments=list(segments), task_id=bundle["task_id"], bundle_signature=bundle["signature"],
        bundle_manifest_sha256=bundle["manifest_sha256"],
        split_spec_sha256=bundle["spec_sha256"], prediction_files_sha256=grid["file_sha256"],
        sample_rows=len(sample_rows), prediction_rows=len(prediction_rows),
        files_sha256={p.name: digest(p) for p in folder.iterdir() if p.is_file()})
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
