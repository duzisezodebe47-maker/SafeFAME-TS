"""Seal a verified cal/decision route with external byte provenance, never test data."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .core import EvidenceError, freeze_route, jsonl, sha256
from .null_bridge import verify_selection_stage


def _object(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise EvidenceError(f"expected JSON object: {path}")
    return value


def seal_route(stage_dir: Path, null_path: Path, route_path: Path,
               frozen_spec_path: Path, review_path: Path, out: Path) -> dict:
    """Recompute route and input hashes, then write immutable route plus SHA sidecar.

    The review file is a separate, explicit human/code-audit gate for refit
    provenance. Numerical CSV agreement alone cannot establish actual refits.
    """
    stage_dir, null_path, route_path = map(Path, (stage_dir, null_path, route_path))
    frozen_spec_path, review_path, out = map(Path, (frozen_spec_path, review_path, out))
    sidecar = out.with_name(out.name + ".sha256")
    if out.exists() or sidecar.exists():
        raise EvidenceError("sealed route output exists; refusing overwrite")
    expected_names = {"manifest.json", "task.json", "spec.json", "samples.jsonl", "predictions.jsonl"}
    actual_names = {p.name for p in stage_dir.iterdir()}
    if actual_names != expected_names:
        raise EvidenceError(f"selection directory has extra/missing files: {sorted(actual_names ^ expected_names)}")
    stage, task, spec, samples, predictions = verify_selection_stage(stage_dir)
    frozen = _object(frozen_spec_path)
    if frozen.get("status") != "frozen" or not frozen.get("approved_by"):
        raise EvidenceError("master split spec is not frozen")
    frozen_sha = sha256(frozen_spec_path)
    if frozen_sha != stage["split_spec_sha256"]:
        raise EvidenceError("selection stage was built from another frozen split spec")
    route = _object(route_path)
    expected_hashes = {"task": sha256(stage_dir / "task.json"),
                       "spec": sha256(stage_dir / "spec.json"),
                       "samples": sha256(stage_dir / "samples.jsonl"),
                       "predictions": sha256(stage_dir / "predictions.jsonl"),
                       "nulls": sha256(null_path)}
    if route.get("inputs_sha256") != expected_hashes:
        raise EvidenceError("route input hashes differ from stage/null evidence")
    nulls = list(jsonl(null_path))
    if len(nulls) != 2 or {x.get("candidate_id") for x in nulls} != set(spec["candidate_variants"]):
        raise EvidenceError("route null evidence lacks exactly two registered candidates")
    code_commits = sorted({x.get("model_code_commit") for x in nulls})
    if any(not isinstance(x, str) or not re.fullmatch(r"[0-9a-f]{40}", x) for x in code_commits):
        raise EvidenceError("null conversion lacks valid model code anchors")
    review = _object(review_path)
    if (review.get("status") != "REFIT_PROVENANCE_VERIFIED" or
            review.get("null_jsonl_sha256") != expected_hashes["nulls"] or
            review.get("model_code_commits") != code_commits or
            not review.get("reviewer")):
        raise EvidenceError("independent refit provenance review is missing or mismatched")
    recomputed = freeze_route(task, samples, predictions, nulls, spec)
    for key, value in recomputed.items():
        if route.get(key) != value:
            raise EvidenceError(f"route disagrees with independent cal/decision computation: {key}")
    if set(route) != set(recomputed) | {"inputs_sha256"}:
        raise EvidenceError("raw route has unexpected or missing fields")
    sealed = dict(route, status="frozen", scenario=stage["scenario"],
                  bundle_signature=stage["bundle_signature"],
                  split_spec_sha256=frozen_sha,
                  selection_manifest_sha256=sha256(stage_dir / "manifest.json"),
                  refit_review_sha256=sha256(review_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(sealed, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    digest = sha256(out)
    with sidecar.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{digest}  {out.name}\n")
    return {"status": "SEALED", "route": str(out), "sha256": digest,
            "selection_manifest_sha256": sealed["selection_manifest_sha256"],
            "refit_review_sha256": sealed["refit_review_sha256"]}
