"""Strict data-side reader for a master-frozen round-three Bundle."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify(folder: Path, spec_path: Path, expected_signature: str) -> dict:
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("signature") == expected_signature, "Bundle signature mismatch")
    require(manifest.get("bundle_signature") == expected_signature, "Bundle signature aliases disagree")
    inputs = manifest.get("inputs", {})
    require(inputs.get("mode") == "frozen", "Bundle is not frozen")
    require(inputs.get("split_spec_sha256") == digest(spec_path), "split spec byte hash mismatch")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    require(inputs.get("split_spec") == spec, "embedded split spec differs")
    require(spec.get("status") == "frozen" and bool(spec.get("approved_by")), "split spec is not frozen")
    recorded = manifest.get("files")
    require(isinstance(recorded, dict) and recorded, "manifest file inventory missing")
    actual = {path.relative_to(folder).as_posix() for path in folder.rglob("*")
              if path.is_file() and path.name != "manifest.json"}
    require(actual == set(recorded), "Bundle inventory differs from manifest")
    for relative, expected in recorded.items():
        require(digest(folder / relative) == expected, f"Bundle hash mismatch: {relative}")
    return manifest


def read_task(folder: Path, spec_path: Path, expected_signature: str,
              task_id: str, scenario: str) -> dict:
    require(scenario in ("proxy", "conservative_lag"), "complete_source is audit-only")
    manifest = verify(folder, spec_path, expected_signature)
    schema = json.loads((folder / "schema.json").read_text(encoding="utf-8"))
    require(schema.get("status") == "frozen", "schema status must be exact lowercase frozen")
    task = folder / task_id
    with (task / "samples.csv").open(encoding="utf-8-sig", newline="") as stream:
        samples = list(csv.DictReader(stream))
    names = ("origin_index", "numeric_history", "targets", "targets_raw", "targets_standardized")
    arrays = {name: np.load(task / f"{name}.npy", allow_pickle=False) for name in names}
    arrays.update({name: np.load(task / scenario / f"{name}.npy", allow_pickle=False)
                   for name in ("semantic", "quality", "text_available")})
    count = len(samples)
    require(all(len(value) == count for value in arrays.values()), "sample/array row counts differ")
    require((task / "targets.npy").read_bytes() == (task / "targets_raw.npy").read_bytes(),
            "targets.npy and targets_raw.npy are not byte-identical")
    origins = arrays["origin_index"].astype(int)
    require(origins.tolist() == [int(row["origin_index"]) for row in samples], "origin order differs")
    fit = json.loads((task / "numeric_fit.json").read_text(encoding="utf-8"))
    expected = (arrays["targets"] - float(fit["mean"])) / float(fit["std"])
    require(np.allclose(expected, arrays["targets_standardized"], atol=1e-6, rtol=1e-6),
            "target standardization differs")
    return {"samples": samples, "arrays": arrays, "manifest": manifest,
            "task_id": task_id, "scenario": scenario, "signature": expected_signature}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("split_spec", type=Path)
    parser.add_argument("signature")
    parser.add_argument("task_id")
    parser.add_argument("--scenario", choices=("proxy", "conservative_lag"), default="proxy")
    args = parser.parse_args()
    result = read_task(args.bundle, args.split_spec, args.signature, args.task_id, args.scenario)
    print(json.dumps({"status": "PASS", "task_id": args.task_id, "scenario": args.scenario,
                      "rows": len(result["samples"]), "signature": result["signature"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
