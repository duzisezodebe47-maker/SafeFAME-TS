"""Record byte-anchored, relative-path inventory of a seventh-round Release."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


REVISION = "00281e2d86058286d5548b15a7670e8eda57ef62"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def describe(path: Path, root: Path) -> dict:
    relative = path.relative_to(root).as_posix()
    rows = None
    columns = None
    shape = None
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            columns = next(reader, [])
            rows = sum(1 for _ in reader)
    elif path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            first = f.readline()
            rows = int(bool(first)) + sum(1 for _ in f)
        columns = list(json.loads(first)) if first else []
    elif path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = list(array.shape)
        rows = int(shape[0]) if shape else 1
        columns = [f"axis_{i}:{n}" for i, n in enumerate(shape)]
    elif path.suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        columns = list(obj) if isinstance(obj, dict) else None
        rows = len(obj) if isinstance(obj, list) else 1
    if relative.startswith("inputs/formal_bundle/"):
        generation = "formal Bundle frozen by auxiliary data pipeline; signature a69821be115445265cde..."
    elif relative == "inputs/fact_corpus.csv" or relative == "inputs/text_lineage.csv":
        generation = "辅助电脑02交付01次/audit.py::run_audit (fixed Time-MMD revision)"
    elif relative == "inputs/split_spec_v2.json":
        generation = "master-frozen split_spec_v2.json, copied byte-for-byte"
    elif relative == "EXPECTED_OUTPUTS.json":
        generation = "code/replay.py --build-reference; SHA256 computed from generated output"
    else:
        generation = "seventh-round source code / documentation"
    return {
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "rows": rows,
        "columns": columns,
        "shape": shape,
        "sha256": digest(path),
        "generation_command": generation,
        "upstream_revision": REVISION if relative.startswith("inputs/") else None,
    }


def build(root: Path) -> dict:
    files = [p for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST.json"
             and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    files.sort(key=lambda p: p.relative_to(root).as_posix())
    manifest = {
        "schema_version": 1,
        "status": "frozen_release_inputs",
        "bundle_signature": "a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38",
        "split_spec_sha256": "a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031",
        "upstream_revision": REVISION,
        "self_exclusion": "MANIFEST.json is excluded from its own hash inventory to avoid a circular hash.",
        "files": [describe(path, root) for path in files],
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def package(root: Path, archive: Path, manifest: dict) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as packed:
        for relative in sorted([item["relative_path"] for item in manifest["files"]] + ["MANIFEST.json"]):
            packed.write(root / relative, relative)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("release_root", type=Path)
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    if args.reference_output:
        reference = args.reference_output.resolve()
        report = json.loads((reference / "replay_report.json").read_text(encoding="utf-8"))
        files = {relative: digest(reference / relative) for relative in report["output_sha256"]}
        expected = {"status": "frozen_reference", "files": files}
        (args.release_root / "EXPECTED_OUTPUTS.json").write_text(
            json.dumps(expected, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result = build(args.release_root.resolve())
    if args.zip:
        package(args.release_root.resolve(), args.zip.resolve(), result)
    print(json.dumps({"files": len(result["files"]), "manifest": str(args.release_root / "MANIFEST.json")}, ensure_ascii=False))
