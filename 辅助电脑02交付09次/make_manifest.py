"""Hash every Release member and build a ZIP without placing it in Git."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(root: Path, reference: Path | None, archive: Path | None) -> dict:
    root = root.resolve(strict=True)
    if reference:
        outputs = json.loads((reference / "replay_report.json").read_text(encoding="utf-8"))["output_sha256"]
        (root / "EXPECTED_OUTPUTS.json").write_text(json.dumps({"files": outputs}, ensure_ascii=False,
                                                       indent=2) + "\n", encoding="utf-8")
    anchors = json.loads((root / "source_anchors.json").read_text(encoding="utf-8"))
    files = sorted((p for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST.json"
                    and "__pycache__" not in p.parts and p.suffix != ".pyc"),
                   key=lambda p: p.relative_to(root).as_posix())
    entries = [{"relative_path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size,
                "sha256": sha(p)} for p in files]
    manifest = {"schema_version": 1, "task": anchors["task"],
                "bundle_signature": anchors["bundle_signature"],
                "split_spec_sha256": anchors["split_spec_sha256"],
                "self_exclusion": "MANIFEST.json is excluded from its own file list.", "files": entries}
    (root / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if archive:
        if archive.exists():
            raise ValueError(f"ZIP already exists: {archive}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as output:
            for relative in sorted([entry["relative_path"] for entry in entries] + ["MANIFEST.json"]):
                output.write(root / relative, relative)
    return {"files": len(entries), "manifest_sha256": sha(root / "MANIFEST.json"),
            "zip_bytes": archive.stat().st_size if archive else None,
            "zip_sha256": sha(archive) if archive else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--zip", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.release, args.reference, args.zip), ensure_ascii=False))
