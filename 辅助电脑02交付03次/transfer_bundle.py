"""Copy a frozen Bundle on D: and verify source/destination byte identity."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def inventory(folder: Path) -> dict[str, str]:
    return {path.relative_to(folder).as_posix(): digest(path) for path in folder.rglob("*") if path.is_file()}


def verify_transfer(source: Path, destination: Path) -> tuple[dict[str, str], dict[str, str]]:
    source_files, destination_files = inventory(source), inventory(destination)
    if source_files != destination_files:
        missing = sorted(set(source_files) - set(destination_files))
        extra = sorted(set(destination_files) - set(source_files))
        changed = sorted(name for name in set(source_files) & set(destination_files)
                         if source_files[name] != destination_files[name])
        raise ValueError(f"Destination inventory or SHA256 differs from source: "
                         f"missing={missing[:3]}, extra={extra[:3]}, changed={changed[:3]}")
    return source_files, destination_files


def copy_and_verify(source: Path, destination: Path, receipt: Path,
                    source_commit: str, split_spec_sha256: str) -> dict:
    source, destination = source.resolve(), destination.resolve()
    if source.drive.lower() != "d:" or destination.drive.lower() != "d:":
        raise ValueError("Source and destination must both be on D:; C: staging is forbidden")
    if destination.exists():
        raise ValueError("Destination must be absent; refusing to merge or overwrite")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("inputs", {}).get("mode") != "frozen":
        raise ValueError("Refusing to transfer a non-frozen Bundle")
    shutil.copytree(source, destination)
    source_files, destination_files = verify_transfer(source, destination)
    result = {
        "status": "COPIED_AND_BYTE_VERIFIED_ON_SHARED_D", "receiver_status": "MASTER_ACK_PENDING",
        "source": str(source), "target": str(destination), "source_commit": source_commit,
        "split_spec_sha256": split_spec_sha256, "bundle_signature": manifest["signature"],
        "files": len(source_files),
        "bytes": sum((source / relative).stat().st_size for relative in source_files),
        "source_inventory_sha256": hashlib.sha256(json.dumps(
            source_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "destination_inventory_sha256": hashlib.sha256(json.dumps(
            destination_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "python": platform.python_version(),
        "dependencies": {"numpy": np.__version__, "pandas": pd.__version__},
    }
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--split-spec-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(copy_and_verify(args.source, args.destination, args.receipt,
                                     args.source_commit, args.split_spec_sha256),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
