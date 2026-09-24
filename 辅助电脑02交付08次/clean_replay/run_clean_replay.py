"""Replay a downloaded Climate Release ZIP in a previously nonexistent D: directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("clean_directory", type=Path)
    parser.add_argument("log_directory", type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve(strict=True)
    clean = args.clean_directory.resolve()
    logs = args.log_directory.resolve()
    if clean.exists() or archive.is_relative_to(clean) or logs.is_relative_to(clean):
        parser.error("clean directory must be new; archive/logs must be outside it")
    logs.mkdir(parents=True, exist_ok=True)
    release = clean / "release"
    release.mkdir(parents=True)
    with ZipFile(archive) as packed:
        for name in packed.namelist():
            member = PurePosixPath(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe archive path: {name}")
        packed.extractall(release)
    commands = [
        ("verify", [sys.executable, str(release / "code/verify_climate.py"), "--package", str(release),
                    "--out", str(clean / "output")]),
        ("negative_cases", [sys.executable, str(release / "tests/run_negative_cases.py"),
                            "--package", str(release), "--out", str(clean / "negative_cases.json")]),
    ]
    runs = []
    for label, command in commands:
        started = time.perf_counter()
        process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed = round(time.perf_counter() - started, 3)
        (logs / f"{label}.stdout.txt").write_text(process.stdout, encoding="utf-8")
        (logs / f"{label}.stderr.txt").write_text(process.stderr, encoding="utf-8")
        runs.append({"step": label, "command": " ".join(command), "exit_code": process.returncode,
                     "elapsed_seconds": elapsed, "stdout": f"{label}.stdout.txt", "stderr": f"{label}.stderr.txt"})
        if process.returncode:
            break
    expected = json.loads((release / "EXPECTED_OUTPUTS.json").read_text(encoding="utf-8"))["files"]
    output = clean / "output"
    actual = {relative: digest(output / relative) for relative in expected if (output / relative).is_file()}
    negatives = clean / "negative_cases.json"
    negative_status = json.loads(negatives.read_text(encoding="utf-8"))["status"] if negatives.exists() else "MISSING"
    status = "PASS" if len(runs) == 2 and all(row["exit_code"] == 0 for row in runs) \
        and actual == expected and negative_status == "PASS" else "FAIL"
    record = {"status": status, "archive_name": archive.name, "archive_bytes": archive.stat().st_size,
              "archive_sha256": digest(archive), "clean_directory": str(clean), "python": sys.version.split()[0],
              "runs": runs, "total_elapsed_seconds": round(sum(row["elapsed_seconds"] for row in runs), 3),
              "output_sha256": actual, "expected_output_sha256": expected,
              "negative_cases_status": negative_status, "negative_cases_sha256": digest(negatives) if negatives.exists() else None}
    (logs / "clean_replay_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "archive_sha256": record["archive_sha256"],
                      "outputs": len(actual), "negative_cases": negative_status}, ensure_ascii=False))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
