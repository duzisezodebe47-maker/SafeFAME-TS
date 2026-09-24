"""Unpack a Release in a new directory and record an independent replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("clean_dir", type=Path)
    parser.add_argument("log_dir", type=Path)
    args = parser.parse_args()
    archive = args.archive.resolve(strict=True)
    clean_dir = args.clean_dir.resolve()
    log_dir = args.log_dir.resolve()
    if clean_dir.exists():
        parser.error(f"clean replay directory already exists: {clean_dir}")
    if archive.is_relative_to(clean_dir) or log_dir.is_relative_to(clean_dir):
        parser.error("archive and logs must be outside the clean replay directory")
    log_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True)
    with ZipFile(archive) as packed:
        for name in packed.namelist():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe ZIP entry: {name}")
        packed.extractall(clean_dir)

    checks = [
        ("replay", [sys.executable, "code/replay.py", "--output-dir", "replay_output"]),
        ("negative_cases", [sys.executable, "tests/run_negative_cases.py", "--output", "replay_output/negative_cases_v7.json"]),
    ]
    runs = []
    for name, command in checks:
        started = time.perf_counter()
        result = subprocess.run(command, cwd=clean_dir, capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed = round(time.perf_counter() - started, 3)
        (log_dir / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (log_dir / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
        runs.append({"name": name, "command": " ".join(command), "exit_code": result.returncode, "elapsed_seconds": elapsed,
                     "stdout_file": f"{name}.stdout.txt", "stderr_file": f"{name}.stderr.txt"})
        if result.returncode:
            break

    output_dir = clean_dir / "replay_output"
    expected = json.loads((clean_dir / "EXPECTED_OUTPUTS.json").read_text(encoding="utf-8"))["files"]
    actual = {name: sha256(output_dir / name) for name in expected if (output_dir / name).is_file()}
    status = "PASS" if len(runs) == 2 and all(run["exit_code"] == 0 for run in runs) and actual == expected else "FAIL"
    record = {"status": status, "archive_name": archive.name, "archive_bytes": archive.stat().st_size,
              "archive_sha256": sha256(archive), "clean_dir": str(clean_dir), "python": sys.version.split()[0],
              "runs": runs, "output_sha256": actual, "expected_output_sha256": expected,
              "total_elapsed_seconds": round(sum(run["elapsed_seconds"] for run in runs), 3)}
    (log_dir / "clean_replay_record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "archive_sha256": record["archive_sha256"], "runs": runs}, ensure_ascii=False))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
