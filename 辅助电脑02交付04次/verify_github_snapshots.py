from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DELIVERY = "辅助电脑02交付04次"
DOMAINS = ("Agriculture", "Climate", "SocialGood", "Environment")
SNAPSHOT_PATH = f"{DELIVERY}/clean_snapshots/{{domain}}_numerical.csv"
ROUND3_MANIFEST = "辅助电脑02交付03次/clean_snapshot_manifest.csv"
ROUND2_REGISTRY = "辅助电脑02交付02次/evidence/task_registry.csv"


def git_bytes(spec: str) -> bytes:
    return subprocess.check_output(["git", "cat-file", "blob", spec])


def git_text(spec: str) -> str:
    return git_bytes(spec).decode("utf-8-sig")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def csv_records(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def csv_shape(data: bytes) -> tuple[int, list[str]]:
    reader = csv.reader(io.StringIO(data.decode("utf-8-sig")))
    rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty")
    return len(rows) - 1, rows[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the four clean snapshots from immutable Git blobs."
    )
    parser.add_argument("--commit", required=True, help="Fixed snapshot commit")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    commit = subprocess.check_output(
        ["git", "rev-parse", f"{args.commit}^{{commit}}"], text=True
    ).strip()
    round3 = {
        row["domain"]: row
        for row in csv_records(git_text(f"{commit}:{ROUND3_MANIFEST}"))
    }
    round2 = {
        row["domain"]: row
        for row in csv_records(git_text(f"{commit}:{ROUND2_REGISTRY}"))
    }

    files: list[dict[str, object]] = []
    for domain in DOMAINS:
        path = SNAPSHOT_PATH.format(domain=domain)
        data = git_bytes(f"{commit}:{path}")
        rows, columns = csv_shape(data)
        digest = sha256(data)
        git_oid = subprocess.check_output(
            ["git", "rev-parse", f"{commit}:{path}"], text=True
        ).strip()
        checks = {
            "sha256_matches_round3": digest == round3[domain]["sha256"],
            "bytes_matches_round3": len(data) == int(round3[domain]["bytes"]),
            "rows_matches_round3": rows == int(round3[domain]["rows"]),
            "columns_match_round3": "|".join(columns) == round3[domain]["columns"],
            "sha256_matches_round2": digest == round2[domain]["numerical_sha256"],
            "rows_match_round2": rows == int(round2[domain]["n_rows"]),
        }
        files.append(
            {
                "domain": domain,
                "git_path": path,
                "git_blob_oid": git_oid,
                "sha256": digest,
                "bytes": len(data),
                "rows": rows,
                "columns": columns,
                "checks": checks,
                "status": "PASS" if all(checks.values()) else "FAIL",
            }
        )

    total_bytes = sum(int(item["bytes"]) for item in files)
    passed = all(item["status"] == "PASS" for item in files)
    receipt = {
        "schema_version": 1,
        "status": "COMPLETE" if passed else "FAILED",
        "scope": "stage1_github_blob_snapshot_only",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_snapshot_commit": commit,
        "branch": "3218151885-creator",
        "comparison_sources": {
            "round3_manifest": ROUND3_MANIFEST,
            "round2_task_registry": ROUND2_REGISTRY,
        },
        "files": files,
        "file_count": len(files),
        "total_bytes": total_bytes,
        "commands": {
            "extract_one": "git show <fixed_snapshot_commit>:<git_path> > <destination>",
            "verify": (
                ".venv/Scripts/python.exe "
                f'"{DELIVERY}/verify_github_snapshots.py" '
                "--commit <fixed_snapshot_commit> "
                f'--output "{DELIVERY}/github_snapshot_receipt.json"'
            ),
        },
        "claim_boundary": (
            "Confirms bytes stored in the fixed Git commit and agreement with prior "
            "registries. It does not confirm master download, protocol freeze, formal "
            "Bundle construction, or cross-reader acceptance."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": receipt["status"], "commit": commit,
                      "files": len(files), "bytes": total_bytes}, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
