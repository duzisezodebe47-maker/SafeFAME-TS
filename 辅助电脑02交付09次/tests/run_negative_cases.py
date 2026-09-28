"""Disposable mutations must each be rejected by the actual verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


sys.dont_write_bytecode = True


def reanchor(root: Path, relative: str) -> None:
    file = root / relative
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(e for e in manifest["files"] if e["relative_path"] == relative)
    entry["bytes"] = file.stat().st_size
    entry["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test(package: Path, output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    cases = (
        ("test_truth_injection", "test_features/targets_raw.npy", "test directory whitelist violation"),
        ("test_numeric_history_injection", "test_features/numeric_history.npy", "test directory whitelist violation"),
        ("test_frequency_injection", "test_features/frequency.npy", "test directory whitelist violation"),
        ("selection_slice_tamper", "selection_fit/targets_raw.npy", "selection slice commitment mismatch"),
        ("mapping_wrong_row", "mapping.csv", "selection_fit row mapping mismatch"),
        ("bundle_signature_wrong", "source_anchors.json", "frozen Bundle signature or spec anchor mismatch"),
        ("raw_audit_tamper", "raw_clean_audit.json", "raw-to-clean numeric count/missing/duplicate audit mismatch"),
        ("frozen_spec_wrong", "anchors/split_spec_v2.json", "frozen spec SHA256 mismatch"),
    )
    results = []
    for name, relative, expected in cases:
        with tempfile.TemporaryDirectory(prefix="socialgood-negative-", dir=output.parent) as temporary:
            root = Path(temporary) / "package"
            shutil.copytree(package, root)
            file = root / relative
            if name.endswith("injection"):
                source_name = "targets_raw.npy" if "truth" in name else (
                    "numeric_history.npy" if "history" in name else "frequency.npy")
                shutil.copyfile(root / "selection_fit" / source_name, file)
            elif name == "selection_slice_tamper":
                array = np.load(file, allow_pickle=False)
                array[0, 0] += 1.0
                np.save(file, array, allow_pickle=False)
                reanchor(root, relative)
            elif name == "mapping_wrong_row":
                frame = pd.read_csv(file)
                frame.loc[0, "package_row"] = 1
                frame.to_csv(file, index=False)
                reanchor(root, relative)
            elif name == "bundle_signature_wrong":
                payload = json.loads(file.read_text(encoding="utf-8"))
                payload["bundle_signature"] = "0" * 64
                file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                reanchor(root, relative)
            elif name == "raw_audit_tamper":
                payload = json.loads(file.read_text(encoding="utf-8"))
                payload["numeric"]["removed_missing_OT_rows"] += 1
                file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                reanchor(root, relative)
                anchors = root / "source_anchors.json"
                payload = json.loads(anchors.read_text(encoding="utf-8"))
                payload["raw_clean_audit_sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
                anchors.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                reanchor(root, "source_anchors.json")
            else:
                payload = json.loads(file.read_text(encoding="utf-8"))
                payload["seed"] += 1
                file.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                reanchor(root, relative)
            command = [sys.executable, str(root / "code/verify_socialgood.py"),
                       "--package", str(root), "--out", str(Path(temporary) / "output")]
            result = subprocess.run(command, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace")
            results.append({"case": name, "exit_code": result.returncode,
                            "expected_reason": expected, "observed_stderr": result.stderr.strip(),
                            "pass": result.returncode != 0 and expected in result.stderr})
    summary = {"status": "PASS" if all(item["pass"] for item in results) else "FAIL",
               "passed": sum(item["pass"] for item in results), "total": len(results), "cases": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("status", "passed", "total")}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if test(args.package.resolve(strict=True), args.out)["status"] == "PASS" else 2)
