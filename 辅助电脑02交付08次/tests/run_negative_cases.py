"""Mutate disposable Release copies; every prohibited state must exit nonzero."""

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
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file = root / relative
    for entry in manifest["files"]:
        if entry["relative_path"] == relative:
            entry["bytes"] = file.stat().st_size
            entry["sha256"] = hashlib.sha256(file.read_bytes()).hexdigest()
            break
    else:
        raise ValueError(f"fixture file not in manifest: {relative}")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def test(package: Path, output: Path) -> dict:
    cases = [
        ("test_target_column", "test_features/metadata.csv", "test metadata contains target/truth column"),
        ("test_truth_file", "test_features/truth.npy", "unexpected Release file"),
        ("selection_contains_test_row", "selection_fit/metadata.csv", "selection package contains test row"),
        ("duplicate_origin", "mapping.csv", "duplicate origin"),
        ("origin_horizon_overflow", "mapping.csv", "origin+horizon crosses segment boundary"),
        ("feature_differs_from_bundle", "test_features/proxy/quality.npy", "feature mismatch with formal Bundle"),
        ("test_numeric_history_injection", "test_features/numeric_history.npy", "unexpected Release file"),
    ]
    results = []
    for name, relative, expected in cases:
        with tempfile.TemporaryDirectory(prefix="climate-negative-") as temporary:
            base = Path(temporary)
            fixture = base / "package"
            shutil.copytree(package, fixture)
            file = fixture / relative
            if name == "test_target_column":
                frame = pd.read_csv(file)
                frame["target"] = 0.0
                frame.to_csv(file, index=False)
                reanchor(fixture, relative)
            elif name == "test_truth_file":
                shutil.copyfile(fixture / "selection_fit/targets_raw.npy", file)
            elif name == "selection_contains_test_row":
                frame = pd.read_csv(file)
                frame.loc[0, "segment"] = "test"
                frame.to_csv(file, index=False)
                reanchor(fixture, relative)
            elif name == "duplicate_origin":
                frame = pd.read_csv(file)
                frame.loc[1, "origin_id"] = frame.loc[0, "origin_id"]
                frame.to_csv(file, index=False)
                reanchor(fixture, relative)
            elif name == "origin_horizon_overflow":
                frame = pd.read_csv(file)
                last_test = frame.index[frame.segment.eq("test")][-1]
                frame.loc[last_test, "origin_index"] = 1142
                frame.loc[last_test, "origin_id"] = "overflow:1142"
                frame.to_csv(file, index=False)
                reanchor(fixture, relative)
            elif name == "feature_differs_from_bundle":
                array = np.load(file, allow_pickle=False)
                array[0, 0] += 1.0
                np.save(file, array, allow_pickle=False)
                reanchor(fixture, relative)
            elif name == "test_numeric_history_injection":
                shutil.copyfile(fixture / "selection_fit/numeric_history.npy", file)
            command = [sys.executable, str(fixture / "code/verify_climate.py"),
                       "--package", str(fixture), "--out", str(base / "output"), "--reference"]
            process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
            observed = process.stderr.strip()
            passed = process.returncode != 0 and expected in observed
            results.append({"case": name, "exit_code": process.returncode,
                            "expected_reason": expected, "observed_reason": observed, "pass": passed})
    result = {"status": "PASS" if all(item["pass"] for item in results) else "FAIL", "cases": results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "passed": sum(item["pass"] for item in results),
                      "total": len(results)}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()
    raise SystemExit(0 if test(arguments.package.resolve(strict=True), arguments.out)["status"] == "PASS" else 2)
