"""Round-three contract tests, including three-reader interoperability and faults."""
from __future__ import annotations

import argparse
import gc
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FIRST = ROOT / "辅助电脑02交付01次"
SECOND = ROOT / "辅助电脑02交付02次"
sys.path[:0] = [str(HERE), str(SECOND), str(FIRST)]

from audit import digest, write_json  # noqa: E402
from bundle import read_bundle as first_read_bundle  # noqa: E402
from bundle_read_example import read_task  # noqa: E402
from round3_pipeline import current_master_spec, latest_preview, promote  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def reseal_file(bundle: Path, relative: str) -> None:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][relative] = digest(bundle / relative)
    write_json(manifest_path, manifest)


def run_regression(command: list[str], evidence: Path, expected: int) -> dict:
    result = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise AssertionError(f"Regression command failed ({result.returncode}): {result.stdout[-1000:]}")
    report = json.loads(evidence.read_text(encoding="utf-8"))
    observed = int(report.get("count", len(report.get("tests", []))))
    if observed != expected or report.get("status") != "PASS":
        raise AssertionError(f"Expected {expected} passing checks, got {observed}: {report.get('status')}")
    return {"test": f"regression_{expected}", "status": "PASS", "exit_code": 0,
            "checks": observed, "command": subprocess.list2cmdline(command)}


def make_fixture_spec(path: Path) -> dict:
    spec, _raw, _sha, _commit = current_master_spec()
    registry = pd.read_csv(SECOND / "evidence/task_registry.csv").set_index("task_id")
    spec["status"] = "frozen"
    spec["approved_by"] = "TEST_FIXTURE_ONLY_CONTRACT_TEST"
    spec["test_fixture_nonce"] = uuid.uuid4().hex
    for task in spec["tasks"]:
        task_id = f"{task['domain']}_h{task['horizon']}_f{task['fold_id']}"
        task["numerical_sha256"] = registry.loc[task_id, "numerical_sha256"]
    write_json(path, spec)
    return spec


def cleanup_stale_test_fixtures(root: Path) -> int:
    removed = 0
    resolved_root = root.resolve()
    for candidate in root.glob("round3-test-fixture-*"):
        if candidate.parent.resolve() != resolved_root or not candidate.is_dir():
            continue
        manifest_path = candidate / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if manifest.get("inputs", {}).get("test_fixture_only") is True:
            shutil.rmtree(candidate)
            removed += 1
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-reader", type=Path, required=True)
    parser.add_argument("--model-reader", type=Path, required=True)
    args = parser.parse_args()
    python = ROOT / ".venv/Scripts/python.exe"
    tests = [
        run_regression([str(python), str(FIRST / "test_pipeline.py")],
                       FIRST / "evidence/tests.json", 19),
        run_regression([str(python), str(SECOND / "test_round2.py"), str(latest_preview())],
                       SECOND / "evidence/tests_round2.json", 15),
    ]
    transfer = load_module("round3_transfer_bundle", HERE / "transfer_bundle.py")
    master_source = args.master_reader.resolve().parent.parent
    sys.path.insert(0, str(master_source))
    master = importlib.import_module("team_eval.v2")
    os.environ["DATA_SIDE_DIR"] = str(FIRST)
    model = load_module("model_bundle_reader_round2", args.model_reader.resolve())
    temp_root = ROOT / "data_processed/team_data"
    temp_root.mkdir(parents=True, exist_ok=True)
    stale_removed = cleanup_stale_test_fixtures(temp_root)
    fixture_spec = temp_root / "round3_test_fixture_split_spec.json"
    make_fixture_spec(fixture_spec)
    bundle, manifest = promote(latest_preview(), fixture_spec,
                               SECOND / "evidence/task_registry.csv", test_only=True)
    signature = manifest["signature"]
    task_id, scenario = "Agriculture_h12_f1", "proxy"

    data_value = read_task(bundle, fixture_spec, signature, task_id, scenario)
    first_samples, first_arrays, _ = first_read_bundle(bundle, task_id, scenario, signature)
    master_value = master.load_bundle(bundle, signature, fixture_spec, task_id, scenario)
    model_value = model.read_frozen_bundle(bundle, task_id, scenario, signature)
    origins = data_value["arrays"]["origin_index"]
    same = (np.array_equal(origins, first_arrays["origin_index"]) and
            np.array_equal(origins, master_value["arrays"]["origin_index"]) and
            np.array_equal(origins, model_value.arrays["origin_index"]) and
            first_samples.origin_id.tolist() == [row["origin_id"] for row in data_value["samples"]] ==
            [row["origin_id"] for row in master_value["samples"]] == model_value.samples.origin_id.tolist() and
            np.array_equal(data_value["arrays"]["targets_standardized"], first_arrays["targets_standardized"]) and
            np.array_equal(data_value["arrays"]["targets_standardized"], master_value["arrays"]["targets_standardized"]) and
            np.array_equal(data_value["arrays"]["targets_standardized"], model_value.arrays["targets_standardized"]))
    if not same:
        raise AssertionError("Three readers disagree on Agriculture identity/order/scale")
    tests.append({"test": "four_reader_interoperability", "status": "PASS", "exit_code": 0,
                  "readers": ["round3_data", "round1_data", "master_v2", "model_round2"],
                  "task_id": task_id, "scenario": scenario, "rows": len(origins)})
    targets = bundle / task_id / "targets.npy"
    targets_raw = bundle / task_id / "targets_raw.npy"
    if targets.read_bytes() != targets_raw.read_bytes():
        raise AssertionError("targets.npy and targets_raw.npy differ")
    tests.append({"test": "targets_alias_byte_identical", "status": "PASS", "exit_code": 0,
                  "sha256": digest(targets)})
    del data_value, first_samples, first_arrays, master_value, model_value, origins
    gc.collect()

    probe = HERE / "contract_probe.py"
    common_master = ["--master-reader", str(args.master_reader.resolve()), "--bundle", str(bundle),
                     "--spec", str(fixture_spec), "--signature", signature,
                     "--task", task_id, "--scenario", scenario]

    def reject_command(name: str, command: list[str], expected: str) -> None:
        result = subprocess.run(command, cwd=ROOT, text=True, encoding="utf-8",
                                errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = result.stdout.strip()
        if result.returncode == 0:
            raise AssertionError(f"{name}: injected fault was accepted")
        if expected.lower() not in output.lower():
            raise AssertionError(f"{name}: wrong rejection ({result.returncode}): {output[-1000:]}")
        tests.append({"test": name, "status": "PASS", "exit_code": result.returncode,
                      "expected_rejection": True, "command": subprocess.list2cmdline(command),
                      "error": output[-2000:]})

    schema_path = bundle / "schema.json"
    original_schema = schema_path.read_bytes()
    schema = json.loads(original_schema)
    schema["status"] = "FROZEN"
    write_json(schema_path, schema)
    reseal_file(bundle, "schema.json")
    reject_command("status_case_mismatch", [str(python), str(probe), "data-read",
                   "--bundle", str(bundle), "--spec", str(fixture_spec), "--signature", signature,
                   "--task", task_id, "--scenario", scenario], "lowercase frozen")
    schema_path.write_bytes(original_schema)
    reseal_file(bundle, "schema.json")

    original_target = targets.read_bytes()
    targets.unlink()
    reject_command("missing_targets_npy", [str(python), str(probe), "master-read", *common_master],
                   "inventory mismatch")
    targets.write_bytes(original_target)

    changed_spec = temp_root / "round3_test_fixture_split_spec_changed.json"
    changed_spec.write_bytes(fixture_spec.read_bytes() + b"\n")
    changed_master = ["--master-reader", str(args.master_reader.resolve()), "--bundle", str(bundle),
                      "--spec", str(changed_spec), "--signature", signature]
    reject_command("split_spec_byte_change", [str(python), str(probe), "master-verify", *changed_master],
                   "another split spec")

    rogue = bundle / "rogue.tmp"
    rogue.write_bytes(b"unexpected")
    reject_command("manifest_extra_file", [str(python), str(probe), "master-verify", *common_master],
                   "inventory mismatch")
    rogue.unlink()
    manifest_path = bundle / "manifest.json"
    original_manifest = manifest_path.read_bytes()
    changed_manifest = json.loads(original_manifest)
    changed_manifest["files"].pop("schema.json")
    write_json(manifest_path, changed_manifest)
    reject_command("manifest_missing_entry", [str(python), str(probe), "master-verify", *common_master],
                   "inventory mismatch")
    manifest_path.write_bytes(original_manifest)

    origin_path = bundle / task_id / "origin_index.npy"
    origin_original = origin_path.read_bytes()
    origin = np.load(origin_path, allow_pickle=False)
    origin[[0, 1]] = origin[[1, 0]]
    np.save(origin_path, origin, allow_pickle=False)
    reseal_file(bundle, f"{task_id}/origin_index.npy")
    reject_command("origin_order_swap", [str(python), str(probe), "master-read", *common_master],
                   "identity or array order")
    origin_path.write_bytes(origin_original)
    reseal_file(bundle, f"{task_id}/origin_index.npy")

    standardized_path = bundle / task_id / "targets_standardized.npy"
    standardized_original = standardized_path.read_bytes()
    standardized = np.load(standardized_path, allow_pickle=False)
    standardized[0, 0] += 1.0
    np.save(standardized_path, standardized, allow_pickle=False)
    reseal_file(bundle, f"{task_id}/targets_standardized.npy")
    reject_command("bad_target_standardization", [str(python), str(probe), "master-read", *common_master],
                   "targets disagree")
    standardized_path.write_bytes(standardized_original)
    reseal_file(bundle, f"{task_id}/targets_standardized.npy")

    with tempfile.TemporaryDirectory(prefix="round3-transfer-test-", dir=temp_root) as temp:
        destination = Path(temp) / "received"
        receipt = Path(temp) / "receipt.json"
        transfer.copy_and_verify(bundle, destination, receipt, "0" * 40, digest(fixture_spec))
        corrupt = destination / "schema.json"
        corrupt.write_bytes(corrupt.read_bytes() + b"x")
        reject_command("transfer_single_byte_corruption",
                       [str(python), str(probe), "transfer-verify", "--source", str(bundle),
                        "--destination", str(destination)], "differs")

    current_spec, current_raw, current_sha, master_commit = current_master_spec()
    current_path = temp_root / "round3_current_master_split_spec.json"
    current_path.write_bytes(current_raw)
    before = len(tests)
    reject_command("master_freeze_gate",
                   [str(python), str(HERE / "round3_pipeline.py"), "build", "--split-spec", str(current_path)],
                   "requires master status=frozen")
    tests[-1].update({"observed_status": current_spec.get("status"),
                      "approved_by": current_spec.get("approved_by"),
                      "split_spec_sha256": current_sha, "master_commit": master_commit})
    if len(tests) != before + 1:
        raise AssertionError("Master freeze rejection was not recorded")
    report = {"status": "PASS_WITH_FORMAL_BUILD_BLOCKED", "data_status": "DATA_FORMAL_PENDING",
              "count": len(tests), "tests": tests, "test_fixture_only": True,
              "stale_test_fixtures_removed": stale_removed,
              "fixture_bundle_signature": signature,
              "limitation": "Contract fixture is not a master-approved formal Bundle and was deleted after testing."}
    write_json(HERE / "tests_round3.json", report)
    if bundle.is_dir() and bundle.parent == temp_root and bundle.name.startswith("round3-test-fixture-"):
        shutil.rmtree(bundle)
    fixture_spec.unlink(missing_ok=True)
    changed_spec.unlink(missing_ok=True)
    current_path.unlink(missing_ok=True)
    print(json.dumps({"status": report["status"], "tests": len(tests),
                      "formal_status": report["data_status"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
