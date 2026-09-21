"""Read-only compact-evidence audit; never claims to verify absent Bundle bytes."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def audit(data_dir, model_dir, master_dir):
    data_dir, model_dir, master_dir = map(Path, (data_dir, model_dir, master_dir))
    evidence = data_dir / "evidence"
    manifest = read_json(data_dir / "manifest.json")
    preview = read_json(evidence / "bundle_manifest.json")
    registry = rows(evidence / "task_registry.csv")
    coverage = rows(evidence / "coverage.csv")
    sample_summary = rows(evidence / "sample_audit_summary.csv")
    tests = read_json(evidence / "tests_round2.json")
    master_spec = read_json(master_dir / "split_spec_v2.json")
    checks = []

    def add(name, okay, detail):
        checks.append({"check": name, "status": "PASS" if okay else "FAIL", "detail": detail})

    referenced = {**manifest["compact_evidence_sha256"], **manifest["pipeline_code_sha256"]}
    mismatches = [name for name, expected in referenced.items() if digest(data_dir / name) != expected]
    add("data_compact_and_code_hashes", not mismatches,
        {"checked_files": len(referenced), "mismatches": mismatches})
    canonical = json.dumps(preview["inputs"], sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    sig = hashlib.sha256(canonical).hexdigest()
    add("data_preview_manifest_signature",
        sig == preview["signature"] == preview["bundle_signature"] == manifest["bundle"]["bundle_signature"],
        {"signature": sig, "file_inventory": len(preview["files"])})
    add("data_preview_manifest_file_count", len(preview["files"]) == manifest["bundle"]["files"] == 88,
        {"recorded": len(preview["files"]), "claimed": manifest["bundle"]["files"]})

    expected = {f"{t['domain']}_h{t['horizon']}_f{t['fold_id']}": t for t in master_spec["tasks"]}
    mapped = {r["task_id"]: r for r in registry}
    bad_tasks = []
    for task_id, row in mapped.items():
        if task_id not in expected:
            bad_tasks.append(task_id)
            continue
        task = expected[task_id]
        okay = (int(row["n_rows"]) == task["n_rows_expected"] and
                int(row["input_len"]) == task["input_len"] and
                int(row["horizon"]) == task["horizon"] and
                int(row["lag_days"]) == task["lag_days"] and
                [int(row[c]) for c in ("train_end", "calibration_end", "decision_end", "test_end")] == task["bounds"] and
                len(row["numerical_sha256"]) == 64)
        if not okay:
            bad_tasks.append(task_id)
    add("task_registry_matches_master_draft", len(registry) == 4 and set(mapped) == set(expected) and not bad_tasks,
        {"tasks": list(mapped), "mismatches": bad_tasks})
    retained = sum(int(r["expected_retained_origins"]) for r in registry)
    add("registry_retained_total", retained == 16438 == tests["tests"][1].get("rows"),
        {"retained": retained, "reported_contract_rows": tests["tests"][1].get("rows")})

    groups = defaultdict(dict)
    for row in coverage:
        groups[(row["task_id"], row["segment"])][row["scenario"]] = row
    coverage_ok = len(coverage) == 48 and len(groups) == 16
    for scenarios in groups.values():
        if set(scenarios) != {"proxy", "conservative_lag", "complete_source_audit_only"}:
            coverage_ok = False
            continue
        proxy, lag, complete = (scenarios[k] for k in ("proxy", "conservative_lag", "complete_source_audit_only"))
        coverage_ok &= (int(proxy["retained_origins"]) == int(lag["retained_origins"]) == int(complete["retained_origins"])
                        and int(lag["text_available_origins"]) <= int(proxy["text_available_origins"])
                        and int(complete["text_available_origins"]) == 0)
    add("coverage_scenarios_consistent", bool(coverage_ok), {"rows": len(coverage), "task_segments": len(groups)})
    totals = defaultdict(int)
    for row in sample_summary:
        totals[(row["task_id"], row["segment"], row["reason"])] += int(row["origins"])
    summary_ok = all(totals[(task, segment, "retained")] == int(scenarios["proxy"]["retained_origins"])
                     for (task, segment), scenarios in groups.items())
    add("sample_summary_matches_coverage", summary_ok, {"summary_rows": len(sample_summary)})
    add("recorded_data_tests", tests.get("status") == "PASS" and tests.get("count") == 15 and
        len(tests.get("tests", [])) == 15 and all(t.get("status") == "PASS" for t in tests["tests"]),
        {"claimed": tests.get("count"), "independently_rerun": False})

    equal = {name: digest(master_dir / name) == digest(model_dir / "protocol" / name)
             for name in ("split_spec_v2.json", "prediction_contract_v2.json")}
    add("model_protocol_byte_identity", all(equal.values()), equal)
    sources = list(data_dir.glob("*.py")) + list(model_dir.glob("model/*.py")) + list(model_dir.glob("model/tests/*.py"))
    for path in sources:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    add("python_source_syntax", True, {"parsed_files": len(sources)})

    data_code = (data_dir / "round2_pipeline.py").read_text(encoding="utf-8")
    model_reader = (model_dir / "model/bundle_reader.py").read_text(encoding="utf-8")
    conflict = ("targets_raw=y_raw" in data_code and '"targets"' in model_reader and
                "status='FROZEN' if formal" in data_code and 'schema.get("status") != "frozen"' in model_reader)
    add("cross_branch_bundle_schema_compatible", not conflict,
        {"conflict": "data writes targets_raw.npy and FROZEN; model/master require targets.npy and frozen"})
    add("formal_large_bundle_available_in_git", False,
        {"reason": "only manifest and compact evidence committed; 88 Bundle files are absent"})
    add("real_model_predictions_available_in_git", False,
        {"reason": "model delivery contains source and synthetic tests, no real predictions or null CSV"})
    return {"status": "PARTIAL_ENGINEERING_DELIVERY_WITH_BLOCKERS",
            "scope": "read-only Git compact evidence and source syntax",
            "data_commit": "81e3f2e90ea8c7bbd9358cafcbc5724dff821b9a",
            "model_commit": "84e665be9d2e7fa1f6fe946764482e476c7096e9",
            "checks": checks,
            "limitations": ["large Bundle bytes unavailable", "Python 3.12 scientific dependencies unavailable on master audit host",
                            "no real model predictions or 999-refit nulls"]}


def main():
    parser = argparse.ArgumentParser()
    for name in ("data-dir", "model-dir", "master-dir", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.data_dir, args.model_dir, args.master_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "pass": sum(c["status"] == "PASS" for c in report["checks"]),
                      "fail": sum(c["status"] == "FAIL" for c in report["checks"]), "out": str(args.out)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
