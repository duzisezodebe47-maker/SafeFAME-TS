"""Read-only standard-library audit of two Git branch deliveries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

A = None
B = None


def rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def data_audit():
    summary = json.loads((A / "audit_summary.json").read_text(encoding="utf-8"))
    tests = json.loads((A / "tests.json").read_text(encoding="utf-8"))
    numerical, textual = rows(A / "numerical_sources.csv"), rows(A / "textual_sources.csv")
    coverage = rows(A / "smoke_coverage.csv")
    legacy = rows(A / "legacy_proxy_coverage.csv")
    compat = rows(A / "v4_compatibility.csv")
    assert len(numerical) == 10 and len(textual) == 20 and len(legacy) == 10 and len(compat) == 10
    assert sum(int(r["raw_rows"]) for r in numerical) == summary["raw_numerical_rows"]
    assert sum(int(r["clean_rows"]) for r in numerical) == summary["clean_numerical_rows"]
    assert sum(int(r["unknown_target_rows"]) for r in numerical) == summary["unknown_target_rows"]
    assert sum(int(r["raw_rows"]) for r in textual) == summary["raw_text_rows"]
    assert sum(int(r["retained"]) for r in textual) == summary["canonical_facts"]
    assert sum(int(r["origins"]) for r in legacy) == summary["legacy_proxy_origins_checked"]
    assert all(r["v4_target_and_dates_equal"] == "True" for r in compat)
    assert tests["status"] == "PASS" and tests["count"] == len(tests["tests"]) == 19
    assert all(r["status"] == "PASS" for r in tests["tests"])
    assert sum(int(r["valid_origins"]) for r in coverage if r["scenario"] == "proxy") == tests["sample_count"] == 388
    assert sum(int(r["excluded_origins"]) for r in coverage if r["scenario"] == "proxy") == tests["cross_boundary_excluded"] == 8
    assert all(int(r["covered_origins"]) == 0 for r in coverage if r["scenario"] == "complete_source")
    return {"status": "COMPACT_EVIDENCE_CONSISTENT", "numeric_series": len(numerical), "text_files": len(textual),
            "raw_numeric_rows": summary["raw_numerical_rows"], "clean_numeric_rows": summary["clean_numerical_rows"],
            "raw_text_rows": summary["raw_text_rows"], "canonical_facts": summary["canonical_facts"],
            "recorded_tests": len(tests["tests"]), "smoke_samples": tests["sample_count"],
            "full_raw_replay": False}


def model_audit():
    overall = json.loads((B / "ablation_manifest.json").read_text(encoding="utf-8"))
    summary = rows(B / "ablation_summary.csv")
    assert overall["completed"] == len(overall["candidates"]) == len(summary) == 6
    assert overall["failures"] == []
    expected = set(overall["candidates"])
    assert {r["candidate"] for r in summary} == expected
    common_grid = None
    feature_hashes = set()
    commits = set()
    for row in summary:
        candidate = row["candidate"]
        folder = B / candidate.replace("+", "_")
        manifest = json.loads((folder / "run_manifest.json").read_text(encoding="utf-8"))
        pred = rows(folder / "predictions.csv")
        assert manifest["status"] == "completed" and manifest["rows_written"] == len(pred) == 1008
        assert manifest["config"]["candidate"] == candidate
        assert manifest["n_parameters"] == int(row["n_parameters"])
        assert sum(manifest["branch_widths"].values()) == int(row["n_features"])
        assert manifest["alpha_by_group"] == json.loads(row["alpha_by_group"])
        digest = hashlib.sha256(json.dumps(manifest["config"], sort_keys=True, ensure_ascii=False,
                                          separators=(",", ":")).encode("utf-8")).hexdigest()
        assert digest == manifest["config_hash"]
        assert all(r["candidate_id"] == candidate and r["task_id"] == "Climate_h4" and r["fold_id"] == "0"
                   and r["seed"] == "2026" for r in pred)
        assert all(r["config_hash"] == digest and r["feature_hash"] == manifest["feature_hash"] for r in pred)
        assert all(math.isfinite(float(r["y_pred"])) for r in pred)
        keys = [(int(r["origin_id"]), int(r["horizon"])) for r in pred]
        assert len(set(keys)) == 1008
        assert {step for _, step in keys} == {1, 2, 3, 4}
        origins = {origin for origin, _ in keys}
        assert len(origins) == int(row["n_origins"]) == 252
        assert set(keys) == {(origin, step) for origin in origins for step in range(1, 5)}
        if common_grid is None:
            common_grid = set(keys)
        assert set(keys) == common_grid
        assert math.isclose(100 * (1 - float(row["test_mse"]) / float(row["last_anchor_mse"])),
                            float(row["gain_vs_last_pct"]), abs_tol=0.002)
        feature_hashes.add(manifest["feature_hash"])
        commits.update(r["code_commit"] for r in pred)
    assert len(feature_hashes) == 1 and len(commits) == 1
    return {"status": "COMPACT_EVIDENCE_CONSISTENT", "candidates": len(summary),
            "prediction_rows": len(summary) * 1008, "unique_origin_step_grid": len(common_grid),
            "feature_hash": next(iter(feature_hashes)), "recorded_prediction_commit": next(iter(commits)),
            "truth_replay": False, "readme_claimed_tests": 87, "tests_independently_rerun": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-delivery", type=Path, required=True,
                        help="path to 辅助电脑02交付01次")
    parser.add_argument("--model-delivery", type=Path, required=True,
                        help="path to 03 辅助电脑二 交付一次")
    args = parser.parse_args()
    A = args.data_delivery / "evidence"
    B = args.model_delivery / "evidence"
    print(json.dumps({"data": data_audit(), "model": model_audit()}, ensure_ascii=False, indent=2))
