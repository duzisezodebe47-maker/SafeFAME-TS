"""本侧数值基线 vs 主控数值基线：**逐点**比对（第六轮交付物）。

上一轮 README 把「本侧 `numeric_baselines` 与主控基线在真 Bundle 上逐点相同」
称作本轮最强的正确性证据，却**没有留下任何产物** —— 只有一句话和几个数字，
主控无法复核。本模块把这句话变成可核对的表。

比对对象：主控 `team_work/main/round6/results/agriculture_proxy_baselines/`
的 `numeric_selection_predictions.csv`（4968 行 = 3 候选 × 138 起点 × 12 步）
与其 `baseline_selection_manifest.json`。

做法：
  1. 先核**主控清单自报的 CSV 哈希** == CSV 实际字节哈希（清单本身也可能被改）；
  2. 用真 Bundle 跑本侧 `numeric_baselines`（**推理视图**，测试真值根本不传入）；
  3. 按 `(candidate_id, origin_id, step)` 逐点对齐，算 `max_abs_err`；
  4. 同时比对 α、校准 MSE、fit/cal 行数、季节周期这些**标量**参数。

判据：三个候选的 `max_abs_err` 必须**逐位为 0**（`==`，不是容差内）才算 PASS。
达不到就如实写小数值并把 verdict 置为 FAIL —— 不把"接近"写成"相同"。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bundle_reader import (  # noqa: E402
    BundleUnavailable, assert_grid, read_frozen_bundle, samples_order_view,
    segment_bounds, sha256_file, task_seasonal_period,
)
from numeric_fallbacks import numeric_baselines  # noqa: E402
from predict_io import warn_if_dirty  # noqa: E402
from runtime_profile import cpu_seconds, script_entry_snapshot  # noqa: E402
from train import RUNNABLE_SCENARIOS, to_inference_bundle  # noqa: E402

REPO_ROOT = HERE.parents[1]

COMPARE_SEGMENTS = ("calibration", "decision")
MASTER_BASELINES = ("Last", "SeasonalNaive", "AR-Ridge")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scenario", required=True, choices=RUNNABLE_SCENARIOS)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--split-spec", type=Path, required=True)
    parser.add_argument("--master-csv", type=Path, required=True,
                        help="主控 numeric_selection_predictions.csv")
    parser.add_argument("--master-manifest", type=Path, required=True,
                        help="主控 baseline_selection_manifest.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    cpu_started = cpu_seconds()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    master_manifest = json.loads(args.master_manifest.read_text(encoding="utf-8"))
    # 判据 0：主控清单自报的 CSV 哈希 == 实际字节哈希
    master_csv_sha = sha256_file(args.master_csv)
    declared = str(master_manifest.get("csv_sha256", ""))
    hash_ok = master_csv_sha == declared

    try:
        bundle = read_frozen_bundle(args.bundle, args.task, args.scenario,
                                    args.signature, args.split_spec)
        bounds = segment_bounds(args.split_spec, args.task)
    except BundleUnavailable as exc:
        print(f"BundleUnavailable: {exc}", file=sys.stderr)
        return 3

    horizon = int(args.task.split("_h")[1].split("_")[0])
    for seg in ("train",) + COMPARE_SEGMENTS:
        assert_grid(bundle, seg, bounds[seg], horizon)

    # 本侧基线：推理视图 + 只含 train/cal 的 fit_targets（测试真值传不进来）
    view = samples_order_view(bundle, ("train",) + COMPARE_SEGMENTS)
    features = to_inference_bundle(view)
    segments = np.asarray(view["segment"])
    fit_targets = np.full((len(segments), horizon), np.nan)
    for seg in ("train", "calibration"):
        mask = segments == seg
        fit_targets[mask] = np.asarray(bundle.segment(seg)["targets_standardized"],
                                       dtype=float)

    period = task_seasonal_period(args.split_spec, args.task)
    result = numeric_baselines(features, segments, fit_targets, seasonal_period=period)

    # 逐点对齐
    master_rows = list(csv.DictReader(open(args.master_csv, encoding="utf-8")))
    row_pos = {seg: np.flatnonzero(segments == seg) for seg in COMPARE_SEGMENTS}
    key_to_pos: dict[tuple[str, str, int], int] = {}
    for seg in COMPARE_SEGMENTS:
        oids = np.asarray(view["origin_id"])[row_pos[seg]]
        for local, oid in enumerate(oids):
            for step in range(1, horizon + 1):
                key_to_pos[(seg, str(oid), step)] = int(row_pos[seg][local])

    per_candidate: dict[str, dict] = {name: {"n": 0, "max_abs_err": 0.0, "exact": 0,
                                             "missing_key": 0}
                                      for name in MASTER_BASELINES}
    detail_path = out / "baseline_crosscheck.csv"
    with detail_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["candidate_id", "segment", "origin_id", "step",
                         "mine", "master", "abs_err", "exact_match"])
        for row in master_rows:
            name = row["candidate_id"]
            if name not in per_candidate:
                continue
            key = (row["segment"], row["origin_id"], int(row["step"]))
            pos = key_to_pos.get(key)
            if pos is None:
                per_candidate[name]["missing_key"] += 1
                continue
            # 预测矩阵形状 (n_rows, H)，列对应 step 1..H（step 是 1-based）
            mine = float(result["predictions"][name][pos, int(row["step"]) - 1])
            master = float(row["y_pred"])
            err = abs(mine - master)
            entry = per_candidate[name]
            entry["n"] += 1
            entry["max_abs_err"] = max(entry["max_abs_err"], err)
            exact = mine == master
            entry["exact"] += int(exact)
            writer.writerow([name, row["segment"], row["origin_id"], row["step"],
                             repr(mine), repr(master), repr(err), int(exact)])

    scalars = {
        "ridge_alpha": {"mine": result["ridge_alpha"],
                        "master": master_manifest.get("ridge_alpha")},
        "ridge_calibration_mse": {"mine": result["ridge_calibration_mse"],
                                  "master": master_manifest.get("ridge_calibration_mse")},
        "fit_rows": {"mine": result["fit_rows"],
                     "master": master_manifest.get("fit_rows")},
        "calibration_rows": {"mine": result["calibration_rows"],
                            "master": master_manifest.get("calibration_rows")},
        "seasonal_period": {"mine": period,
                            "master": master_manifest.get("seasonal_period")},
        "alpha_grid": {"mine": list(result["alpha_grid"]),
                       "master": master_manifest.get("alpha_grid")},
    }
    for entry in scalars.values():
        entry["equal"] = entry["mine"] == entry["master"]

    all_exact = all(e["n"] > 0 and e["exact"] == e["n"] and e["missing_key"] == 0
                    for e in per_candidate.values())
    checks = {
        "master_csv_hash_matches_manifest": hash_ok,
        "all_points_bitwise_equal": all_exact,
        "scalars_equal": all(v["equal"] for v in scalars.values()),
    }
    passed = all(checks.values())

    report = {
        "task": args.task, "scenario": args.scenario,
        "bundle_signature": bundle.signature,
        "split_spec_sha256": sha256_file(args.split_spec),
        "master_csv": {"path_name": args.master_csv.name,
                       "sha256": master_csv_sha,
                       "declared_sha256": declared,
                       "sha256_matches_declared": hash_ok,
                       "rows": len(master_rows)},
        "master_manifest": {
            "bundle_signature": master_manifest.get("bundle_signature"),
            "bundle_manifest_sha256": master_manifest.get("bundle_manifest_sha256"),
            "split_spec_sha256": master_manifest.get("split_spec_sha256"),
            "baseline_implementation_commit": master_manifest.get(
                "baseline_implementation_commit"),
            "origin_count": master_manifest.get("origin_count"),
        },
        "per_candidate": per_candidate,
        "scalars": scalars,
        "checks": checks,
        "verdict": "PASS" if passed else "FAIL",
        "code_provenance": warn_if_dirty(REPO_ROOT, HERE),
        "runtime": script_entry_snapshot(started, cpu_started),
        "note": "逐点比对主控选择期基线 CSV；exact 列要求**逐位相等**，非容差内",
    }
    (out / "baseline_crosscheck.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    print(json.dumps({"status": report["verdict"],
                      "master_csv_hash_ok": hash_ok,
                      "per_candidate": {k: {"n": v["n"], "exact": v["exact"],
                                            "max_abs_err": v["max_abs_err"]}
                                        for k, v in per_candidate.items()},
                      "scalars_equal": checks["scalars_equal"]}, ensure_ascii=False))
    return 0 if passed else 6


if __name__ == "__main__":
    raise SystemExit(main())
