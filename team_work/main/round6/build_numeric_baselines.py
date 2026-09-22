"""Export fixed master OT baselines from an accepted formal Bundle.

Selection evidence uses calibration and decision only. The output manifest is a
byte-level handoff for a later route; this command does not freeze a route.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "team_work/main/src"))
from team_eval.v2 import (  # noqa: E402
    digest, load_bundle, load_prediction_grid, numeric_baselines,
    write_baseline_predictions,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--split-spec", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scenario", choices=("proxy", "conservative_lag"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("refusing to overwrite nonempty baseline output directory")
    spec = json.loads(args.split_spec.read_text(encoding="utf-8"))
    task = next((t for t in spec["tasks"] if
                 f"{t['domain']}_h{t['horizon']}_f{t['fold_id']}" == args.task), None)
    if task is None:
        raise SystemExit("task absent from frozen split spec")
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    bundle = load_bundle(args.bundle, args.signature, args.split_spec,
                         args.task, args.scenario)
    baselines = numeric_baselines(bundle, spec["alpha_grid"],
                                  spec["seasonal_periods"][task["domain"]])
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "numeric_selection_predictions.csv"
    rows = write_baseline_predictions(csv_path, bundle, baselines, args.scenario,
                                      commit, seed=spec["seed"])
    grid = load_prediction_grid([csv_path], bundle, ("calibration", "decision"))
    if grid["origins"] * task["horizon"] * 3 != rows:
        raise SystemExit("numeric baseline row count differs from expected grid")
    manifest = {
        "status": "SELECTION_BASELINES_READY",
        "task_id": args.task, "scenario": args.scenario,
        "segments": ["calibration", "decision"],
        "bundle_signature": args.signature,
        "bundle_manifest_sha256": digest(args.bundle / "manifest.json"),
        "split_spec_sha256": digest(args.split_spec),
        "baseline_implementation_commit": commit,
        "baseline_implementation": "team_work/main/src/team_eval/v2.py::numeric_baselines",
        "csv_sha256": digest(csv_path), "prediction_rows": rows,
        "origin_count": grid["origins"], "horizon": grid["horizon"],
        "seed": spec["seed"], "candidates": ["Last", "SeasonalNaive", "AR-Ridge"],
        "config_sha256": {candidate: grid["configs"][(candidate, spec["seed"])]
                          for candidate in ("Last", "SeasonalNaive", "AR-Ridge")},
        "ridge_alpha": baselines["ridge_alpha"],
        "ridge_calibration_mse": baselines["ridge_calibration_mse"],
        "fit_rows": baselines["fit_rows"],
        "calibration_rows": baselines["calibration_rows"],
        "alpha_grid": baselines["alpha_grid"],
        "seasonal_period": spec["seasonal_periods"][task["domain"]],
        "test_evidence_used": False,
    }
    (out / "baseline_selection_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "out": str(out),
                      "rows": rows, "ridge_alpha": manifest["ridge_alpha"]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
