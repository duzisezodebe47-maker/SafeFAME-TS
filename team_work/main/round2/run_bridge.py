"""Run the master-side v2 Bundle and prediction contract checks on D: artifacts."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from team_eval.core import EvidenceError
from team_eval.v2 import (load_bundle, load_prediction_grid, numeric_baselines,
                          write_baseline_predictions, write_stage_inputs)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate-bundle", "validate-predictions", "export-baselines", "export-stage"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-signature", required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--scenario", choices=("proxy", "conservative_lag"), default="proxy")
    parser.add_argument("--predictions", type=Path, nargs="+")
    parser.add_argument("--segments", nargs="+", choices=("calibration", "decision", "test"),
                        default=("calibration", "decision"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        bundle = load_bundle(args.bundle, args.bundle_signature, args.spec, args.task_id, args.scenario)
        report = {"status": "PASS_CONTRACT_ONLY", "task_id": args.task_id, "scenario": args.scenario,
                  "bundle_signature": bundle["signature"], "bundle_manifest_sha256": bundle["manifest_sha256"],
                  "split_spec_sha256": bundle["spec_sha256"], "sample_rows": len(bundle["samples"])}
        if args.command in ("validate-predictions", "export-stage"):
            if not args.predictions:
                parser.error("--predictions is required")
            grid = load_prediction_grid(args.predictions, bundle, tuple(args.segments))
            report.update(prediction_origins=grid["origins"], prediction_horizon=grid["horizon"],
                          prediction_files_sha256=grid["file_sha256"],
                          candidate_seeds=[f"{c}:{s}" for c, s in sorted(grid["values"])])
            if args.command == "export-stage":
                if args.out is None:
                    parser.error("--out is required")
                spec = json.loads(args.spec.read_text(encoding="utf-8"))
                stage = write_stage_inputs(args.out, bundle, grid, spec, tuple(args.segments))
                report.update(status=stage["status"], stage_manifest=stage, output=str(args.out))
        elif args.command == "export-baselines":
            if args.out is None:
                parser.error("--out is required")
            spec = json.loads(args.spec.read_text(encoding="utf-8"))
            config = json.loads((Path(__file__).resolve().parents[3] / "configs/safefame_v4.json").read_text(encoding="utf-8"))
            domain = bundle["task"]["domain"]
            period = int(config["domains"][domain]["seasonal_period"])
            result = numeric_baselines(bundle, spec["alpha_grid"], period)
            repo = Path(__file__).resolve().parents[3]
            commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            count = write_baseline_predictions(args.out, bundle, result, args.scenario, commit,
                                               segments=tuple(args.segments))
            report.update(status="EXPORTED_BASELINES_CONTRACT_ONLY", rows_written=count,
                          ridge_alpha=result["ridge_alpha"], ridge_calibration_mse=result["ridge_calibration_mse"],
                          output=str(args.out))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (EvidenceError, OSError, ValueError, KeyError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
