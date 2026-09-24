"""Master-only Agriculture test scoring after the route has been sealed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))

from team_eval.core import evaluate_test, jsonl, validate  # noqa: E402
from team_eval.v2 import load_bundle, load_prediction_grid  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def iid_origin_bootstrap(delta: list[float], draws: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(delta)
    means = [sum(delta[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws)]
    return [quantile(means, 0.025), quantile(means, 0.975)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-signature", required=True)
    parser.add_argument("--split-spec", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--evaluator-spec", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path, required=True)
    parser.add_argument("--selected-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"output directory must be fresh: {out}")
    out.mkdir(parents=True, exist_ok=True)

    task = json.loads(args.task.read_text(encoding="utf-8"))
    spec = json.loads(args.evaluator_spec.read_text(encoding="utf-8"))
    route = json.loads(args.route.read_text(encoding="utf-8"))
    bundle = load_bundle(args.bundle, args.bundle_signature, args.split_spec,
                         task["task_id"], "proxy")
    grid = load_prediction_grid([args.baseline_csv, args.selected_csv], bundle, ("test",))

    chosen = str(route["selected"])
    fallback = str(route["fallback"])
    allowed = {fallback, chosen}
    groups = {key: value for key, value in grid["values"].items() if key[0] in allowed}
    configs = {key: value for key, value in grid["configs"].items() if key[0] in allowed}
    if set(candidate for candidate, _seed in groups) != allowed:
        raise RuntimeError(f"test evidence lacks selected/fallback: {allowed}")
    if any(seed != 2026 for _candidate, seed in groups):
        raise RuntimeError("unexpected test seed")

    h = int(task["horizon"])
    sample_rows: list[dict] = []
    prediction_rows: list[dict] = []
    bound_rows: list[dict] = []
    samples = [(i, row) for i, row in enumerate(bundle["samples"])
               if row["segment"] == "test"]
    if len(samples) != 42:
        raise RuntimeError(f"expected 42 test origins, got {len(samples)}")

    matrices: dict[str, list[list[float]]] = {fallback: [], chosen: []}
    truth_matrix: list[list[float]] = []
    for i, sample in samples:
        origin = int(sample["origin_index"])
        truth = bundle["arrays"]["targets_standardized"][i].astype(float).tolist()
        truth_matrix.append(truth)
        sample_rows.append({
            "task_id": task["task_id"], "fold_id": task["fold_id"],
            "origin_id": origin, "original_origin_id": sample["origin_id"],
            "segment": "test", "horizon": h, "target": truth,
            "cutoff_index": origin - 1, "target_start_index": origin,
            "snapshot_sha256": task["snapshot_sha256"],
            "feature_manifest_sha256": bundle["signature"],
            "preprocessing_fit_end": task["train_end"],
        })
        for candidate in (fallback, chosen):
            key = (candidate, 2026)
            forecast = [groups[key][("test", sample["origin_id"], origin, step)]
                        for step in range(1, h + 1)]
            matrices[candidate].append(forecast)
            prediction_rows.append({
                "task_id": task["task_id"], "fold_id": task["fold_id"],
                "origin_id": origin, "original_origin_id": sample["origin_id"],
                "segment": "test", "candidate_id": candidate, "seed": 2026,
                "prediction": forecast, "config_sha256": configs[key],
                "feature_manifest_sha256": bundle["signature"],
            })
        for step in range(h):
            bound_rows.append({
                "origin_id": sample["origin_id"], "origin_index": origin,
                "step": step + 1, "y_true_standardized": truth[step],
                "fallback_pred_standardized": matrices[fallback][-1][step],
                "selected_pred_standardized": matrices[chosen][-1][step],
            })

    samples_path = out / "test_samples.jsonl"
    predictions_path = out / "test_predictions.jsonl"
    samples_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in sample_rows),
                            encoding="utf-8")
    predictions_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in prediction_rows),
                                encoding="utf-8")
    task_copy, spec_copy, route_copy = out / "task.json", out / "spec.json", out / "sealed_route.json"
    task_copy.write_bytes(args.task.read_bytes())
    spec_copy.write_bytes(args.evaluator_spec.read_bytes())
    route_copy.write_bytes(args.route.read_bytes())

    checked_samples, checked_predictions, _ = validate(
        task, jsonl(samples_path), jsonl(predictions_path), require_test=True, spec=spec)
    protocol = evaluate_test(task, checked_samples, checked_predictions, route, spec)

    truth_arr = np.asarray(truth_matrix, dtype=float)
    fallback_arr = np.asarray(matrices[fallback], dtype=float)
    selected_arr = np.asarray(matrices[chosen], dtype=float)
    fallback_origin_mse = np.mean((truth_arr - fallback_arr) ** 2, axis=1)
    selected_origin_mse = np.mean((truth_arr - selected_arr) ** 2, axis=1)
    delta = (fallback_origin_mse - selected_origin_mse).tolist()
    protocol["iid_origin_bootstrap_delta_ci95"] = iid_origin_bootstrap(
        delta, int(spec["uncertainty"]["draws"]), int(spec["uncertainty"]["seed"]))
    protocol["origin_win_rate"] = float(np.mean(selected_origin_mse < fallback_origin_mse))
    protocol["selected_rmse"] = float(np.sqrt(protocol["selected_mse"]))
    protocol["fallback_rmse"] = float(np.sqrt(protocol["fallback_mse"]))
    protocol["test_prediction_sha256"] = sha256(args.selected_csv)
    protocol["baseline_prediction_sha256"] = sha256(args.baseline_csv)
    protocol["route_sha256"] = sha256(args.route)
    protocol["bundle_signature"] = bundle["signature"]
    protocol["split_spec_sha256"] = sha256(args.split_spec)

    numeric_fit = json.loads((args.bundle / task["task_id"] / "numeric_fit.json").read_text(encoding="utf-8"))
    scale = float(numeric_fit["std"])
    protocol["raw_scale"] = {
        "unit": "original OT unit",
        "fallback_mse": float(protocol["fallback_mse"] * scale * scale),
        "selected_mse": float(protocol["selected_mse"] * scale * scale),
        "fallback_rmse": float(protocol["fallback_rmse"] * scale),
        "selected_rmse": float(protocol["selected_rmse"] * scale),
        "selected_mae": float(protocol["selected_mae"] * scale),
    }

    half = len(samples) // 2
    halves = []
    for name, slc in (("first_half", slice(0, half)), ("second_half", slice(half, None))):
        fb = float(np.mean((truth_arr[slc] - fallback_arr[slc]) ** 2))
        se = float(np.mean((truth_arr[slc] - selected_arr[slc]) ** 2))
        halves.append({"half": name, "origins": int(len(truth_arr[slc])),
                       "fallback_mse": fb, "selected_mse": se,
                       "gain_pct": 100.0 * (1.0 - se / fb)})

    horizon_rows = []
    for step in range(h):
        fb = float(np.mean((truth_arr[:, step] - fallback_arr[:, step]) ** 2))
        se = float(np.mean((truth_arr[:, step] - selected_arr[:, step]) ** 2))
        horizon_rows.append({"step": step + 1, "fallback_mse": fb, "selected_mse": se,
                             "gain_pct": 100.0 * (1.0 - se / fb)})

    write_json(out / "Agriculture_test_score.json", protocol)
    for filename, rows in (("Agriculture_test_predictions_bound.csv", bound_rows),
                           ("Agriculture_horizon_metrics.csv", horizon_rows),
                           ("Agriculture_half_metrics.csv", halves)):
        with (out / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    with (out / "Agriculture_paired_bootstrap.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["origin_index", "fallback_mse", "selected_mse", "delta"])
        writer.writeheader()
        for (_i, sample), fb, se, de in zip(samples, fallback_origin_mse, selected_origin_mse, delta):
            writer.writerow({"origin_index": sample["origin_index"], "fallback_mse": fb,
                             "selected_mse": se, "delta": de})

    manifest = {"status": "PASS", "files_sha256": {p.name: sha256(p) for p in out.iterdir() if p.is_file()},
                "input_sha256": {"baseline_csv": sha256(args.baseline_csv),
                                  "selected_csv": sha256(args.selected_csv),
                                  "route": sha256(args.route), "task": sha256(args.task),
                                  "spec": sha256(args.evaluator_spec)},
                "note": "Positive paired delta means N+S+Q+SF has lower origin-level MSE than AR-Ridge."}
    write_json(out / "scoring_manifest.json", manifest)
    print(json.dumps({"status": "PASS", "selected_mse": protocol["selected_mse"],
                      "fallback_mse": protocol["fallback_mse"],
                      "gain_pct": protocol["gain_pct"],
                      "delta_ci95": protocol["delta_ci95"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
