"""Independent arithmetic and routing audit of compact, tracked v4 CSV evidence."""
import csv
import json
import math
import statistics
from pathlib import Path

from .core import EvidenceError, sha256


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def audit(repo):
    repo = Path(repo)
    folder = repo / "outputs/safefame_v4"
    cfg = json.loads((repo / "configs/safefame_v4.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    tasks, candidates = read_csv(folder / "task_results.csv"), read_csv(folder / "candidate_results.csv")
    expected = {(domain, str(h), str(f)) for domain, info in cfg["domains"].items()
                for h in info["horizons"] for f in range(1, len(cfg["folds"]) + 1)}
    keys = [(x["domain"], x["horizon"], x["fold"]) for x in tasks]
    if len(keys) != len(expected) or set(keys) != expected:
        raise EvidenceError("legacy: task coverage or duplicate mismatch")
    grouped = {}
    for row in candidates:
        key = (row["domain"], row["horizon"], row["fold"])
        grouped.setdefault(key, []).append(row)
    if set(grouped) != expected or any({r["variant"] for r in group} != {"semantic_residual", "frequency_residual"} or len(group) != 2 for group in grouped.values()):
        raise EvidenceError("legacy: candidate coverage or duplicate mismatch")
    p_threshold = float(cfg["p_threshold"])
    route_errors = []
    numeric = 0
    for row in tasks:
        key = (row["domain"], row["horizon"], row["fold"])
        eligible = [r for r in grouped[key] if r["eligible"] == "True"]
        for r in grouped[key]:
            # Decision fallback MSE is only in omitted task-level audit; verify available gates only.
            if r["eligible"] == "True" and (r["segment_wins"] != "True" or float(r["p_value"]) > p_threshold):
                route_errors.append(key + (r["variant"], "invalid gate"))
        selected = min(eligible, key=lambda r: (float(r["decision_mse"]), r["variant"]))["variant"] if eligible else "numeric_fallback"
        if row["selected"] != selected:
            route_errors.append(key + (row["selected"], selected))
        if row["selected"] == "numeric_fallback":
            numeric += 1
            if not math.isclose(float(row["selected_mse"]), float(row["fallback_mse"]), rel_tol=1e-9, abs_tol=1e-9):
                route_errors.append(key + ("fallback mse",))
        else:
            chosen = next(r for r in grouped[key] if r["variant"] == row["selected"])
            if not math.isclose(float(row["selected_mse"]), float(chosen["mse"]), rel_tol=1e-9, abs_tol=1e-9):
                route_errors.append(key + ("selected mse",))
        base = float(row["fallback_mse"])
        gain = 100 * (1 - float(row["selected_mse"]) / base)
        delta = base - float(row["selected_mse"])
        if not math.isclose(gain, float(row["selected_gain_pct"]), rel_tol=1e-9, abs_tol=1e-9) or not math.isclose(delta, float(row["selected_delta"]), rel_tol=1e-9, abs_tol=1e-9):
            route_errors.append(key + ("gain/delta arithmetic",))
    if route_errors:
        raise EvidenceError(f"legacy: {len(route_errors)} routing/arithmetic errors; first={route_errors[0]}")
    gains = [float(r["selected_gain_pct"]) for r in tasks]
    if len(tasks) != summary["task_folds"] or len(candidates) != summary["candidate_paths"] or numeric != summary["numeric_fallback"] or not math.isclose(statistics.mean(gains), summary["selected_mean_gain_pct"], rel_tol=1e-9, abs_tol=1e-9):
        raise EvidenceError("legacy: summary mismatch")
    by_family = {}
    for row in tasks:
        by_family.setdefault((row["domain"], row["horizon"]), []).append(float(row["selected_gain_pct"]))
    families = sorted(((domain, horizon, statistics.mean(vals)) for (domain, horizon), vals in by_family.items()), key=lambda x: x[2], reverse=True)
    strongest = families[0]
    remaining = [gain for row, gain in zip(tasks, gains) if (row["domain"], row["horizon"]) != strongest[:2]]
    return {"status": "PARTIAL_PASS", "reason": "tracked aggregate CSV only; no task predictions, frozen-route files, or model replay in fresh clone",
            "source_commit": "c63793236d532d2ffdcccd4c7cb0c8e194f96eab",
            "task_rows": len(tasks), "candidate_rows": len(candidates), "numeric_fallback": numeric,
            "text_selected": len(tasks) - numeric, "mean_task_fold_gain_pct": statistics.mean(gains),
            "median_task_fold_gain_pct": statistics.median(gains), "largest_family": {"domain": strongest[0], "horizon": strongest[1], "mean_gain_pct": strongest[2]},
            "mean_gain_excluding_largest_family_pct": statistics.mean(remaining), "task_family_count": len(families),
            "sha256": {p.name: sha256(p) for p in [folder / "task_results.csv", folder / "candidate_results.csv", folder / "summary.json"]}}
