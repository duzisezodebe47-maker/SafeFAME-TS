"""Validate aligned forecasts and freeze a route before looking at test truth."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path


class EvidenceError(ValueError):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvidenceError(f"{path}:{line_no}: invalid JSON") from exc


def need(row, keys, label):
    missing = set(keys) - set(row)
    if missing:
        raise EvidenceError(f"{label}: missing {sorted(missing)}")


def finite_vector(values, horizon, label):
    if not isinstance(values, list) or len(values) != horizon or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values
    ):
        raise EvidenceError(f"{label}: expected {horizon} finite numbers")


def boundaries(task, spec=None):
    need(task, ["task_id", "fold_id", "horizon", "input_len", "n_rows", "train_end", "cal_end", "dec_end", "test_end", "snapshot_sha256", "feature_manifest_sha256", "seasonal_period"], "task")
    n, tr, ca, de, te = (int(task[x]) for x in ("n_rows", "train_end", "cal_end", "dec_end", "test_end"))
    if not 0 < tr < ca < de < te <= n or int(task["horizon"]) < 1 or int(task["input_len"]) < 1:
        raise EvidenceError("task: invalid ordered boundaries")
    if spec is not None and "fold_boundaries" in spec:
        fold = int(task["fold_id"])
        try:
            expected = [int(n * fraction) for fraction in spec["fold_boundaries"][fold - 1]]
        except (IndexError, TypeError, ValueError) as exc:
            raise EvidenceError("task: unknown fold") from exc
        if fold < 1 or [tr, ca, de, te] != expected:
            raise EvidenceError(f"task: split differs from frozen fold {fold}: {[tr, ca, de, te]} != {expected}")
    return {"train": (0, tr), "cal": (tr, ca), "dec": (ca, de), "test": (de, te)}, tr


def validate(task, samples, predictions, require_test=False, spec=None):
    spans, train_end = boundaries(task, spec)
    h = int(task["horizon"])
    by_sample = {}
    for row in samples:
        need(row, ["task_id", "fold_id", "origin_id", "segment", "horizon", "target", "cutoff_index", "target_start_index", "snapshot_sha256", "feature_manifest_sha256", "preprocessing_fit_end"], "sample")
        if (row["task_id"], row["fold_id"]) != (task["task_id"], task["fold_id"]):
            raise EvidenceError("sample: wrong task/fold")
        if row["snapshot_sha256"] != task["snapshot_sha256"] or row["feature_manifest_sha256"] != task["feature_manifest_sha256"]:
            raise EvidenceError("sample: snapshot or feature hash mismatch")
        segment = row["segment"]
        if segment not in spans:
            raise EvidenceError("sample: unknown segment")
        origin = int(row["origin_id"])
        lo, hi = spans[segment]
        if origin < lo or origin + h > hi or origin != int(row["target_start_index"]):
            raise EvidenceError(f"sample: target window crosses {segment} boundary at {origin}")
        if int(row["cutoff_index"]) >= origin or int(row["preprocessing_fit_end"]) > train_end:
            raise EvidenceError("sample: future feature/preprocessing leakage")
        if int(row["horizon"]) != h:
            raise EvidenceError("sample: horizon mismatch")
        finite_vector(row["target"], h, "sample target")
        key = (segment, origin)
        if key in by_sample:
            raise EvidenceError(f"sample: duplicate origin {key}")
        by_sample[key] = row
    if not by_sample or (require_test and not any(k[0] == "test" for k in by_sample)):
        raise EvidenceError("sample: no required evidence")
    by_prediction = {}
    candidate_configs = {}
    for row in predictions:
        need(row, ["task_id", "fold_id", "origin_id", "segment", "candidate_id", "seed", "prediction", "config_sha256", "feature_manifest_sha256"], "prediction")
        if (row["task_id"], row["fold_id"]) != (task["task_id"], task["fold_id"]):
            raise EvidenceError("prediction: wrong task/fold")
        if row["feature_manifest_sha256"] != task["feature_manifest_sha256"]:
            raise EvidenceError("prediction: feature hash mismatch")
        skey = (row["segment"], int(row["origin_id"]))
        if skey not in by_sample:
            raise EvidenceError(f"prediction: unmatched origin {skey}")
        finite_vector(row["prediction"], h, "prediction")
        candidate = str(row["candidate_id"])
        if candidate in candidate_configs and candidate_configs[candidate] != row["config_sha256"]:
            raise EvidenceError("prediction: mixed config versions")
        candidate_configs[candidate] = row["config_sha256"]
        key = (*skey, candidate, int(row["seed"]))
        if key in by_prediction:
            raise EvidenceError(f"prediction: duplicate {key}")
        by_prediction[key] = row["prediction"]
    if not by_prediction:
        raise EvidenceError("prediction: empty")
    # Every candidate/seed must cover the same origin set within each supplied segment.
    for candidate in candidate_configs:
        seeds = {k[3] for k in by_prediction if k[2] == candidate}
        for seed in seeds:
            covered = {(k[0], k[1]) for k in by_prediction if k[2:] == (candidate, seed)}
            expected = {k for k in by_sample if k[0] in ("cal", "dec", "test")}
            if covered != expected:
                raise EvidenceError(f"prediction: incomplete origin coverage for {candidate}/{seed}")
    return by_sample, by_prediction, candidate_configs


def ensemble(predictions, segment, candidate, origins):
    seeds = sorted({key[3] for key in predictions if key[0] == segment and key[2] == candidate})
    if not seeds:
        raise EvidenceError(f"missing {segment} predictions: {candidate}")
    return [[sum(predictions[(segment, origin, candidate, seed)][j] for seed in seeds) / len(seeds)
             for j in range(len(predictions[(segment, origin, candidate, seeds[0])]))] for origin in origins]


def losses(truth, forecast):
    return [sum((a - b) ** 2 for a, b in zip(y, p)) / len(y) for y, p in zip(truth, forecast)]


def mse(truth, forecast):
    values = losses(truth, forecast)
    return sum(values) / len(values)


def mae(truth, forecast):
    return sum(sum(abs(a - b) for a, b in zip(y, p)) / len(y) for y, p in zip(truth, forecast)) / len(truth)


def segment_data(samples, predictions, segment, candidate):
    origins = sorted(k[1] for k in samples if k[0] == segment)
    if not origins:
        raise EvidenceError(f"no {segment} samples")
    return origins, [samples[(segment, o)]["target"] for o in origins], ensemble(predictions, segment, candidate, origins)


def freeze_route(task, samples, predictions, null_rows, spec):
    """No test target access: only cal/dec keys are consulted here."""
    numeric = spec["numeric_candidates"]
    text = spec["candidate_variants"]
    present = {k[2] for k in predictions}
    if not set(numeric + text) <= present:
        raise EvidenceError(f"missing candidate paths: {sorted(set(numeric + text) - present)}")
    cal = {}
    for candidate in numeric:
        _, y, p = segment_data(samples, predictions, "cal", candidate)
        cal[candidate] = mse(y, p)
    fallback = min(numeric, key=lambda c: (cal[c], c))
    origins, truth, fallback_pred = segment_data(samples, predictions, "dec", fallback)
    fallback_mse = mse(truth, fallback_pred)
    middle = (int(task["cal_end"]) + int(task["dec_end"])) // 2
    h = int(task["horizon"])
    first = [i for i, o in enumerate(origins) if o + h <= middle]
    second = [i for i, o in enumerate(origins) if o >= middle]
    nulls = {}
    for row in null_rows:
        need(row, ["task_id", "fold_id", "candidate_id", "decision_null_mse", "method", "seed", "source_sha256"], "row null")
        if (row["task_id"], row["fold_id"], row["method"]) != (task["task_id"], task["fold_id"], "row-permutation-refit"):
            raise EvidenceError("row null: wrong task or method")
        candidate = row["candidate_id"]
        if candidate in nulls:
            raise EvidenceError("row null: duplicate candidate")
        values = row["decision_null_mse"]
        finite_vector(values, int(spec["selection"]["row_null_draws"]), "row null scores")
        nulls[candidate] = values
    if set(nulls) != set(text):
        raise EvidenceError("row null: missing or unexpected candidate")
    diagnostics = {}
    for candidate in text:
        _, _, pred = segment_data(samples, predictions, "dec", candidate)
        score = mse(truth, pred)
        pvalue = (1 + sum(x <= score for x in nulls[candidate])) / (len(nulls[candidate]) + 1)
        wins = bool(first and second) and all(
            mse([truth[i] for i in idx], [pred[i] for i in idx]) <
            mse([truth[i] for i in idx], [fallback_pred[i] for i in idx])
            for idx in (first, second)
        )
        diagnostics[candidate] = {"decision_mse": score, "row_p": pvalue, "segment_wins": wins,
                                  "eligible": score < fallback_mse and wins and pvalue <= spec["selection"]["p_threshold"]}
    eligible = [x for x in text if diagnostics[x]["eligible"]]
    selected = min(eligible, key=lambda c: (diagnostics[c]["decision_mse"], c)) if eligible else "numeric_fallback"
    return {"task_id": task["task_id"], "fold_id": task["fold_id"], "fallback": fallback,
            "selected": selected, "calibration_mse": cal, "decision_fallback_mse": fallback_mse,
            "candidates": diagnostics, "selection_data_segments": ["cal", "dec"]}


def moving_block_ci(delta, block, draws, seed):
    if len(delta) < block or draws < 1:
        raise EvidenceError("not enough origins for block interval")
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        sample = []
        while len(sample) < len(delta):
            start = rng.randrange(len(delta) - block + 1)
            sample.extend(delta[start:start + block])
        means.append(sum(sample[:len(delta)]) / len(delta))
    means.sort()
    def quantile(q):
        pos = (len(means) - 1) * q
        a = int(pos)
        return means[a] + (means[min(a + 1, len(means) - 1)] - means[a]) * (pos - a)
    return [quantile(.025), quantile(.975)]


def evaluate_test(task, samples, predictions, route, spec):
    """Call only after route is serialized. Test truth enters for metrics, never selection."""
    chosen = route["fallback"] if route["selected"] == "numeric_fallback" else route["selected"]
    origins, truth, base = segment_data(samples, predictions, "test", route["fallback"])
    _, _, selected = segment_data(samples, predictions, "test", chosen)
    base_losses, selected_losses = losses(truth, base), losses(truth, selected)
    delta = [a - b for a, b in zip(base_losses, selected_losses)]
    block = max(int(task["horizon"]), min(int(task["seasonal_period"]), 24))
    baseline_mse, selected_mse = sum(base_losses) / len(base_losses), sum(selected_losses) / len(selected_losses)
    return {"task_id": task["task_id"], "fold_id": task["fold_id"], "selected": route["selected"],
            "fallback": route["fallback"], "test_windows": len(origins), "test_origin_ids": origins,
            "fallback_mse": baseline_mse, "selected_mse": selected_mse, "selected_mae": mae(truth, selected),
            "gain_pct": 100 * (1 - selected_mse / baseline_mse) if baseline_mse else None,
            "paired_loss_delta": sum(delta) / len(delta), "block_length": block,
            "delta_ci95": moving_block_ci(delta, block, spec["uncertainty"]["draws"], spec["uncertainty"]["seed"])}
