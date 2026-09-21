"""Build a tiny artificial integration fixture. Never use it as research evidence."""
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
base.mkdir(parents=True, exist_ok=True)
task = dict(task_id="SMOKE_H1", fold_id=1, domain="synthetic", horizon=1, n_rows=50,
            train_end=20, cal_end=28, dec_end=40, test_end=50, seasonal_period=2,
            snapshot_sha256="a"*64, feature_manifest_sha256="b"*64)
(base / "task.json").write_text(json.dumps(task), encoding="utf-8")
for name, series in (("selection", (("cal", [20, 21]), ("dec", [28, 29, 35, 36]))),
                     ("test", (("test", [40, 41, 42, 43]),))):
    samples, predictions = [], []
    for segment, origins in series:
        for origin in origins:
            samples.append(dict(task_id=task["task_id"], fold_id=1, origin_id=origin,
                                segment=segment, horizon=1, target=[1.0], cutoff_index=origin-1,
                                target_start_index=origin, snapshot_sha256=task["snapshot_sha256"],
                                feature_manifest_sha256=task["feature_manifest_sha256"], preprocessing_fit_end=20))
            for candidate, value in (("Last", 0.0), ("SeasonalNaive", -.1), ("AR-Ridge", -.2),
                                     ("DLinear-M", -.3), ("PatchTST", -.4),
                                     ("semantic_residual", 1.0), ("frequency_residual", -1.0)):
                predictions.append(dict(task_id=task["task_id"], fold_id=1, origin_id=origin,
                                        segment=segment, candidate_id=candidate, seed=2026,
                                        prediction=[value], config_sha256=(candidate[0]*64),
                                        feature_manifest_sha256=task["feature_manifest_sha256"]))
    for filename, rows in ((f"{name}_samples.jsonl", samples), (f"{name}_predictions.jsonl", predictions)):
        (base / filename).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
nulls = [dict(task_id=task["task_id"], fold_id=1, candidate_id=candidate,
              decision_null_mse=[.5]*999, method="row-permutation-refit", seed=2026,
              source_sha256="c"*64) for candidate in ("semantic_residual", "frequency_residual")]
(base / "row_nulls.jsonl").write_text("".join(json.dumps(row) + "\n" for row in nulls), encoding="utf-8")
