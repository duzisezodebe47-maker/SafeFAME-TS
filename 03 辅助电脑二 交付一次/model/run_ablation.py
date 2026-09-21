"""跑完整候选族并汇总消融表。

任务书 §4.4 要求给出 ``N``、``N+Q``、``N+S``、``N+S+Q``、``N+S+Q+F`` 和
数值宽度对照的可比结果。每个候选逐起点写出预测（§4.3），并汇总成一张对照表。

**当前数据源为 v2 特征、单一领域单跨度 —— 这是工程冒烟级对照，不是科研结论。**
正式的 v4 全量消融需等待辅助电脑一交付 `data_processed/v4/`。

用法::

    python run_ablation.py --domain Climate --horizon 4 --output-dir <目录>
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
sys.path.insert(0, str(HERE))

from candidates import CANDIDATES, BranchResidualCandidate  # noqa: E402
from predict_io import PredictionSet, code_commit, hash_config, hash_features, validate_alignment  # noqa: E402
from train import build_bundle, split_origins, ROOT, load_time_ordered_frame, NUMERIC_ROOT  # noqa: E402


def run_one(candidate: str, domain: str, horizon: int, input_len: int,
            fold: int, seed: int, out: Path, config: dict) -> dict:
    """跑一个候选，写预测与清单，返回汇总行。"""
    frame, _ = load_time_ordered_frame(NUMERIC_ROOT / domain / f"{domain}.csv")
    parts = split_origins(len(frame), input_len, horizon)

    train, y_train = build_bundle(domain, input_len, horizon, parts["train"])
    cal, y_cal = build_bundle(domain, input_len, horizon, parts["calibration"])
    test, y_test = build_bundle(domain, input_len, horizon, parts["test"])

    for name, part in parts.items():
        bundle = {"train": train, "calibration": cal, "test": test}[name]
        validate_alignment(bundle.origins, np.array(list(part), dtype=np.int64), name)

    feature_hash = hash_features(train)
    cfg = dict(config, candidate=candidate, branches=list(CANDIDATES[candidate]))
    cfg_hash = hash_config(cfg)

    started = time.perf_counter()
    model = BranchResidualCandidate(CANDIDATES[candidate])
    model.fit_design(train)
    model.select_alphas(train, cal, y_train, y_cal)
    fit_bundle = type(train)(
        x=np.r_[train.x, cal.x], report=np.r_[train.report, cal.report],
        search=np.r_[train.search, cal.search], quality=np.r_[train.quality, cal.quality],
        origins=np.r_[train.origins, cal.origins],
    )
    model.refit(fit_bundle, np.r_[y_train, y_cal])
    predictions = model.predict(test)
    elapsed = time.perf_counter() - started

    candidate_dir = out / candidate.replace("+", "_")
    candidate_dir.mkdir(parents=True, exist_ok=True)

    ps = PredictionSet()
    ps.extend_grid(
        test.origins, predictions, task_id=f"{domain}_h{horizon}", fold_id=fold,
        candidate_id=candidate, seed=seed, config_hash=cfg_hash,
        feature_hash=feature_hash, commit=code_commit(ROOT),
    )
    ps.write(candidate_dir / "predictions.csv")

    contributions = model.contributions(test)
    manifest = {
        "config": cfg, "config_hash": cfg_hash, "feature_hash": feature_hash,
        "status": "completed", "rows_written": len(ps.records),
        "n_parameters": model.n_parameters,
        "alpha_by_group": model.alpha_by_group,
        "branch_widths": {b.name: b.width for b in model.branches},
        "dropped_zero_variance": {
            b.name: getattr(b.scaler, "n_dropped", None)
            for b in model.branches if hasattr(b, "scaler")
        },
        "resources": {"wall_seconds": round(elapsed, 3), "device": "cpu"},
    }
    (candidate_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    mse = float(np.mean((predictions - y_test) ** 2))
    last_mse = float(np.mean((test.x[:, -1, None] - y_test) ** 2))
    return {
        "candidate": candidate,
        "branches": "+".join(CANDIDATES[candidate]),
        "n_features": sum(b.width for b in model.branches),
        "n_parameters": model.n_parameters,
        "alpha_by_group": json.dumps(model.alpha_by_group, ensure_ascii=False),
        "test_mse": round(mse, 6),
        "last_anchor_mse": round(last_mse, 6),
        "gain_vs_last_pct": round(100 * (1 - mse / last_mse), 3),
        "n_origins": len(test.origins),
        "wall_seconds": round(elapsed, 3),
        "s_contribution_abs_mean": round(float(np.abs(contributions["S"]).mean()), 6)
        if "S" in contributions else 0.0,
        "q_contribution_abs_mean": round(float(np.abs(contributions["Q"]).mean()), 6)
        if "Q" in contributions else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="Climate")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--input-len", type=int, default=52)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    base_config = {
        "domain": args.domain, "horizon": args.horizon, "input_len": args.input_len,
        "fold": args.fold, "seed": args.seed, "data_source": "v2smoke",
        "split_source": "v2 frozen 70/10/20 (docs/10) — 主控协议未冻结前的临时取值",
        "note": "工程冒烟级对照，单一领域单跨度，不作为科研结论",
    }

    rows: list[dict] = []
    failures: list[dict] = []
    for candidate in CANDIDATES:
        try:
            rows.append(run_one(candidate, args.domain, args.horizon, args.input_len,
                                args.fold, args.seed, out, base_config))
            print(f"  ok  {candidate}")
        except Exception as exc:  # noqa: BLE001 — 失败留证据
            failures.append({"candidate": candidate, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  FAIL {candidate}: {exc}", file=sys.stderr)

    if rows:
        with (out / "ablation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    (out / "ablation_manifest.json").write_text(
        json.dumps({"base_config": base_config, "candidates": list(CANDIDATES),
                    "completed": len(rows), "failures": failures}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    print(f"\n完成 {len(rows)}/{len(CANDIDATES)} 个候选，失败 {len(failures)} 个")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
