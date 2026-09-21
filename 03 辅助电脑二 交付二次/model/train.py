"""第二轮训练入口：消费主控冻结 Bundle，输出校准 / 决策段预测。

**测试段预测不在本入口范围内**（契约 `required_before_freeze`：冻结路由前只交
calibration / decision 及 999 次 refit null）。测试真值从不进入本进程。

用法::

    python train.py --bundle <Bundle目录> --task Agriculture_h12_f1 \\
                    --scenario proxy --segments calibration decision \\
                    --candidate N+S+Q --seed 2026 --output-dir <目录>

`--bundle` 必填。**没有默认值、没有 v2smoke 回退** —— 第一轮 P1 第 2 条指出的
"入口锁在旧 v2 数据"在此彻底移除。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from branches import FeatureBundle  # noqa: E402
from bundle_reader import BundleUnavailable, assert_grid, read_frozen_bundle  # noqa: E402
from candidates import ALL_CANDIDATES, BranchResidualCandidate  # noqa: E402
from predict_io import PredictionWriter, code_commit, config_sha256  # noqa: E402

REPO_ROOT = HERE.parents[1]
SCENARIOS = ("proxy", "conservative_lag", "complete_source")


def to_feature_bundle(part: dict[str, np.ndarray]) -> FeatureBundle:
    """把 Bundle 的一段转成模型输入。字段缺失立即失败，不代填。"""
    required = ("numeric_history", "semantic", "quality", "text_available",
                "origin_index", "targets", "targets_standardized")
    missing = [k for k in required if k not in part]
    if missing:
        raise BundleUnavailable(f"Bundle 段缺少必需字段: {missing}")
    return FeatureBundle(
        numeric_history=part["numeric_history"],
        semantic=part["semantic"],
        quality=part["quality"],
        text_available=part["text_available"],
        origin_index=part["origin_index"],
        targets=part["targets"],
        targets_standardized=part["targets_standardized"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True, help="主控冻结 Bundle 目录")
    parser.add_argument("--task", required=True, help="任务 ID，如 Agriculture_h12_f1")
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--signature", default=None, help="期望的 bundle_signature")
    parser.add_argument("--segments", nargs="+", default=["calibration", "decision"],
                        choices=["calibration", "decision"])
    parser.add_argument("--candidate", required=True, choices=sorted(ALL_CANDIDATES))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if not args.signature:
        print("必须显式提供 --signature（不接受从 manifest 回读，避免自证）", file=sys.stderr)
        return 2

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    try:
        bundle = read_frozen_bundle(args.bundle, args.task, args.scenario, args.signature)
    except BundleUnavailable as exc:
        print(f"BundleUnavailable: {exc}", file=sys.stderr)
        return 3

    domain = args.task.split("_")[0]
    fold_id = int(args.task.split("_f")[-1])
    horizon = int(args.task.split("_h")[1].split("_")[0])

    config = {
        "task": args.task, "scenario": args.scenario, "candidate": args.candidate,
        "segments": args.segments, "seed": args.seed,
        "bundle_signature": args.signature, "contract": "team-eval-2.0",
    }
    cfg_hash = config_sha256(config)
    commit = code_commit(REPO_ROOT)
    writer = PredictionWriter()
    manifests: dict[str, dict] = {}

    # 训练段起点数由 Bundle 决定；各段起点网格必须先通过契约校验
    for segment in ("train", "calibration", "decision"):
        assert_grid(bundle, segment)

    train_part = to_feature_bundle(bundle.segment("train"))
    cal_part = to_feature_bundle(bundle.segment("calibration")) if len(bundle.segment("calibration")["origin_id"]) else None

    if cal_part is None:
        print("Bundle 的 calibration 段为空，无法选 α", file=sys.stderr)
        return 4

    model = BranchResidualCandidate(args.candidate)
    model.fit_design(train_part)
    model.select_alphas(train_part, cal_part)

    fit_bundle = FeatureBundle(
        numeric_history=np.r_[train_part.numeric_history, cal_part.numeric_history],
        semantic=np.r_[train_part.semantic, cal_part.semantic],
        quality=np.r_[train_part.quality, cal_part.quality],
        text_available=np.r_[train_part.text_available, cal_part.text_available],
        origin_index=np.r_[train_part.origin_index, cal_part.origin_index],
        targets=np.r_[train_part.targets, cal_part.targets],
        targets_standardized=np.r_[train_part.targets_standardized, cal_part.targets_standardized],
    )
    model.refit(fit_bundle)

    for segment in args.segments:
        part = to_feature_bundle(bundle.segment(segment))
        predictions = model.predict(part)
        writer.extend(
            predictions, origin_id=np.asarray(bundle.segment(segment)["origin_id"]),
            origin_index=part.origin_index, task_id=args.task, fold_id=fold_id,
            segment=segment, scenario=args.scenario, candidate_id=args.candidate,
            seed=args.seed, bundle_signature=args.signature,
            config_sha256_value=cfg_hash, commit=commit,
        )
        contrib = model.contributions(part)
        manifests[segment] = {
            "n_origins": int(len(part.origin_index)),
            "n_text_available": int(part.text_available.sum()),
            "contribution_abs_mean": {k: float(np.abs(v).mean()) for k, v in contrib.items()},
        }

    n_rows = writer.write(out / f"{args.candidate.replace('+', '_')}_{args.scenario}_predictions.csv")

    manifest = {
        "config": config, "config_sha256": cfg_hash, "code_commit": commit,
        "bundle_signature": args.signature, "task": args.task, "scenario": args.scenario,
        "candidate": args.candidate, "status": "completed",
        "alpha_by_group": model.alpha_by_group,
        "branch_widths": {b.name: b.width for b in model.branches},
        "s_degraded": model.s_degraded,
        "dropped_zero_variance": {b.name: getattr(b.scaler, "n_dropped", None)
                                  for b in model.branches if hasattr(b, "scaler")},
        "segments": manifests, "rows_written": n_rows,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "note": "只含 calibration/decision；测试预测待主控冻结路由后另行生成",
    }
    (out / f"{args.candidate.replace('+', '_')}_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    print(json.dumps({"status": "completed", "candidate": args.candidate,
                      "rows": n_rows, "seconds": manifest["wall_seconds"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
