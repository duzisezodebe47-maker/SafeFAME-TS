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
from bundle_reader import (  # noqa: E402
    BundleUnavailable, assert_grid, read_frozen_bundle, segment_bounds, sha256_file,
)
from candidates import ALL_CANDIDATES, BranchResidualCandidate  # noqa: E402
from predict_io import (  # noqa: E402
    PredictionWriter, code_commit, config_sha256, weight_hash,
)

REPO_ROOT = HERE.parents[1]

# 第三轮 A.3：`complete_source` 是**纯数据审计情景**（协议 note 里其真实覆盖为 0），
# 不作为训练入口的可运行情景提供给用户。
RUNNABLE_SCENARIOS = ("proxy", "conservative_lag")


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
    parser.add_argument("--scenario", required=True, choices=RUNNABLE_SCENARIOS)
    parser.add_argument("--signature", required=True, help="期望的 bundle_signature（必填）")
    parser.add_argument("--split-spec", type=Path, required=True,
                        help="主控冻结协议文件路径；本入口自己算字节哈希（P1-3 必填）")
    parser.add_argument("--segments", nargs="+", default=["calibration", "decision"],
                        choices=["calibration", "decision"])
    parser.add_argument("--candidate", required=True, choices=sorted(ALL_CANDIDATES))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    try:
        bundle = read_frozen_bundle(args.bundle, args.task, args.scenario,
                                    args.signature, args.split_spec)
        bounds = segment_bounds(args.split_spec, args.task)
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

    # 各段网格必须先通过契约校验 —— **边界与 H 从冻结 spec 读取并传入**（P1-4）
    horizon = int(args.task.split("_h")[1].split("_")[0])
    grid_stats = {s: assert_grid(bundle, s, bounds[s], horizon)
                  for s in ("train", "calibration", "decision")}

    train_part = to_feature_bundle(bundle.segment("train"))
    cal_part = to_feature_bundle(bundle.segment("calibration")) if len(bundle.segment("calibration")["origin_id"]) else None

    if cal_part is None:
        print("Bundle 的 calibration 段为空，无法选 α", file=sys.stderr)
        return 4

    model = BranchResidualCandidate(args.candidate)
    model.fit_design(train_part)
    model.select_alphas(train_part, cal_part)

    fit_bundle = merge_bundles(train_part, cal_part)
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
        # 权重哈希：predict_test.py 用它核对"重拟合的权重与选择期一致"（A.6）
        "weight_hash": weight_hash(model.weights),
        "branch_widths": {b.name: b.width for b in model.branches},
        "s_degraded": model.s_degraded,
        "dropped_zero_variance": {b.name: getattr(b.scaler, "n_dropped", None)
                                  for b in model.branches if hasattr(b, "scaler")},
        "segments": manifests, "rows_written": n_rows,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "split_spec_sha256": sha256_file(args.split_spec),
        "grid": grid_stats,
        "note": "只含 calibration/decision；测试预测待主控冻结路由后另行生成",
    }
    (out / f"{args.candidate.replace('+', '_')}_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    print(json.dumps({"status": "completed", "candidate": args.candidate,
                      "rows": n_rows, "seconds": manifest["wall_seconds"]}, ensure_ascii=False))
    return 0


def merge_bundles(*parts) -> FeatureBundle:
    """把多个 FeatureBundle 切片按行拼接。**顺序即传入顺序**，不重排。"""
    return FeatureBundle(
        numeric_history=np.concatenate([p.numeric_history for p in parts]),
        semantic=np.concatenate([p.semantic for p in parts]),
        quality=np.concatenate([p.quality for p in parts]),
        text_available=np.concatenate([p.text_available for p in parts]),
        origin_index=np.concatenate([p.origin_index for p in parts]),
        targets=np.concatenate([p.targets for p in parts]),
        targets_standardized=np.concatenate([p.targets_standardized for p in parts]),
    )


if __name__ == "__main__":
    raise SystemExit(main())
