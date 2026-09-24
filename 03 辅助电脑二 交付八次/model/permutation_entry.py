"""决策段置换入口（第三轮 A.4 / A.5 修复版）。

修复要点：:

  A.4  逐次记录 (iteration, seed, loss | error)，不再错配；
       requested 与 successful 均达 999 才计算 p。
  A.5  决策半段按**目标时间边界**切，跨半段的 H 窗口剔除；
       `--circular-block` 真正作为移位块长度使用。
       情景只允许 proxy / conservative_lag —— complete_source 是纯数据审计情景。

用法::

    python permutation_entry.py --bundle <目录> --task Agriculture_h12_f1 \\
        --scenario proxy --signature <sig> --candidate N+S+Q \\
        --nulls 999 --output-dir <目录>
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

from bundle_reader import (  # noqa: E402
    BundleUnavailable, assert_grid, decision_halves, read_frozen_bundle,
    segment_bounds, sha256_file, task_seasonal_period,
)
from candidates import GATE_CANDIDATES  # noqa: E402
from isolated_package import (  # noqa: E402
    PackageUnavailable, add_input_args, open_input,
)
from permutation import (  # noqa: E402
    ROW_PERMUTATIONS, circular_shift_null, decision_loss, observed_loss,
    row_permutation_null, write_null,
)
from predict_io import code_commit, config_sha256, warn_if_dirty  # noqa: E402
from runtime_profile import cpu_seconds, script_entry_snapshot  # noqa: E402
from train import RUNNABLE_SCENARIOS, to_feature_bundle  # noqa: E402

REPO_ROOT = HERE.parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    add_input_args(parser)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scenario", required=True, choices=RUNNABLE_SCENARIOS)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--split-spec", type=Path, required=True,
                        help="主控冻结协议文件路径；本入口自己算字节哈希（P1-3 必填）")
    parser.add_argument("--candidate", required=True, choices=sorted(GATE_CANDIDATES),
                        help="只允许门控候选；诊断消融不进门槛")
    parser.add_argument("--nulls", type=int, default=ROW_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--circular-block", type=int, default=7,
                        help="循环移位的真实块长度（A.5）")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    cpu_started = cpu_seconds()

    try:
        # 第八轮：置换同样优先走隔离包（决策段从不涉及 test 真值）；
        # 走正式 Bundle 时用 strict 隔离（test 行从不 materialize）
        bundle, input_kind = open_input(args, args.task, args.scenario,
                                        args.signature, args.split_spec)
        bounds = segment_bounds(args.split_spec, args.task)
    except (BundleUnavailable, PackageUnavailable) as exc:
        print(f"InputUnavailable: {exc}", file=sys.stderr)
        return 3

    # 协议里登记的季节周期（第六轮修复 3）。本候选是分支残差模型、不消费它，
    # 只作协议取值留痕；未登记即拒绝，避免记录一个不存在的协议值。
    try:
        seasonal_period = task_seasonal_period(args.split_spec, args.task)
    except BundleUnavailable as exc:
        print(f"BundleUnavailable: {exc}", file=sys.stderr)
        return 3

    horizon = int(args.task.split("_h")[1].split("_")[0])
    try:
        grid_stats = {s: assert_grid(bundle, s, bounds[s], horizon)
                      for s in ("train", "calibration", "decision")}
    except AssertionError as exc:
        print(f"GridRejected: {exc}", file=sys.stderr)
        return 5

    train = to_feature_bundle(bundle.segment("train"))
    cal = to_feature_bundle(bundle.segment("calibration"))
    decision = to_feature_bundle(bundle.segment("decision"))

    if not len(cal.origin_index) or not len(decision.origin_index):
        print("calibration 或 decision 段为空，无法跑置换", file=sys.stderr)
        return 4

    # 半段按主控规则切：middle=(cal_end+dec_end)//2（P1-5），跨界窗口两边都不收
    half_info = decision_halves(bundle, bounds, horizon)
    halves = {name: decision.slice(np.flatnonzero(half_info[name]["mask"]))
              for name in ("first_half", "second_half")}

    observed, observed_model = observed_loss(args.candidate, train, cal, decision)

    print(f"  {args.candidate}: 观测损失 {observed:.6f}，开始 {args.nulls} 次置换…",
          file=sys.stderr)
    null = row_permutation_null(args.candidate, train, cal, decision,
                                count=args.nulls, seed=args.seed)
    write_null(out / "null_scores.csv", null, observed)

    print(f"  {args.candidate}: 循环移位诊断（块长 {args.circular_block}）…", file=sys.stderr)
    circular = circular_shift_null(args.candidate, train, cal, decision, count=args.nulls,
                                   seed=args.seed, circular_block=args.circular_block)
    write_null(out / "null_scores_circular.csv", circular, observed)

    reason = null.unavailable_reason()
    summary = {
        "candidate": args.candidate, "task": args.task, "scenario": args.scenario,
        "bundle_signature": args.signature,
        "config_sha256": config_sha256({"candidate": args.candidate, "task": args.task,
                                        "scenario": args.scenario, "nulls": args.nulls}),
        "code_commit": code_commit(REPO_ROOT),
        "requested": null.requested,
        "successful": len(null.successful),
        "failed": len(null.failures),
        "contract_minimum": ROW_PERMUTATIONS,
        "observed_decision_loss": observed,
        "p_value": null.p_value(observed),
        "p_value_note": reason or "经验单侧 p（requested 与 successful 均达 999）",
        "circular": {"block": args.circular_block,
                     "successful": len(circular.successful),
                     "failed": len(circular.failures),
                     "p_value": circular.p_value(observed),
                     "role": "敏感性诊断，不回写主门控"},
        # 半段按目标时间边界切，各半段的样本量与损失分别输出
        "decision_halves": {
            name: {"n_origins": int(len(part.origin_index)),
                   "mean_loss": decision_loss(observed_model, part)}
            for name, part in halves.items()
        },
        "decision_halves_rule": half_info["rule"],
        "decision_halves_middle": half_info["middle"],
        "decision_halves_kept": {n: half_info[n]["kept"] for n in ("first_half", "second_half")},
        "decision_halves_excluded": {n: half_info[n]["excluded"]
                                     for n in ("first_half", "second_half")},
        "split_spec_sha256": sha256_file(args.split_spec),
        "grid": grid_stats,
        "wall_seconds": round(time.perf_counter() - started, 2),
        # 第六轮补：CPU / 内存实测（任务书要求交付里附实测值，此前只有墙钟）
        "runtime": script_entry_snapshot(started, cpu_started),
        # 第六轮补：只写 HEAD 会在改动未提交时把 provenance 记成另一个提交
        "code_provenance": warn_if_dirty(REPO_ROOT, HERE),
        "input_kind": input_kind,
        "test_isolation": bundle.isolation,
        # 第六轮修复 3 的可追溯性：把协议里的季节周期记进产物。
        # 本候选是分支残差模型（N/S/Q/SF），**不消费季节周期**；周期只影响主控
        # 数值基线的 SeasonalNaive。这里记录以便主控核对协议取值，并写明适用范围。
        "seasonal_period": {
            "value": seasonal_period,
            "source": f"{Path(args.split_spec).name}::seasonal_periods",
            "applies_to": "numeric_baselines/SeasonalNaive（本置换候选不经过该路径）",
        },
        # 置换协议本身也要可追溯：次数、种子规则、循环移位块长
        "permutation_config": {
            "nulls_requested": args.nulls,
            "seed_base": args.seed,
            "seed_rule": "local_seed = seed_base * 1000 + iteration",
            "circular_block": args.circular_block,
            "refit_each_iteration": True,
            "refit_scope": "PCA / 各分支标度器 / 交互尺度 / α 选择",
            "loss": "决策段 MSE（标准化坐标）",
        },
    }
    (out / "permutation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    print(json.dumps({"status": "completed", "candidate": args.candidate,
                      "successful": summary["successful"], "failed": summary["failed"],
                      "p_value": summary["p_value"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
