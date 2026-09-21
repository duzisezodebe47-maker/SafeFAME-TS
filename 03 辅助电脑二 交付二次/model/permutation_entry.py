"""决策段置换入口（第二轮 P1 第 5 条）。

对拟进入正式门控的候选跑 999 次逐行错位置换，**每次重建 PCA / 各分支尺度 /
SF 交互尺度 / α 选择**（见 `permutation.py` 顶部说明）。另跑循环移位作敏感性诊断，
**不回写主门控**。

用法::

    python permutation_entry.py --bundle <Bundle目录> --task Agriculture_h12_f1 \\
        --scenario proxy --signature <sig> --candidate N+S+Q \\
        --nulls 999 --output-dir <目录>

`--nulls` 的**实际值会被如实写入** summary 的 `requested` 字段；未跑满 999 时
`p_value` 为 `null`，不伪填。
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
from candidates import GATE_CANDIDATES  # noqa: E402
from permutation import (  # noqa: E402
    circular_shift_null, decision_loss, observed_loss, row_permutation_null, write_null,
)
from predict_io import code_commit, config_sha256  # noqa: E402
from train import to_feature_bundle  # noqa: E402

REPO_ROOT = HERE.parents[1]
SCENARIOS = ("proxy", "conservative_lag", "complete_source")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--candidate", required=True, choices=sorted(GATE_CANDIDATES),
                        help="只允许门控候选；诊断消融不进门槛")
    parser.add_argument("--nulls", type=int, default=999)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--circular-block", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    try:
        bundle = read_frozen_bundle(args.bundle, args.task, args.scenario, args.signature)
    except BundleUnavailable as exc:
        print(f"BundleUnavailable: {exc}", file=sys.stderr)
        return 3

    for segment in ("train", "calibration", "decision"):
        assert_grid(bundle, segment)

    train = to_feature_bundle(bundle.segment("train"))
    cal = to_feature_bundle(bundle.segment("calibration"))
    decision = to_feature_bundle(bundle.segment("decision"))

    if not len(cal.origin_index) or not len(decision.origin_index):
        print("calibration 或 decision 段为空，无法跑置换", file=sys.stderr)
        return 4

    # 决策段两个不重叠半段，样本量与损失分别输出（主控口径）
    half = len(decision.origin_index) // 2
    halves = {"first_half": decision.slice(np.arange(half)),
              "second_half": decision.slice(np.arange(half, len(decision.origin_index)))}

    observed, observed_model = observed_loss(args.candidate, train, cal, decision)

    print(f"  {args.candidate}: 观测损失 {observed:.6f}，开始 {args.nulls} 次置换…", file=sys.stderr)
    null = row_permutation_null(args.candidate, train, cal, decision,
                                count=args.nulls, seed=args.seed)
    write_null(out / "null_scores.csv", null, observed)

    print(f"  {args.candidate}: 开始循环移位诊断…", file=sys.stderr)
    circular = circular_shift_null(args.candidate, train, cal, decision,
                                   count=args.nulls, seed=args.seed, block=args.circular_block)
    write_null(out / "null_scores_circular.csv", circular, observed)

    summary = {
        "candidate": args.candidate, "task": args.task, "scenario": args.scenario,
        "bundle_signature": args.signature,
        "config_sha256": config_sha256({"candidate": args.candidate, "task": args.task,
                                        "scenario": args.scenario, "nulls": args.nulls}),
        "code_commit": code_commit(REPO_ROOT),
        # 实际请求次数如实记录 —— 不写死常量
        "requested_permutations": args.nulls,
        "completed_permutations": null.observed_count,
        "failures": len(null.failures),
        "observed_decision_loss": observed,
        "p_value": null.p_value(observed),
        "p_value_note": ("置换未跑满，p 值不可用" if null.observed_count < args.nulls
                         else "经验单侧 p"),
        "circular": {"completed": circular.observed_count, "failures": len(circular.failures),
                     "p_value": circular.p_value(observed),
                     "role": "敏感性诊断，不回写主门控"},
        # 决策段两个不重叠半段的样本量与损失分别输出（主控口径）
        "decision_halves": {
            name: {"n_origins": int(len(part.origin_index)),
                   "mean_loss": decision_loss(observed_model, part)}
            for name, part in halves.items()
        },
        "wall_seconds": round(time.perf_counter() - started, 2),
    }
    (out / "permutation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    p = summary["p_value"]
    print(json.dumps({"status": "completed", "candidate": args.candidate,
                      "completed": null.observed_count, "failures": len(null.failures),
                      "p_value": p}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
