"""无文本审计表与决策半段审计表（第三轮「明确交付物」要求）。

任务书列出的交付物里有「无文本与半段审计表」，此前只做了测试断言、没产出表。
本模块补齐：

  `text_availability_audit.csv`  逐起点：是否有文本、S 贡献、SF 贡献、是否严格为零
  `decision_halves_audit.csv`    两个目标时间半段各自的样本量与损失

两张表都从**主控冻结 Bundle** 生成；无 Bundle 时不产出（不拿合成数据充数）。

用法::

    python audit_tables.py --bundle <目录> --task Agriculture_h12_f1 \\
        --scenario proxy --signature <sig> --candidate N+S+Q --output-dir <目录>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bundle_reader import (  # noqa: E402
    BundleUnavailable, assert_grid, decision_halves_by_target_time, read_frozen_bundle,
)
from candidates import GATE_CANDIDATES  # noqa: E402
from permutation import decision_loss, observed_loss  # noqa: E402
from predict_io import code_commit  # noqa: E402
from train import RUNNABLE_SCENARIOS, to_feature_bundle  # noqa: E402

REPO_ROOT = HERE.parents[1]


def text_availability_rows(model, part, origin_ids) -> list[dict]:
    """逐起点的无文本审计。**关键列是 `contribution_is_zero`** —— 用于核对
    "无文本起点上 S/SF 的贡献严格为零"这一硬要求是否在真实数据上成立。"""
    contrib = model.contributions(part)
    rows = []
    for i, oid in enumerate(origin_ids):
        has_text = bool(part.text_available[i])
        s_val = float(contrib["S"][i].mean()) if "S" in contrib else 0.0
        sf_val = float(contrib["SF"][i].mean()) if "SF" in contrib else 0.0
        rows.append({
            "origin_id": str(oid),
            "origin_index": int(part.origin_index[i]),
            "text_available": int(has_text),
            "s_contribution_mean": s_val,
            "sf_contribution_mean": sf_val,
            "contribution_is_zero": int((s_val == 0.0) and (sf_val == 0.0)),
        })
    return rows


def decision_half_rows(model, halves: dict) -> list[dict]:
    """决策半段审计：样本量与损失分别输出。"""
    rows = []
    for name, part in halves.items():
        rows.append({
            "half": name,
            "n_origins": int(len(part.origin_index)),
            "n_text_available": int(part.text_available.sum()),
            "mean_loss": decision_loss(model, part),
            "origin_index_min": int(part.origin_index.min()) if len(part.origin_index) else "",
            "origin_index_max": int(part.origin_index.max()) if len(part.origin_index) else "",
        })
    return rows


def write_csv(path: Path, rows: list[dict]) -> int:
    if not rows:
        raise ValueError("没有可写出的行")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scenario", required=True, choices=RUNNABLE_SCENARIOS)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--split-spec-sha256", default=None)
    parser.add_argument("--candidate", required=True, choices=sorted(GATE_CANDIDATES))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    try:
        bundle = read_frozen_bundle(args.bundle, args.task, args.scenario, args.signature,
                                    split_spec_sha256=args.split_spec_sha256)
    except BundleUnavailable as exc:
        print(f"BundleUnavailable: {exc}", file=sys.stderr)
        return 3

    for segment in ("train", "calibration", "decision"):
        assert_grid(bundle, segment)

    train = to_feature_bundle(bundle.segment("train"))
    cal = to_feature_bundle(bundle.segment("calibration"))
    decision = to_feature_bundle(bundle.segment("decision"))

    _, model = observed_loss(args.candidate, train, cal, decision)

    n_text = write_csv(out / "text_availability_audit.csv",
                       text_availability_rows(model, decision,
                                              bundle.segment("decision")["origin_id"]))

    masks = decision_halves_by_target_time(bundle, "decision")
    halves = {name: decision.slice(np.flatnonzero(m)) for name, m in masks.items()}
    n_half = write_csv(out / "decision_halves_audit.csv", decision_half_rows(model, halves))

    summary = {
        "candidate": args.candidate, "task": args.task, "scenario": args.scenario,
        "bundle_signature": args.signature, "code_commit": code_commit(REPO_ROOT),
        "text_audit_rows": n_text, "half_audit_rows": n_half,
        "decision_n_origins": int(len(decision.origin_index)),
        "decision_n_text_available": int(decision.text_available.sum()),
        "halves_rule": "按目标时间中点切；跨半段 H 窗口两边都不收",
    }
    (out / "audit_tables_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    print(json.dumps({"status": "completed", "text_rows": n_text, "half_rows": n_half},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
