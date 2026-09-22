"""测试段预测入口（第三轮 A.6）。

**只有在主控冻结路由之后**才允许运行。本入口：

  - 读取冻结路由文件，要求其 `status == "frozen"` 且带路由哈希；
  - 读取冻结 Bundle，核对签名与 `split_spec_sha256`；
  - 用**选择期已冻结的 α** 在 train+calibration+decision 上重拟合，
    并核对重拟合后的**权重哈希**与选择期 `run_manifest.json` 记录的一致；
  - **从不读取测试真值**做任何选择 —— 只把预测写出去。

拒绝条件（任一即退出，不静默放行）：路由未冻结、路由哈希不符、Bundle 签名不符、
选择期配置哈希不符、权重哈希不符、测试段起点网格不合法。

用法::

    python predict_test.py --bundle <目录> --task Agriculture_h12_f1 \\
        --scenario proxy --signature <sig> --split-spec-sha256 <sha> \\
        --route <route.json> --selection-manifest <选择期 run_manifest.json> \\
        --candidate N+S+Q --output-dir <目录>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bundle_reader import BundleUnavailable, assert_grid, read_frozen_bundle  # noqa: E402
from candidates import GATE_CANDIDATES, BranchResidualCandidate  # noqa: E402
from predict_io import PredictionWriter, code_commit, weight_hash  # noqa: E402
from train import RUNNABLE_SCENARIOS, to_feature_bundle  # noqa: E402

REPO_ROOT = HERE.parents[1]


class RouteNotFrozen(RuntimeError):
    """路由未冻结或与预期不符。**不得降级为重拟合或跳过。**"""


def load_frozen_route(path: Path, expected_hash: str | None) -> dict:
    if not path.is_file():
        raise RouteNotFrozen(f"找不到路由文件: {path}")
    route = json.loads(path.read_text(encoding="utf-8"))
    status = route.get("status")
    if status != "frozen":
        raise RouteNotFrozen(f"路由未冻结（status={status!r}）；测试预测只在冻结路由后生成")
    recorded = route.get("route_sha256")
    if not recorded:
        raise RouteNotFrozen("路由文件缺少 route_sha256")
    if expected_hash is not None and recorded != expected_hash:
        raise RouteNotFrozen(f"路由哈希不符: 期望 {expected_hash}，实际 {recorded}")
    return route


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scenario", required=True, choices=RUNNABLE_SCENARIOS)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--split-spec-sha256", default=None)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--route-sha256", default=None)
    parser.add_argument("--selection-manifest", type=Path, required=True,
                        help="选择期 train.py 写出的 run_manifest.json")
    parser.add_argument("--candidate", required=True, choices=sorted(GATE_CANDIDATES))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1) 路由必须已冻结 ----
    try:
        load_frozen_route(args.route, args.route_sha256)
    except RouteNotFrozen as exc:
        print(f"RouteNotFrozen: {exc}", file=sys.stderr)
        return 3

    # ---- 2) 选择期清单必须一致 ----
    if not args.selection_manifest.is_file():
        print(f"找不到选择期清单: {args.selection_manifest}", file=sys.stderr)
        return 3
    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    if selection.get("candidate") != args.candidate:
        print(f"候选不符: 选择期 {selection.get('candidate')} vs 本轮 {args.candidate}",
              file=sys.stderr)
        return 3
    if selection.get("bundle_signature") != args.signature:
        print(f"Bundle 签名与选择期不符: {selection.get('bundle_signature')} vs {args.signature}",
              file=sys.stderr)
        return 3
    frozen_alphas = selection.get("alpha_by_group")
    if not frozen_alphas:
        print("选择期清单缺少 alpha_by_group，无法按冻结 α 重拟合", file=sys.stderr)
        return 3
    expected_weight_hash = selection.get("weight_hash")

    # ---- 3) 读取冻结 Bundle ----
    try:
        bundle = read_frozen_bundle(args.bundle, args.task, args.scenario, args.signature,
                                    split_spec_sha256=args.split_spec_sha256)
    except BundleUnavailable as exc:
        print(f"BundleUnavailable: {exc}", file=sys.stderr)
        return 3

    for segment in ("train", "calibration", "decision", "test"):
        assert_grid(bundle, segment)

    # ---- 4) 用冻结 α 重拟合，核对权重哈希 ----
    parts = {s: to_feature_bundle(bundle.segment(s)) for s in
             ("train", "calibration", "decision", "test")}
    merged = FeatureBundle_merge(parts["train"], parts["calibration"], parts["decision"])

    model = BranchResidualCandidate(args.candidate)
    model.fit_design(parts["train"])
    model.alpha_by_group = {k: float(v) for k, v in frozen_alphas.items()}
    model.refit(merged)

    got_hash = weight_hash(model.weights)
    if expected_weight_hash and got_hash != expected_weight_hash:
        print(f"权重哈希不符: 期望 {expected_weight_hash[:16]}…，实得 {got_hash[:16]}…"
              f"（Bundle 或配置与选择期不一致）", file=sys.stderr)
        return 3

    # ---- 5) 只写测试段预测（不读真值做任何选择）----
    test_part = parts["test"]
    predictions = model.predict(test_part)
    writer = PredictionWriter()
    writer.extend(
        predictions,
        origin_id=np.asarray(bundle.segment("test")["origin_id"]),
        origin_index=test_part.origin_index,
        task_id=args.task, fold_id=int(args.task.split("_f")[-1]), segment="test",
        scenario=args.scenario, candidate_id=args.candidate, seed=args.seed,
        bundle_signature=args.signature,
        config_sha256_value=selection.get("config_sha256", ""), commit=code_commit(REPO_ROOT),
    )
    n_rows = writer.write(out / f"{args.candidate.replace('+', '_')}_test_predictions.csv")

    manifest = {
        "candidate": args.candidate, "task": args.task, "scenario": args.scenario,
        "bundle_signature": args.signature, "route_sha256": args.route_sha256,
        "selection_config_sha256": selection.get("config_sha256"),
        "weight_hash": got_hash, "n_rows": n_rows,
        "n_test_origins": int(len(test_part.origin_index)),
        "frozen_alphas": frozen_alphas,
        "code_commit": code_commit(REPO_ROOT),
        "note": "测试段预测；生成时未读取测试真值。最终指标由主控独立计分。",
    }
    (out / f"{args.candidate.replace('+', '_')}_test_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    print(json.dumps({"status": "completed", "rows": n_rows,
                      "weight_hash": got_hash[:16]}, ensure_ascii=False))
    return 0


def FeatureBundle_merge(*parts):
    """把多个切片按行拼接成一个 FeatureBundle。"""
    from branches import FeatureBundle

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
