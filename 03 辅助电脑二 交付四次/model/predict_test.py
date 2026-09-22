"""测试段预测入口（第四轮返修版）。

## 相对第三轮的修复

**P0-1：核对两个不同训练集合的权重哈希。**第三轮在 train+cal+decision 上重拟合，
却要求新权重与选择期（train+calibration）的 `weight_hash` 完全相同 —— 权重本来就会变，
真实测试会被**错误拒绝**。现在分两阶段：

  阶段 1（重演）：按选择期完全相同的 train+calibration + 冻结 α + 配置重拟合，
                  与选择期 `weight_hash` 核对。不一致才报错。
  阶段 2（扩展）：一致后，按冻结方案扩至 train+calibration+decision 重拟合，
                  另存 `test_fit_weight_hash`。**扩展后的哈希不与选择期比较。**

**P0-2：路由未强制执行。**
  - `--route-sha256` 必填，且必须等于路由文件的**实际字节** SHA256
    （不是 JSON 里某个自称的字段 —— 也不得构造文件自引用哈希）；
  - 核对 `route.task_id` / `fold_id` / `selection_data_segments == ["cal","dec"]`；
  - `route.selected` 是门控候选时**只允许该候选**；是 `numeric_fallback` 时
    按 `route.fallback` 运行路由指定的数值回退模型。
  - 主控 `freeze_route` 输出里**没有** `status`/内嵌 `route_sha256`，故不再要求它们。

用法::

    python predict_test.py --bundle <目录> --task Agriculture_h12_f1 \\
        --scenario proxy --signature <sig> --split-spec <冻结spec路径> \\
        --route <路由.json> --route-sha256 <路由文件字节哈希> \\
        --selection-manifest <选择期 run_manifest.json> \\
        --candidate <必须等于 route.selected> --output-dir <目录>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from branches import FeatureBundle  # noqa: E402
from bundle_reader import (  # noqa: E402
    BundleUnavailable, assert_grid, read_frozen_bundle, segment_bounds, sha256_file,
)
from candidates import GATE_CANDIDATES, BranchResidualCandidate  # noqa: E402
from numeric_fallbacks import numeric_fallback_predict  # noqa: E402
from predict_io import PredictionWriter, code_commit, weight_hash  # noqa: E402
from train import RUNNABLE_SCENARIOS, merge_bundles, to_feature_bundle  # noqa: E402

REPO_ROOT = HERE.parents[1]


class RouteRejected(RuntimeError):
    """路由不满足冻结前置条件。**不得降级、不得改用别的候选。**"""


def load_route(path: Path, expected_file_sha256: str) -> dict:
    """以**文件字节 SHA256** 为外部锚读取路由。

    不比较路由文件内任何自称的哈希字段 —— 那可以自洽地伪造。
    """
    path = Path(path)
    if not path.is_file():
        raise RouteRejected(f"找不到路由文件: {path}")
    actual = sha256_file(path)
    if actual != expected_file_sha256:
        raise RouteRejected(
            f"路由文件字节哈希不符: 期望 {expected_file_sha256}，实际 {actual}")
    route = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(route, dict):
        raise RouteRejected("路由文件不是 JSON 对象")
    return route


def check_route(route: dict, task_id: str, fold_id: int, candidate: str) -> str:
    """核对路由锚与选择结果，返回**实际要跑的模型名**。"""
    for key in ("task_id", "fold_id", "selected", "fallback", "selection_data_segments"):
        if key not in route:
            raise RouteRejected(f"路由缺少字段: {key}")
    if str(route["task_id"]) != task_id:
        raise RouteRejected(f"路由任务不符: {route['task_id']} vs {task_id}")
    if int(route["fold_id"]) != fold_id:
        raise RouteRejected(f"路由折号不符: {route['fold_id']} vs {fold_id}")
    if list(route["selection_data_segments"]) != ["cal", "dec"]:
        raise RouteRejected(
            f"路由的选择数据段不是 [cal, dec]: {route['selection_data_segments']}")

    selected = str(route["selected"])
    if selected == "numeric_fallback":
        # 数值回退：按路由指定的回退模型跑，不接受调用方另选
        fallback = str(route["fallback"])
        if candidate not in GATE_CANDIDATES and candidate != fallback:
            raise RouteRejected(
                f"路由选中数值回退 {fallback!r}，但 --candidate={candidate!r}；"
                f"数值回退路径不接受其它候选")
        return fallback
    if selected not in GATE_CANDIDATES:
        raise RouteRejected(f"路由的 selected={selected!r} 既非门控候选也非 numeric_fallback")
    if candidate != selected:
        raise RouteRejected(
            f"路由选中 {selected!r}，但 --candidate={candidate!r} —— "
            f"不得用未中选候选生成测试预测")
    return selected


def _fit(model: BranchResidualCandidate, fit: FeatureBundle) -> np.ndarray:
    model.refit(fit)
    if model.weights is None:
        raise AssertionError("refit 后权重为空")
    return model.weights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scenario", required=True, choices=RUNNABLE_SCENARIOS)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--split-spec", type=Path, required=True,
                        help="主控冻结协议文件路径（本入口自己算字节哈希）")
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--route-sha256", required=True,
                        help="路由文件的**实际字节** SHA256（必填）")
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seasonal-period", type=int, default=12,
                        help="SeasonalNaive 回退的季节周期")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    task_match_h = int(args.task.split("_h")[1].split("_")[0])
    fold_id = int(args.task.split("_f")[-1])

    # ---- 1) 路由：文件字节锚 + 强制执行选择结果 ----
    try:
        route = load_route(args.route, args.route_sha256)
        model_name = check_route(route, args.task, fold_id, args.candidate)
    except RouteRejected as exc:
        print(f"RouteRejected: {exc}", file=sys.stderr)
        return 3

    # ---- 2) 选择期清单 ----
    if not args.selection_manifest.is_file():
        print(f"找不到选择期清单: {args.selection_manifest}", file=sys.stderr)
        return 3
    selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
    if selection.get("candidate") != args.candidate:
        print(f"候选与选择期不符: {selection.get('candidate')} vs {args.candidate}",
              file=sys.stderr)
        return 3
    if selection.get("bundle_signature") != args.signature:
        print(f"Bundle 签名与选择期不符: {selection.get('bundle_signature')} vs {args.signature}",
              file=sys.stderr)
        return 3
    frozen_alphas = selection.get("alpha_by_group")
    expected_weight_hash = selection.get("weight_hash")

    # ---- 3) 冻结 Bundle ----
    try:
        bundle = read_frozen_bundle(args.bundle, args.task, args.scenario,
                                    args.signature, args.split_spec)
        bounds = segment_bounds(args.split_spec, args.task)
    except BundleUnavailable as exc:
        print(f"BundleUnavailable: {exc}", file=sys.stderr)
        return 3

    # 正式入口**必须**传入边界与 H（P1-4）
    grid_stats = {s: assert_grid(bundle, s, bounds[s], task_match_h)
                  for s in ("train", "calibration", "decision", "test")}

    parts = {s: to_feature_bundle(bundle.segment(s)) for s in
             ("train", "calibration", "decision", "test")}
    input_sha = {s: weight_hash(p.numeric_history) for s, p in parts.items()}

    # ---- 4) 两阶段：先重演核对选择期哈希，再扩展 ----
    report: dict = {"model_run": model_name, "grid": grid_stats,
                    "input_sha256": input_sha,
                    "selection_rows": 0, "extended_rows": 0}

    if model_name in ("Last", "SeasonalNaive", "AR-Ridge"):
        # 数值回退路径：不需要 α 与权重重演
        predictions = numeric_fallback_predict(
            model_name, parts["train"], parts["test"],
            seasonal_period=args.seasonal_period)
        report["path"] = "numeric_fallback"
        report["weight_hash"] = None
        report["test_fit_weight_hash"] = None
    else:
        model = BranchResidualCandidate(model_name)
        model.fit_design(parts["train"])
        if not frozen_alphas:
            print("选择期清单缺少 alpha_by_group，无法重演", file=sys.stderr)
            return 3
        model.alpha_by_group = {k: float(v) for k, v in frozen_alphas.items()}

        # 阶段 1：与选择期完全相同的 train+calibration
        selection_fit = merge_bundles(parts["train"], parts["calibration"])
        weights_replay = _fit(model, selection_fit)
        replay_hash = weight_hash(weights_replay)
        report["selection_rows"] = int(len(selection_fit.numeric_history))
        if expected_weight_hash and replay_hash != expected_weight_hash:
            print(f"选择期权重重演不符: 期望 {expected_weight_hash[:16]}…，"
                  f"实得 {replay_hash[:16]}…（→ Bundle 或配置与选择期不一致）", file=sys.stderr)
            return 3

        # 阶段 2：按冻结方案扩至 train+calibration+decision（哈希另存，不与选择期比较）
        extended_fit = merge_bundles(parts["train"], parts["calibration"], parts["decision"])
        weights_extended = _fit(model, extended_fit)
        report["extended_rows"] = int(len(extended_fit.numeric_history))
        report["weight_hash"] = replay_hash
        report["test_fit_weight_hash"] = weight_hash(weights_extended)
        predictions = model.predict(parts["test"])
        report["path"] = "gate_candidate"

    if predictions.ndim != 2:
        print(f"预测形状异常: {predictions.shape}", file=sys.stderr)
        return 4

    # ---- 5) 只写测试段预测 ----
    writer = PredictionWriter()
    writer.extend(
        predictions,
        origin_id=np.asarray(bundle.segment("test")["origin_id"]),
        origin_index=parts["test"].origin_index,
        task_id=args.task, fold_id=fold_id, segment="test", scenario=args.scenario,
        candidate_id=model_name, seed=args.seed, bundle_signature=args.signature,
        config_sha256_value=selection.get("config_sha256", ""), commit=code_commit(REPO_ROOT),
    )
    n_rows = writer.write(out / f"{model_name.replace('+', '_')}_test_predictions.csv")

    manifest = {
        "task": args.task, "scenario": args.scenario, "candidate": model_name,
        "requested_candidate": args.candidate,
        "route_selected": route.get("selected"), "route_fallback": route.get("fallback"),
        "route_file_sha256": args.route_sha256,
        "bundle_signature": args.signature,
        "split_spec_sha256": sha256_file(args.split_spec),
        "selection_config_sha256": selection.get("config_sha256"),
        "n_rows": n_rows, "n_test_origins": int(len(parts["test"].origin_index)),
        "code_commit": code_commit(REPO_ROOT),
        "note": ("测试段预测。生成时未读取测试真值；路由已冻结且强制执行。"
                 "最终指标由主控独立计分。"),
        **report,
    }
    (out / f"{model_name.replace('+', '_')}_test_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    print(json.dumps({"status": "completed", "model": model_name, "rows": n_rows,
                      "test_fit_weight_hash": report.get("test_fit_weight_hash")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
