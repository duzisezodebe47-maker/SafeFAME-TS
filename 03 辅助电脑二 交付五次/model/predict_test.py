"""测试段预测入口（第五轮返修版）。

## 相对第四轮的修复（第五轮 A.1 / A.3 / A.4 / A.5）

**A.1 数值回退旁路**：第四轮的条件是
``if candidate not in GATE_CANDIDATES and candidate != fallback: raise`` ——
调用方**请求门控候选**时条件为假，于是不报错、**直接执行回退模型**。
现在要求 ``--candidate`` **严格等于** ``route.fallback``，否则拒绝。

**A.3 主控基线的清单**：`Last`/`SeasonalNaive`/`AR-Ridge` 由主控导出，
只有预测 CSV 与配置哈希，**没有模型侧 `run_manifest.json`**。
它们改为按"路由 + 主控固定配置"验证，**不虚构模型侧清单**。
`N` 属模型候选型数值回退，仍走选择期清单路径。

**A.4 两锚必填**：正式路由**必须**同时携带 `split_spec_sha256` 与 `bundle_signature`，
任一缺失或不一致即拒绝。旧的无锚 smoke 路由**不能**作为正式测试许可证。

**A.5 不向模型提供测试真值**：测试段构建**仅含特征的推理视图**，
`targets`/`targets_standardized` 一律不传入。评分由主控另一进程读取真值。

用法::

    python predict_test.py --bundle <目录> --task Agriculture_h12_f1 \\
        --scenario proxy --signature <sig> --split-spec <冻结spec路径> \\
        --route <路由.json> --route-sha256 <路由文件字节哈希> \\
        --candidate <必须等于 route.selected / route.fallback> \\
        [--selection-manifest <选择期 run_manifest.json>]   # 门控候选必填
        --output-dir <目录>
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

from branches import FeatureBundle  # noqa: E402
from bundle_reader import (  # noqa: E402
    BundleUnavailable, assert_grid, read_frozen_bundle, segment_bounds, sha256_file,
)
from candidates import GATE_CANDIDATES, PROTOCOL_ALPHAS, BranchResidualCandidate  # noqa: E402
from numeric_fallbacks import numeric_baselines  # noqa: E402
from predict_io import PredictionWriter, code_commit, weight_hash  # noqa: E402
from train import (  # noqa: E402
    RUNNABLE_SCENARIOS, merge_bundles, to_feature_bundle, to_inference_bundle,
)

REPO_ROOT = HERE.parents[1]

MASTER_BASELINES = ("Last", "SeasonalNaive", "AR-Ridge")


class RouteRejected(RuntimeError):
    """路由不满足冻结前置条件。**不得降级、不得改用别的候选。**"""


def load_route(path: Path, expected_file_sha256: str) -> dict:
    """以**文件字节 SHA256** 为外部锚读取路由（不比较路由内自称的哈希字段）。"""
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


def check_route(
    route: dict, task_id: str, fold_id: int, candidate: str,
    *, bundle_signature: str, split_spec_sha256: str,
) -> str:
    """核对路由锚与选择结果，返回**实际要跑的模型名**。

    A.4：`split_spec_sha256` 与 `bundle_signature` **两锚都必须在路由里且一致**，
    任一缺失即拒绝 —— 旧无锚 smoke 路由不能作为正式测试许可证。
    A.1：`--candidate` 必须与路由的选择结果**严格相等**，不存在旁路。
    """
    for key in ("task_id", "fold_id", "selected", "fallback", "selection_data_segments",
                "split_spec_sha256", "bundle_signature"):
        if key not in route:
            raise RouteRejected(f"路由缺少字段: {key}（A.4 要求两锚必填）")
    if str(route["task_id"]) != task_id:
        raise RouteRejected(f"路由任务不符: {route['task_id']} vs {task_id}")
    if int(route["fold_id"]) != fold_id:
        raise RouteRejected(f"路由折号不符: {route['fold_id']} vs {fold_id}")
    if list(route["selection_data_segments"]) != ["cal", "dec"]:
        raise RouteRejected(
            f"路由的选择数据段不是 [cal, dec]: {route['selection_data_segments']}")
    if str(route["split_spec_sha256"]) != str(split_spec_sha256):
        raise RouteRejected(
            f"路由的 split_spec_sha256 与本轮不符: "
            f"{route['split_spec_sha256']} vs {split_spec_sha256}")
    if str(route["bundle_signature"]) != str(bundle_signature):
        raise RouteRejected(
            f"路由的 bundle_signature 与本轮不符: "
            f"{route['bundle_signature']} vs {bundle_signature}")

    selected = str(route["selected"])
    if selected == "numeric_fallback":
        fallback = str(route["fallback"])
        # A.1：**严格相等**。第四轮允许 GATE_CANDIDATES 作为请求值却执行回退模型，
        # 是旁路 —— 现在请求什么就必须是什么。
        if candidate != fallback:
            raise RouteRejected(
                f"路由选中数值回退 {fallback!r}，但 --candidate={candidate!r}；"
                f"必须严格等于 route.fallback")
        if fallback not in MASTER_BASELINES and fallback != "N":
            raise RouteRejected(f"路由的回退模型 {fallback!r} 不在已实现范围内")
        return fallback
    if selected not in GATE_CANDIDATES:
        raise RouteRejected(f"路由的 selected={selected!r} 既非门控候选也非 numeric_fallback")
    if candidate != selected:
        raise RouteRejected(
            f"路由选中 {selected!r}，但 --candidate={candidate!r} —— "
            f"不得用未中选候选生成测试预测")
    return selected


def bundle_input_sha256(bundle: FeatureBundle) -> str:
    """一个训练**集合**的输入哈希：覆盖全部进入模型的特征与目标。"""
    digest = hashlib.sha256()
    for name in ("numeric_history", "semantic", "quality", "text_available",
                 "origin_index", "targets_standardized"):
        array = np.ascontiguousarray(getattr(bundle, name))
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


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
    parser.add_argument("--split-spec", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--route-sha256", required=True)
    parser.add_argument("--selection-manifest", type=Path, default=None,
                        help="门控候选必填；主控数值基线（Last/SeasonalNaive/AR-Ridge）不需要")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seasonal-period", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    task_match_h = int(args.task.split("_h")[1].split("_")[0])
    fold_id = int(args.task.split("_f")[-1])

    if not args.split_spec.is_file():
        print(f"找不到冻结协议文件: {args.split_spec}", file=sys.stderr)
        return 3
    spec_sha = sha256_file(args.split_spec)

    # ---- 1) 路由：文件字节锚 + 两锚必填 + 严格相等 ----
    try:
        route = load_route(args.route, args.route_sha256)
        model_name = check_route(route, args.task, fold_id, args.candidate,
                                 bundle_signature=args.signature,
                                 split_spec_sha256=spec_sha)
    except RouteRejected as exc:
        print(f"RouteRejected: {exc}", file=sys.stderr)
        return 3

    is_master_baseline = model_name in MASTER_BASELINES

    # ---- 2) 选择期清单：门控候选必填；主控基线不需要（A.3）----
    selection: dict = {}
    frozen_alphas = None
    expected_weight_hash = None
    if is_master_baseline:
        # 主控基线只导出预测 CSV 与配置哈希，**没有模型侧 run_manifest.json**。
        # 因此按"路由 + 主控固定配置"验证，不虚构模型侧清单。
        print(f"NOTE: {model_name} 是主控数值基线，按路由与主控固定配置验证，"
              f"不使用模型侧选择期清单（A.3）。", file=sys.stderr)
    else:
        if args.selection_manifest is None or not args.selection_manifest.is_file():
            print("门控候选必须提供 --selection-manifest", file=sys.stderr)
            return 3
        selection = json.loads(args.selection_manifest.read_text(encoding="utf-8"))
        if selection.get("candidate") != args.candidate:
            print(f"候选与选择期不符: {selection.get('candidate')} vs {args.candidate}",
                  file=sys.stderr)
            return 3
        if selection.get("bundle_signature") != args.signature:
            print(f"Bundle 签名与选择期不符", file=sys.stderr)
            return 3
        frozen_alphas = selection.get("alpha_by_group")
        expected_weight_hash = selection.get("weight_hash")
        if model_name != "N" and not frozen_alphas:
            print("选择期清单缺少 alpha_by_group，无法重演", file=sys.stderr)
            return 3

    # ---- 3) 冻结 Bundle ----
    try:
        bundle = read_frozen_bundle(args.bundle, args.task, args.scenario,
                                    args.signature, args.split_spec)
        bounds = segment_bounds(args.split_spec, args.task)
    except BundleUnavailable as exc:
        print(f"BundleUnavailable: {exc}", file=sys.stderr)
        return 3

    try:
        grid_stats = {s: assert_grid(bundle, s, bounds[s], task_match_h)
                      for s in ("train", "calibration", "decision", "test")}
    except AssertionError as exc:
        print(f"GridRejected: {exc}", file=sys.stderr)
        return 5

    # ---- 4) 构建视图 ----
    # 拟合视图含真值；**测试视图只含特征**（A.5：不向 predict 提供测试真值）
    fit_parts = {s: to_feature_bundle(bundle.segment(s))
                 for s in ("train", "calibration", "decision")}
    test_inference = to_inference_bundle(bundle.segment("test"))
    input_sha = {s: weight_hash(p.numeric_history) for s, p in fit_parts.items()}

    # 三种路径的标签、分支组成、求解器各不相同。`N` 是**模型候选型**数值回退
    # （有选择期清单、有权重重演），不是主控基线，也不是门控候选 ——
    # 早先把它标成 "gate_candidate" 是错的。
    if model_name in GATE_CANDIDATES:
        path_label = "gate_candidate"
        branches = list(BranchResidualCandidate(model_name).branch_names)
        solver = "group-penalized ridge, cholesky, float64 normal equations"
    elif model_name == "N":
        path_label = "numeric_fallback_candidate"
        branches = list(BranchResidualCandidate("N").branch_names)
        solver = "group-penalized ridge, cholesky, float64 normal equations"
    else:
        path_label = "master_numeric_baseline"
        branches = []
        solver = f"master numeric baseline: {model_name}"

    report: dict = {
        "model_run": model_name, "grid": grid_stats, "input_sha256": input_sha,
        "route_anchors_checked": ["split_spec_sha256", "bundle_signature"],
        "route_anchors_missing": [], "test_view_has_targets": False,
        "model_config": {
            "candidate": model_name, "branches": branches,
            "alpha_grid": list(PROTOCOL_ALPHAS),
            "frozen_alpha_by_group": frozen_alphas,
            "solver": solver, "target_scale": "train_only_standardized_OT",
        },
        "selection_input_sha256": None, "extended_input_sha256": None,
        "selection_rows": 0, "extended_rows": 0,
    }

    if is_master_baseline:
        # 主控基线：在**全量**行上拟合与预测，再切测试段（与主控实现的行序一致）
        parts_all = {s: to_feature_bundle(bundle.segment(s))
                     for s in ("train", "calibration", "decision", "test")}
        full = merge_bundles(parts_all["train"], parts_all["calibration"],
                             parts_all["decision"], parts_all["test"])
        segments = np.concatenate([
            np.full(len(parts_all[s].numeric_history), s)
            for s in ("train", "calibration", "decision", "test")])
        baseline = numeric_baselines(full, segments,
                                     seasonal_period=args.seasonal_period)
        n_test = len(test_inference.numeric_history)
        predictions = baseline["predictions"][model_name][-n_test:]
        report["path"] = path_label
        report["ridge_alpha"] = baseline["ridge_alpha"]
        report["ridge_calibration_mse"] = baseline["ridge_calibration_mse"]
        report["baseline_fit_rows"] = baseline["fit_rows"]
        report["baseline_calibration_rows"] = baseline["calibration_rows"]
        report["weight_hash"] = None
        report["test_fit_weight_hash"] = None
    else:
        model = BranchResidualCandidate(model_name)
        model.fit_design(fit_parts["train"])
        selection_fit = merge_bundles(fit_parts["train"], fit_parts["calibration"])
        if frozen_alphas:
            model.alpha_by_group = {k: float(v) for k, v in frozen_alphas.items()}
        else:
            model.select_alphas(fit_parts["train"], fit_parts["calibration"])

        weights_replay = _fit(model, selection_fit)
        replay_hash = weight_hash(weights_replay)
        report["selection_rows"] = int(len(selection_fit.numeric_history))
        report["selection_input_sha256"] = bundle_input_sha256(selection_fit)
        if expected_weight_hash and replay_hash != expected_weight_hash:
            print(f"选择期权重重演不符: 期望 {expected_weight_hash[:16]}…，"
                  f"实得 {replay_hash[:16]}…", file=sys.stderr)
            return 3

        extended_fit = merge_bundles(fit_parts["train"], fit_parts["calibration"],
                                     fit_parts["decision"])
        weights_extended = _fit(model, extended_fit)
        report["extended_rows"] = int(len(extended_fit.numeric_history))
        report["extended_input_sha256"] = bundle_input_sha256(extended_fit)
        report["weight_hash"] = replay_hash
        report["test_fit_weight_hash"] = weight_hash(weights_extended)
        # A.5：只喂测试**特征**视图
        predictions = model.predict(test_inference)
        report["path"] = path_label

    if predictions.ndim != 2 or predictions.shape[0] != len(test_inference.origin_index):
        print(f"预测形状异常: {predictions.shape}", file=sys.stderr)
        return 4

    writer = PredictionWriter()
    writer.extend(
        predictions,
        origin_id=np.asarray(bundle.segment("test")["origin_id"]),
        origin_index=test_inference.origin_index,
        task_id=args.task, fold_id=fold_id, segment="test", scenario=args.scenario,
        candidate_id=model_name, seed=args.seed, bundle_signature=args.signature,
        config_sha256_value=selection.get("config_sha256", spec_sha),
        commit=code_commit(REPO_ROOT),
    )
    safe_name = model_name.replace("+", "_").replace("-", "_")
    n_rows = writer.write(out / f"{safe_name}_test_predictions.csv")

    manifest = {
        "task": args.task, "scenario": args.scenario, "candidate": model_name,
        "requested_candidate": args.candidate,
        "route_selected": route.get("selected"), "route_fallback": route.get("fallback"),
        "route_file_sha256": args.route_sha256,
        "bundle_signature": args.signature, "split_spec_sha256": spec_sha,
        "selection_config_sha256": selection.get("config_sha256"),
        "n_rows": n_rows, "n_test_origins": int(len(test_inference.origin_index)),
        "code_commit": code_commit(REPO_ROOT),
        "note": ("测试段预测。生成时未读取测试真值；路由已冻结且强制执行。"
                 "最终指标由主控独立计分。"),
        **report,
    }
    (out / f"{safe_name}_test_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    print(json.dumps({"status": "completed", "model": model_name, "rows": n_rows,
                      "path": report["path"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
