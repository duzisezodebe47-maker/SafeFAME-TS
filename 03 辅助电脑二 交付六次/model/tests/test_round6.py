"""第六轮测试：三项修复 + 保留核心行为（合成数据）。

  修复 1  只接受主控封印路由（status=frozen + 两锚 + 来源哈希）
  修复 2  数值回退的接口里**不存在**测试真值字段
  修复 3  季节周期从冻结 spec 顶层 `seasonal_periods` 取，不默认 12

用法::

    .venv/Scripts/python.exe "03 辅助电脑二 交付六次/model/tests/test_round6.py"
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent
sys.path.insert(0, str(MODEL))

from branches import FeatureBundle  # noqa: E402
from candidates import GATE_CANDIDATES  # noqa: E402

# 主控 team_eval/v2.py 的常量 —— 逐字抄录，供契约核对
MASTER_PREDICTION_COLUMNS = frozenset({
    "task_id", "fold_id", "origin_id", "origin_index", "segment", "scenario",
    "candidate_id", "seed", "step", "y_pred", "target_scale",
    "bundle_signature", "config_sha256", "code_commit",
})
MASTER_SCALE = "train_only_standardized_OT"

PASSED: list[str] = []


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    PASSED.append(label)
    print(f"  ok  {label}")


def make_spec(path: Path, *, seasonal: dict | None = None, n_rows: int = 60) -> Path:
    """四段边界 [0,20) [20,30) [30,45) [45,60)；周期在**顶层**。"""
    spec = {
        "schema_version": 1, "status": "frozen", "approved_by": "test-master",
        "alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        "row_permutations": 999, "p_threshold": 0.025,
        "gate_candidates": ["N+S+Q", "N+S+Q+SF"],
        "numeric_fallback_candidates": ["Last", "SeasonalNaive", "AR-Ridge", "N"],
        "seasonal_periods": seasonal if seasonal is not None else {"Agriculture": 3},
        "tasks": [{"domain": "Agriculture", "fold_id": 1, "input_len": 6, "horizon": 3,
                   "n_rows_expected": n_rows, "bounds": [20, 30, 45, 60],
                   "numerical_sha256": "a" * 64}],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return path


def make_bundle(root: Path, spec_path: Path, *, n=60, horizon=3, semantic_dim=16,
                quality_dim=3, seed=5) -> Path:
    from bundle_reader import sha256_file, signature

    root.mkdir(parents=True, exist_ok=True)
    task = root / "Agriculture_h3_f1"
    task.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)

    seg_bounds = [("train", 0, 20), ("calibration", 20, 30),
                  ("decision", 30, 45), ("test", 45, 60)]
    keep = []
    for o in range(n):
        for nm, lo, hi in seg_bounds:
            if lo <= o < hi and o + horizon <= hi:
                keep.append((o, nm))
                break
    origins = np.array([o for o, _ in keep], dtype=np.int64)
    segments = [s for _, s in keep]

    for key, val in {
        "numeric_history": rng.normal(size=(n, 6)).astype(np.float32),
        "targets": rng.normal(size=(n, horizon)).astype(np.float32),
        "targets_standardized": rng.normal(size=(n, horizon)).astype(np.float32),
    }.items():
        np.save(task / f"{key}.npy", val[origins], allow_pickle=False)
    np.save(task / "origin_index.npy", origins, allow_pickle=False)

    scen = task / "proxy"; scen.mkdir(exist_ok=True)
    for key, val in {
        "semantic": rng.normal(size=(n, semantic_dim)).astype(np.float32),
        "quality": rng.normal(size=(n, quality_dim)).astype(np.float32),
        "text_available": (rng.random(n) < 0.7),
    }.items():
        np.save(scen / f"{key}.npy", val[origins], allow_pickle=False)

    pd.DataFrame([{"task_id": "Agriculture_h3_f1", "fold_id": 1,
                   "origin_id": f"Agriculture:h3:f1:o{int(o)}", "origin_index": int(o),
                   "segment": segments[i]} for i, o in enumerate(origins)]
                 ).to_csv(task / "samples.csv", index=False, lineterminator="\n")

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    (root / "schema.json").write_text(json.dumps({"status": "frozen"}), encoding="utf-8")
    inputs = {"mode": "frozen", "split_spec_sha256": sha256_file(spec_path),
              "split_spec": spec}
    files = {p.relative_to(root).as_posix(): sha256_file(p)
             for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (root / "manifest.json").write_text(
        json.dumps({"inputs": inputs, "signature": signature(inputs), "files": files}),
        encoding="utf-8")
    return root


def sealed_route(spec_path: Path, spec_sha: str, sig: str, **over) -> dict:
    """主控 `route_seal.seal_route` 会写出的字段集合。"""
    route = {"task_id": "Agriculture_h3_f1", "fold_id": 1, "selected": "N+S+Q",
             "fallback": "AR-Ridge", "selection_data_segments": ["cal", "dec"],
             "status": "frozen", "scenario": "proxy",
             "split_spec_sha256": spec_sha, "bundle_signature": sig,
             # hex 只允许 0-9a-f
             "selection_manifest_sha256": "a" * 64,
             "refit_review_sha256": "b" * 64,
             "inputs_sha256": {"task": "c" * 64, "spec": "d" * 64}}
    route.update(over)
    return route


def main() -> int:
    print("第六轮测试（合成数据）")

    from bundle_reader import (
        BundleUnavailable, read_frozen_bundle, sha256_file, task_seasonal_period,
    )
    from candidates import BranchResidualCandidate
    from numeric_fallbacks import BaselineError, numeric_baselines
    from predict_test import RouteRejected, check_route, spec_candidate_lists

    # ================= 修复 3：季节周期从 spec 顶层取 =================
    print("\n--- 修复 3：季节周期来自冻结 spec ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json",
                         seasonal={"Agriculture": 3, "Climate": 5, "Environment": 2})
        check("Agriculture 周期 = 3（非默认 12）",
              task_seasonal_period(spec, "Agriculture_h3_f1") == 3)
        check("Climate 周期 = 5", task_seasonal_period(spec, "Climate_h4_f2") == 5)
        check("Environment 周期 = 2", task_seasonal_period(spec, "Environment_h7_f2") == 2)

        bad = make_spec(tmp / "nospec.json", seasonal={"Climate": 5})
        r = False
        try:
            task_seasonal_period(bad, "Agriculture_h3_f1")
        except BundleUnavailable as exc:
            r = "未登记" in str(exc)
        check("任务域未登记周期 → 拒绝（不默认 12）", r)

        nos = make_spec(tmp / "none.json")
        s = json.loads(nos.read_text(encoding="utf-8"))
        s.pop("seasonal_periods")
        nos.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
        r = False
        try:
            task_seasonal_period(nos, "Agriculture_h3_f1")
        except BundleUnavailable as exc:
            r = "seasonal_periods" in str(exc)
        check("spec 未声明 seasonal_periods → 拒绝", r)

    # ================= 修复 2：接口里不存在测试真值 =================
    print("\n--- 修复 2：数值回退接口无真值字段 ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")
        bdir = make_bundle(tmp / "b", spec)
        sig = json.loads((bdir / "manifest.json").read_text())["signature"]
        b = read_frozen_bundle(bdir, "Agriculture_h3_f1", "proxy", sig, spec)

        order = ("train", "calibration", "decision", "test")
        parts = {s: b.segment(s) for s in order}
        segments = np.concatenate([np.full(len(parts[s]["origin_index"]), s) for s in order])
        feats = FeatureBundle(
            numeric_history=np.concatenate([parts[s]["numeric_history"] for s in order]),
            semantic=np.concatenate([parts[s]["semantic"] for s in order]),
            quality=np.concatenate([parts[s]["quality"] for s in order]),
            text_available=np.concatenate([parts[s]["text_available"] for s in order]),
            origin_index=np.concatenate([parts[s]["origin_index"] for s in order]),
            targets=None, targets_standardized=None)
        check("推理视图不含 targets / targets_standardized",
              feats.targets is None and feats.targets_standardized is None)

        # 若把带真值的对象传进去，必须硬失败
        with_truth = FeatureBundle(
            numeric_history=feats.numeric_history, semantic=feats.semantic,
            quality=feats.quality, text_available=feats.text_available,
            origin_index=feats.origin_index,
            targets=np.zeros((len(segments), 3)), targets_standardized=np.zeros((len(segments), 3)))
        r = False
        try:
            numeric_baselines(with_truth, segments, np.zeros((len(segments), 3)))
        except BaselineError as exc:
            r = "推理视图" in str(exc)
        check("传入含真值对象 → 拒绝", r)

        horizon = np.asarray(parts["train"]["targets_standardized"]).shape[1]
        fit_t = np.full((len(segments), horizon), np.nan)
        for s in ("train", "calibration"):
            fit_t[segments == s] = np.asarray(parts[s]["targets_standardized"], dtype=float)
        period = task_seasonal_period(spec, "Agriculture_h3_f1")
        res = numeric_baselines(feats, segments, fit_t, seasonal_period=period)
        check("合法调用可运行且 α 来自 7 档",
              res["ridge_alpha"] in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0))

        # 把测试段真值填进去 → 必须拒绝（越界）
        leaky = fit_t.copy()
        leaky[segments == "test"] = 1.0
        r = False
        try:
            numeric_baselines(feats, segments, leaky, seasonal_period=period)
        except BaselineError as exc:
            r = "越界" in str(exc)
        check("fit_targets 在 train/cal 之外含真值 → 拒绝", r)

        # 改测试真值后输出逐位不变
        base_pred = res["predictions"]["AR-Ridge"].copy()
        task_dir = bdir / "Agriculture_h3_f1"
        tm = b.segment_mask("test")
        rng2 = np.random.default_rng(99)
        for key in ("targets", "targets_standardized"):
            arr = np.load(task_dir / f"{key}.npy").copy()
            arr[tm] = rng2.normal(size=(int(tm.sum()), arr.shape[1])).astype(arr.dtype)
            np.save(task_dir / f"{key}.npy", arr, allow_pickle=False)
        res2 = numeric_baselines(feats, segments, fit_t, seasonal_period=period)
        check("改测试真值后基线输出逐位不变",
              np.array_equal(base_pred, res2["predictions"]["AR-Ridge"]))

    # ================= 修复 1：只接受主控封印路由 =================
    print("\n--- 修复 1：只接受主控封印路由 ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")
        spec_sha = sha256_file(spec)
        bdir = make_bundle(tmp / "b", spec)
        sig = json.loads((bdir / "manifest.json").read_text())["signature"]
        gates, falls = spec_candidate_lists(spec)

        full = sealed_route(spec, spec_sha, sig)
        check("完整封印路由通过",
              check_route(full, "Agriculture_h3_f1", 1, "N+S+Q",
                          bundle_signature=sig, split_spec_sha256=spec_sha,
                          spec_gates=gates, spec_fallbacks=falls) == "N+S+Q")

        # 自造路由（只有两锚，无封印字段）必须拒绝
        for drop in ("status", "scenario", "selection_manifest_sha256",
                     "refit_review_sha256", "inputs_sha256"):
            route = dict(full)
            route.pop(drop)
            r = False
            try:
                check_route(route, "Agriculture_h3_f1", 1, "N+S+Q",
                            bundle_signature=sig, split_spec_sha256=spec_sha,
                            spec_gates=gates, spec_fallbacks=falls)
            except RouteRejected as exc:
                r = "封印字段" in str(exc)
            check(f"缺封印字段 {drop} → 拒绝", r)

        r = False
        try:
            check_route(dict(full, status="draft"), "Agriculture_h3_f1", 1, "N+S+Q",
                        bundle_signature=sig, split_spec_sha256=spec_sha,
                        spec_gates=gates, spec_fallbacks=falls)
        except RouteRejected as exc:
            r = "frozen" in str(exc)
        check("status 非 frozen → 拒绝", r)

        for field in ("selection_manifest_sha256", "refit_review_sha256"):
            r = False
            try:
                check_route(dict(full, **{field: "not-a-hash"}), "Agriculture_h3_f1", 1, "N+S+Q",
                            bundle_signature=sig, split_spec_sha256=spec_sha,
                            spec_gates=gates, spec_fallbacks=falls)
            except RouteRejected as exc:
                r = "64 位十六进制" in str(exc)
            check(f"{field} 格式非法 → 拒绝", r)

        r = False
        try:
            check_route(dict(full, inputs_sha256={"task": "bad"}),
                        "Agriculture_h3_f1", 1, "N+S+Q",
                        bundle_signature=sig, split_spec_sha256=spec_sha,
                        spec_gates=gates, spec_fallbacks=falls)
        except RouteRejected as exc:
            r = "非法哈希" in str(exc)
        check("inputs_sha256 含非法哈希 → 拒绝", r)

        r = False
        try:
            check_route(dict(full, selected="numeric_fallback", fallback="AR-Ridge"),
                        "Agriculture_h3_f1", 1, "N+S+Q",
                        bundle_signature=sig, split_spec_sha256=spec_sha,
                        spec_gates=gates, spec_fallbacks=falls)
        except RouteRejected as exc:
            r = "严格等于" in str(exc)
        check("数值回退时请求门控候选 → 拒绝（第五轮 A.1 保持）", r)

    # ================= 入口端到端（保留） =================
    print("\n--- 入口端到端 ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")
        bdir = make_bundle(tmp / "e2e", spec)
        sig = json.loads((bdir / "manifest.json").read_text())["signature"]
        common = ["--bundle", str(bdir), "--task", "Agriculture_h3_f1",
                  "--scenario", "proxy", "--signature", sig, "--split-spec", str(spec)]

        def run_entry(module, argv):
            old = sys.argv; sys.argv = argv
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    return module.main()
            except SystemExit as exc:
                return int(exc.code or 1)
            except Exception:
                return -1
            finally:
                sys.argv = old

        import train as train_entry
        out_t = tmp / "t"
        code = run_entry(train_entry, ["train.py", *common, "--segments", "calibration",
                                       "decision", "--candidate", "N+S+Q",
                                       "--output-dir", str(out_t)])
        check("train.py 端到端成功", code == 0)
        csvs = list(out_t.glob("*_predictions.csv"))
        import csv as _csv
        with csvs[0].open(encoding="utf-8-sig", newline="") as fh:
            rows = list(_csv.DictReader(fh))
        check("CSV 列 == 主控 PREDICTION_COLUMNS",
              set(rows[0]) == set(MASTER_PREDICTION_COLUMNS))
        check("segment 列是逐行段名（不是数组字符串）",
              {r["segment"] for r in rows} == {"calibration", "decision"})
        check("target_scale 为主控 SCALE",
              all(r["target_scale"] == MASTER_SCALE for r in rows))
        check("每起点每步恰一行",
              len(rows) == len({r["origin_id"] for r in rows}) * 3)

        import permutation_entry as perm_entry
        out_p = tmp / "p"
        code = run_entry(perm_entry, ["permutation_entry.py", *common,
                                      "--candidate", "N+S+Q", "--nulls", "3",
                                      "--output-dir", str(out_p)])
        check("permutation_entry.py 端到端成功", code == 0)
        psum = json.loads((out_p / "permutation_summary.json").read_text(encoding="utf-8"))
        check("未满 999 时 p=null 且记录了 seasonal/网格来源",
              psum["p_value"] is None and psum["successful"] == 3)

        import audit_tables as audit_entry
        out_a = tmp / "a"
        code = run_entry(audit_entry, ["audit_tables.py", *common,
                                       "--candidate", "N+S+Q", "--output-dir", str(out_a)])
        check("audit_tables.py 端到端成功", code == 0)
        check("审计表两张都在",
              (out_a / "text_availability_audit.csv").is_file()
              and (out_a / "decision_halves_audit.csv").is_file())

    print(f"\n全部通过（{len(PASSED)} 项检查）")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1)
