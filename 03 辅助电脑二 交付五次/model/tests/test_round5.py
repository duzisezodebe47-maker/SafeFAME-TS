"""第五轮测试（合成数据；正式 Bundle 未交付）。

覆盖第五轮任务书 A 部分五项，以及沿用前轮的契约/边界/半段负例：

  A.1  numeric_fallback 旁路：请求门控候选却执行回退模型 —— 必须从**入口**拒绝
  A.2  AR-Ridge 与主控 `v2.numeric_baselines` 的**逐行等价**（含 α）
  A.3  主控基线不需模型侧选择期清单；`N` 仍需
  A.4  正式路由必须同时携带 `split_spec_sha256` 与 `bundle_signature` 两锚
  A.5  测试预测不得接收测试真值 —— 删/改真值后预测字节不变

用法::

    .venv/Scripts/python.exe "03 辅助电脑二 交付五次/model/tests/test_round5.py"
"""

from __future__ import annotations

import contextlib
import io
import json
import math
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

# 主控 team_eval/v2.py 的 PREDICTION_COLUMNS —— 逐字抄录，供 CSV 契约核对
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


# ---------------- 合成冻结 spec 与 Bundle ----------------

def make_spec(path: Path, *, status="frozen", approved="master", n_rows=60) -> Path:
    """四段边界严格递增：train [0,20) cal [20,30) dec [30,45) test [45,60]。"""
    spec = {
        "schema_version": 1, "status": status, "approved_by": approved,
        "alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        "row_permutations": 999, "p_threshold": 0.025,
        "gate_candidates": ["N+S+Q", "N+S+Q+SF"],
        "numeric_fallback_candidates": ["Last", "SeasonalNaive", "AR-Ridge", "N"],
        "tasks": [{"domain": "Agriculture", "fold_id": 1, "input_len": 12, "horizon": 3,
                   "n_rows_expected": n_rows, "bounds": [20, 30, 45, 60],
                   "numerical_sha256": "a" * 64}],
    }
    path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    return path


def make_bundle(root: Path, spec_path: Path, *, n=60, horizon=3, semantic_dim=16,
                quality_dim=3, seed=5, tamper=None) -> Path:
    from bundle_reader import sha256_file, signature

    root.mkdir(parents=True, exist_ok=True)
    task = root / "Agriculture_h3_f1"
    task.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)

    seg_bounds = [("train", 0, 20), ("calibration", 20, 30),
                  ("decision", 30, 45), ("test", 45, 60)]
    keep = []
    for o in range(n):
        for name, lo, hi in seg_bounds:
            if lo <= o < hi and o + horizon <= hi:
                keep.append((o, name))
                break
    origins = np.array([o for o, _ in keep], dtype=np.int64)
    segments = [s for _, s in keep]

    full = {
        "numeric_history": rng.normal(size=(n, 12)).astype(np.float32),
        "targets": rng.normal(size=(n, horizon)).astype(np.float32),
        "targets_standardized": rng.normal(size=(n, horizon)).astype(np.float32),
    }
    for key, val in full.items():
        np.save(task / f"{key}.npy", val[origins], allow_pickle=False)
    np.save(task / "origin_index.npy", origins, allow_pickle=False)

    scen = task / "proxy"; scen.mkdir(exist_ok=True)
    per_scen = {
        "semantic": rng.normal(size=(n, semantic_dim)).astype(np.float32),
        "quality": rng.normal(size=(n, quality_dim)).astype(np.float32),
        "text_available": (rng.random(n) < 0.7),
    }
    for key, val in per_scen.items():
        np.save(scen / f"{key}.npy", val[origins], allow_pickle=False)

    rows = [{"task_id": "Agriculture_h3_f1", "fold_id": 1,
             "origin_id": f"Agriculture:h3:f1:o{int(o)}", "origin_index": int(o),
             "segment": segments[i]} for i, o in enumerate(origins)]
    pd.DataFrame(rows).to_csv(task / "samples.csv", index=False, lineterminator="\n")

    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    (root / "schema.json").write_text(json.dumps({"status": "frozen"}), encoding="utf-8")
    inputs = {"mode": "frozen", "split_spec_sha256": sha256_file(spec_path),
              "split_spec": spec}
    files = {p.relative_to(root).as_posix(): sha256_file(p)
             for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    manifest = {"inputs": inputs, "signature": signature(inputs), "files": files}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    if tamper:
        tamper(root, manifest)
    return root


# ---------------- 主控实现的参考移植（A.2 的比对基准）----------------

def reference_numeric_baselines(samples, arrays, alpha_grid, seasonal_period) -> dict:
    """`team_eval/v2.py::numeric_baselines` 的**逐行抄录**，仅作比对基准。

    刻意不改任何细节（连 `np.linalg.solve` 与 (loss, alpha) 平局规则都保留），
    否则比对就失去意义。
    """
    x = np.asarray(arrays["numeric_history"], dtype=float)
    y = np.asarray(arrays["targets_standardized"], dtype=float)
    train = np.array([r["segment"] == "train" for r in samples])
    cal = np.array([r["segment"] == "calibration" for r in samples])
    if train.sum() < 2 or not cal.any() or seasonal_period < 1 or seasonal_period > x.shape[1]:
        raise RuntimeError("numeric baseline lacks train/cal rows or usable seasonal history")

    last = np.repeat(x[:, -1:], y.shape[1], axis=1)
    seasonal = np.stack([x[:, -seasonal_period + (step % seasonal_period)]
                         for step in range(y.shape[1])], axis=1)
    x_mean, y_mean = x[train].mean(axis=0), y[train].mean(axis=0)
    xc, yc = x[train] - x_mean, y[train] - y_mean
    gram, cross = xc.T @ xc, xc.T @ yc
    candidates = {}
    for alpha in sorted(set(map(float, alpha_grid))):
        weights = np.linalg.solve(gram + alpha * np.eye(x.shape[1]), cross)
        prediction = (x - x_mean) @ weights + y_mean
        loss = float(np.mean((prediction[cal] - y[cal]) ** 2))
        candidates[alpha] = (loss, prediction)
    chosen_alpha = min(candidates, key=lambda a: (candidates[a][0], a))
    predictions = {"Last": last, "SeasonalNaive": seasonal,
                   "AR-Ridge": candidates[chosen_alpha][1]}
    return {"predictions": predictions, "ridge_alpha": chosen_alpha,
            "ridge_calibration_mse": candidates[chosen_alpha][0],
            "fit_rows": int(train.sum()), "calibration_rows": int(cal.sum())}


def main() -> int:
    print("第五轮测试（合成数据；正式 Bundle 未交付）")

    from bundle_reader import sha256_file, signature
    from candidates import BranchResidualCandidate
    from numeric_fallbacks import numeric_baselines
    from predict_io import weight_hash
    from bundle_reader import samples_order_view
    from train import merge_bundles, to_feature_bundle, to_inference_bundle

    # ---------------- A.2 与主控实现的逐行等价 ----------------
    print("\n--- A.2 数值基线与主控实现等价 ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")
        bdir = make_bundle(tmp / "a2", spec)
        sig = json.loads((bdir / "manifest.json").read_text())["signature"]
        from bundle_reader import read_frozen_bundle
        b = read_frozen_bundle(bdir, "Agriculture_h3_f1", "proxy", sig, spec)

        parts = {s: to_feature_bundle(b.segment(s))
                 for s in ("train", "calibration", "decision", "test")}
        full = merge_bundles(*[parts[s] for s in
                               ("train", "calibration", "decision", "test")])
        segs = np.concatenate([np.full(len(parts[s].numeric_history), s)
                               for s in ("train", "calibration", "decision", "test")])

        mine = numeric_baselines(full, segs, seasonal_period=12)
        ref = reference_numeric_baselines(
            [{"segment": s} for s in segs],
            {"numeric_history": full.numeric_history,
             "targets_standardized": full.targets_standardized},
            [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], 12)

        check("A.2 α 选择与主控一致", mine["ridge_alpha"] == ref["ridge_alpha"])
        check("A.2 校准 MSE 与主控一致（容差内）",
              math.isclose(mine["ridge_calibration_mse"], ref["ridge_calibration_mse"],
                           rel_tol=1e-10, abs_tol=1e-12))
        check("A.2 fit_rows / calibration_rows 与主控一致",
              mine["fit_rows"] == ref["fit_rows"]
              and mine["calibration_rows"] == ref["calibration_rows"])
        # 任务书点名"在 calibration/decision/test 比对" —— 按段分别给结论，
        # 而不是只报一个全量结论（全量通过不排除某一段整体偏移）
        for seg in ("calibration", "decision", "test"):
            mask = segs == seg
            for name in ("Last", "SeasonalNaive", "AR-Ridge"):
                a = mine["predictions"][name][mask]
                b = ref["predictions"][name][mask]
                check(f"A.2 [{seg}] {name} 逐起点逐步输出与主控一致（{int(mask.sum())} 起点）",
                      a.shape == b.shape and np.allclose(a, b, rtol=1e-10, atol=1e-12))
        for name in ("Last", "SeasonalNaive", "AR-Ridge"):
            check(f"A.2 [all] {name} 全量逐点一致",
                  np.allclose(mine["predictions"][name], ref["predictions"][name],
                              rtol=1e-10, atol=1e-12))
        check("A.2 行序与 Bundle 一致",
              mine["predictions"]["AR-Ridge"].shape == ref["predictions"]["AR-Ridge"].shape
              == full.targets_standardized.shape)

    # ---------------- A.1 数值回退旁路（走入口）----------------
    print("\n--- A.1 numeric_fallback 旁路 ---")
    from predict_test import main as pt_main

    def run_entry(tmp: Path, spec: Path, route: dict, candidate: str,
                  selection: dict | None, sig: str) -> tuple[int, Path]:
        rp = tmp / "route.json"
        rp.write_text(json.dumps(route), encoding="utf-8")
        out = tmp / "out"
        argv = ["predict_test.py", "--bundle", str(tmp / "b"), "--task", "Agriculture_h3_f1",
                "--scenario", "proxy", "--signature", sig, "--split-spec", str(spec),
                "--route", str(rp), "--route-sha256", sha256_file(rp),
                "--candidate", candidate, "--output-dir", str(out)]
        if selection is not None:
            sp = tmp / "selection.json"
            sp.write_text(json.dumps(selection), encoding="utf-8")
            argv += ["--selection-manifest", str(sp)]
        old = sys.argv; sys.argv = argv
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                code = pt_main()
        except SystemExit as exc:
            code = int(exc.code or 1)
        finally:
            sys.argv = old
        return code, out

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")
        make_bundle(tmp / "b", spec)
        sig = json.loads((tmp / "b" / "manifest.json").read_text())["signature"]
        spec_sha = sha256_file(spec)
        fb_route = {"task_id": "Agriculture_h3_f1", "fold_id": 1,
                    "selected": "numeric_fallback", "fallback": "AR-Ridge",
                    "selection_data_segments": ["cal", "dec"],
                    "split_spec_sha256": spec_sha, "bundle_signature": sig}
        # 关键负例：路由选中数值回退，却请求一个门控候选
        for request in GATE_CANDIDATES:
            code, out = run_entry(tmp, spec, fb_route, request, None, sig)
            produced = out.is_dir() and any(out.glob("*_test_predictions.csv"))
            check(f"A.1 请求门控候选 {request} 但路由选数值回退 → 拒绝且无预测文件",
                  code != 0 and not produced)
        # 正确请求：严格等于 route.fallback
        code, out = run_entry(tmp, spec, fb_route, "AR-Ridge", None, sig)
        check("A.1 请求值严格等于 route.fallback 时通过", code == 0)

        # A.3：主控基线不需要 selection manifest
        check("A.3 主控基线（AR-Ridge）无 selection manifest 也能跑", code == 0)
        man = json.loads((out / "AR_Ridge_test_manifest.json").read_text(encoding="utf-8"))
        check("A.3 回退路径标记为主控基线且无权重哈希",
              man["path"] == "master_numeric_baseline" and man["weight_hash"] is None)
        check("A.3 记录了主控基线的 α 与拟合行数",
              "ridge_alpha" in man and man["baseline_fit_rows"] > 0)

    # ---------------- A.4 两锚必填 ----------------
    print("\n--- A.4 路由两锚必填 ---")
    from predict_test import RouteRejected as _RR, check_route, load_route

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        base = {"task_id": "Agriculture_h3_f1", "fold_id": 1, "selected": "N+S+Q",
                "fallback": "AR-Ridge", "selection_data_segments": ["cal", "dec"]}
        for drop, label in (("split_spec_sha256", "split_spec_sha256"),
                            ("bundle_signature", "bundle_signature")):
            route = dict(base, split_spec_sha256="s" * 64, bundle_signature="b" * 64)
            route.pop(drop)
            p = tmp / f"r_{drop}.json"
            p.write_text(json.dumps(route), encoding="utf-8")
            r = False
            try:
                check_route(route, "Agriculture_h3_f1", 1, "N+S+Q",
                            bundle_signature="b" * 64, split_spec_sha256="s" * 64,
                            spec_gates={"N+S+Q", "N+S+Q+SF"},
                            spec_fallbacks={"Last", "SeasonalNaive", "AR-Ridge", "N"})
            except _RR as exc:
                r = label in str(exc)
            check(f"A.4 缺 {label} 锚被拒绝", r)

        route = dict(base, split_spec_sha256="s" * 64, bundle_signature="b" * 64)
        r = False
        try:
            check_route(route, "Agriculture_h3_f1", 1, "N+S+Q",
                        bundle_signature="X" * 64, split_spec_sha256="s" * 64,
                        spec_gates={"N+S+Q", "N+S+Q+SF"},
                        spec_fallbacks={"Last", "SeasonalNaive", "AR-Ridge", "N"})
        except _RR as exc:
            r = "bundle_signature" in str(exc)
        check("A.4 bundle_signature 不符被拒绝", r)
        check("A.4 两锚齐备且一致时通过",
              check_route(route, "Agriculture_h3_f1", 1, "N+S+Q",
                          bundle_signature="b" * 64,
                          split_spec_sha256="s" * 64,
                          spec_gates={"N+S+Q", "N+S+Q+SF"},
                          spec_fallbacks={"Last", "SeasonalNaive", "AR-Ridge", "N"}
                          ) == "N+S+Q")

        # 候选合法性必须对照**冻结 spec**，而不是本模块的硬编码常量 ——
        # 否则主控改了 spec 的 gate_candidates，两边会静默分歧。
        r = False
        try:
            check_route(dict(base, selected="N+S+Q+F", split_spec_sha256="s" * 64,
                             bundle_signature="b" * 64),
                        "Agriculture_h3_f1", 1, "N+S+Q+F",
                        bundle_signature="b" * 64, split_spec_sha256="s" * 64,
                        spec_gates={"N+S+Q", "N+S+Q+SF"},
                        spec_fallbacks={"Last", "SeasonalNaive", "AR-Ridge", "N"})
        except _RR as exc:
            r = "冻结 spec 的门控候选" in str(exc)
        check("A.4 selected 不在 spec 的 gate_candidates 中 → 拒绝", r)

        r = False
        try:
            check_route(dict(base, selected="numeric_fallback", fallback="NotABaseline",
                             split_spec_sha256="s" * 64, bundle_signature="b" * 64),
                        "Agriculture_h3_f1", 1, "NotABaseline",
                        bundle_signature="b" * 64, split_spec_sha256="s" * 64,
                        spec_gates={"N+S+Q"}, spec_fallbacks={"Last", "N"})
        except _RR as exc:
            r = "numeric_fallback_candidates" in str(exc)
        check("A.4 fallback 不在 spec 的 numeric_fallback_candidates 中 → 拒绝", r)

        # spec 缺候选清单时也必须硬失败，不能"没声明就放行"
        from predict_test import spec_candidate_lists
        with tempfile.TemporaryDirectory() as t2:
            bad_spec = Path(t2) / "nospec.json"
            bad_spec.write_text(json.dumps({"gate_candidates": []}), encoding="utf-8")
            r = False
            try:
                spec_candidate_lists(bad_spec)
            except _RR as exc:
                r = "gate_candidates" in str(exc)
            check("A.4 spec 未声明 gate_candidates → 拒绝", r)

            bad_spec2 = Path(t2) / "nospec2.json"
            bad_spec2.write_text(json.dumps({"gate_candidates": ["A"], "numeric_fallback_candidates": []}),
                                 encoding="utf-8")
            r = False
            try:
                spec_candidate_lists(bad_spec2)
            except _RR as exc:
                r = "numeric_fallback_candidates" in str(exc)
            check("A.4 spec 未声明 numeric_fallback_candidates → 拒绝", r)

    # ---------------- A.5 不向模型提供测试真值 ----------------
    print("\n--- A.5 测试真值不进入预测 -------------")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")
        bdir = make_bundle(tmp / "a5", spec)
        sig = json.loads((bdir / "manifest.json").read_text())["signature"]

        from bundle_reader import read_frozen_bundle
        b = read_frozen_bundle(bdir, "Agriculture_h3_f1", "proxy", sig, spec)
        view = to_inference_bundle(b.segment("test"))
        check("A.5 推理视图不含 targets", view.targets is None)
        check("A.5 推理视图不含 targets_standardized", view.targets_standardized is None)
        check("A.5 推理视图特征完整",
              len(view.numeric_history) == len(view.origin_index) == 13)

        model = BranchResidualCandidate("N+S+Q")
        model.fit_design(to_feature_bundle(b.segment("train")))
        model.select_alphas(to_feature_bundle(b.segment("train")),
                            to_feature_bundle(b.segment("calibration")))
        model.refit(merge_bundles(to_feature_bundle(b.segment("train")),
                                  to_feature_bundle(b.segment("calibration"))))
        before = model.predict(view)

        # 把磁盘上的**测试段**真值随机改掉，并重新签名。
        #
        # ⚠️ 只能改 test 段的行！直接替换整个数组会把 train/calibration 的真值
        # 一起改掉 —— 那样 α 会（正确地）变化，测到的是"训练数据被毁"，
        # 而不是"是否偷看测试真值"。第一版就踩了这个坑。
        task = bdir / "Agriculture_h3_f1"
        test_mask = b.segment_mask("test")
        rng_tamper = np.random.default_rng(99)
        for key in ("targets", "targets_standardized"):
            arr = np.load(task / f"{key}.npy")
            arr = arr.copy()
            arr[test_mask] = rng_tamper.normal(size=(int(test_mask.sum()), arr.shape[1]),
                                               ).astype(arr.dtype)
            np.save(task / f"{key}.npy", arr, allow_pickle=False)

        def resign(root: Path, spec_path: Path) -> str:
            s = json.loads(Path(spec_path).read_text(encoding="utf-8"))
            inputs = {"mode": "frozen", "split_spec_sha256": sha256_file(spec_path),
                      "split_spec": s}
            files = {p.relative_to(root).as_posix(): sha256_file(p)
                     for p in sorted(root.rglob("*"))
                     if p.is_file() and p.name != "manifest.json"}
            m = {"inputs": inputs, "signature": signature(inputs), "files": files}
            (root / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
            return m["signature"]

        sig2 = resign(bdir, spec)
        b2 = read_frozen_bundle(bdir, "Agriculture_h3_f1", "proxy", sig2, spec)
        after = model.predict(to_inference_bundle(b2.segment("test")))
        check("A.5 改动测试真值（并重新签名）后预测逐位不变",
              np.array_equal(before, after))

        # 关键：**重新跑一遍完整选路**（fit_design + select_alphas + refit），
        # 断言 α 与权重都不变。只比预测的话，用篡改前就拟合好的模型等于没测到
        # "选路过程是否依赖真值"。
        m2 = BranchResidualCandidate("N+S+Q")
        m2.fit_design(to_feature_bundle(b2.segment("train")))
        m2.select_alphas(to_feature_bundle(b2.segment("train")),
                         to_feature_bundle(b2.segment("calibration")))
        m2.refit(merge_bundles(to_feature_bundle(b2.segment("train")),
                               to_feature_bundle(b2.segment("calibration"))))
        check("A.5 重新选路后 α 完全一致",
              m2.alpha_by_group == model.alpha_by_group)
        check("A.5 重新拟合后权重逐位一致",
              np.array_equal(m2.weights, model.weights))

        # 删除测试真值文件也不应影响预测
        for key in ("targets", "targets_standardized"):
            (task / f"{key}.npy").unlink()
        view3 = to_inference_bundle({"numeric_history": b.segment("test")["numeric_history"],
                                     "semantic": b.segment("test")["semantic"],
                                     "quality": b.segment("test")["quality"],
                                     "text_available": b.segment("test")["text_available"],
                                     "origin_index": b.segment("test")["origin_index"]})
        check("A.5 测试真值缺失时仍能预测且结果不变",
              np.array_equal(model.predict(view3), before))

    # ---------------- A.3 `N` 路径单独测 ----------------
    # 任务书：`N` 属**模型候选型**数值回退，保留其选择期清单并**单独测路径**。
    # 它与主控基线（Last/SeasonalNaive/AR-Ridge）走的是**不同分支**，必须分别验证。
    print("\n--- A.3 `N` 作为数值回退的独立路径 ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")
        bdir = make_bundle(tmp / "npath", spec)
        sig = json.loads((bdir / "manifest.json").read_text())["signature"]
        spec_sha = sha256_file(spec)

        b = read_frozen_bundle(bdir, "Agriculture_h3_f1", "proxy", sig, spec)
        np_parts = {s: to_feature_bundle(b.segment(s))
                    for s in ("train", "calibration", "decision")}
        nm = BranchResidualCandidate("N")
        nm.fit_design(np_parts["train"])
        nm.select_alphas(np_parts["train"], np_parts["calibration"])
        nm.refit(merge_bundles(np_parts["train"], np_parts["calibration"]))
        n_selection = {"candidate": "N", "bundle_signature": sig,
                       "alpha_by_group": nm.alpha_by_group,
                       "weight_hash": weight_hash(nm.weights),
                       "config_sha256": "cfg-n"}

        n_route = {"task_id": "Agriculture_h3_f1", "fold_id": 1,
                   "selected": "numeric_fallback", "fallback": "N",
                   "selection_data_segments": ["cal", "dec"],
                   "split_spec_sha256": spec_sha, "bundle_signature": sig}
        rp = tmp / "route_n.json"
        rp.write_text(json.dumps(n_route), encoding="utf-8")
        sp = tmp / "selection_n.json"
        sp.write_text(json.dumps(n_selection), encoding="utf-8")
        out = tmp / "out_n"
        old = sys.argv
        sys.argv = ["predict_test.py", "--bundle", str(bdir),
                    "--task", "Agriculture_h3_f1", "--scenario", "proxy",
                    "--signature", sig, "--split-spec", str(spec),
                    "--route", str(rp), "--route-sha256", sha256_file(rp),
                    "--selection-manifest", str(sp), "--candidate", "N",
                    "--output-dir", str(out)]
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                code = pt_main()
        except SystemExit as exc:
            code = int(exc.code or 1)
        finally:
            sys.argv = old

        check("A.3 `N` 作为数值回退走清单路径，端到端成功", code == 0)
        nman = json.loads((out / "N_test_manifest.json").read_text(encoding="utf-8"))
        check("A.3 `N` 标为模型候选型回退（既非门控候选也非主控基线）",
              nman["path"] == "numeric_fallback_candidate"
              and nman["model_config"]["branches"] == ["N"])
        check("A.3 `N` 路径记录了选择期与扩展期权重哈希",
              nman["weight_hash"] is not None and nman["test_fit_weight_hash"] is not None
              and nman["weight_hash"] != nman["test_fit_weight_hash"])

        # `N` 缺清单时必须拒绝（与主控基线的差别正在于此）
        old = sys.argv
        sys.argv = ["predict_test.py", "--bundle", str(bdir),
                    "--task", "Agriculture_h3_f1", "--scenario", "proxy",
                    "--signature", sig, "--split-spec", str(spec),
                    "--route", str(rp), "--route-sha256", sha256_file(rp),
                    "--candidate", "N", "--output-dir", str(tmp / "out_n2")]
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                code2 = pt_main()
        except SystemExit as exc:
            code2 = int(exc.code or 1)
        finally:
            sys.argv = old
        check("A.3 `N` 缺选择期清单时被拒绝（区别于主控基线）", code2 != 0)

    # ---------------- 导出必须按 Bundle 行序（主控的 grid 是有序比较）----------------
    # 主控 v2.load_prediction_grid 的最终校验是 `grid != expected_grid` —— 逐元素有序比较，
    # 而 expected_grid 按 bundle["samples"] 的原始行序生成。若按"段名顺序"拼接，
    # 段序不一致时会被判 order_bad。用**交错段序**的 Bundle 才测得到这一点。
    print("\n--- 导出按 Bundle 行序（交错段序）---")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")
        bdir = make_bundle(tmp / "order", spec)
        sig = json.loads((bdir / "manifest.json").read_text())["signature"]

        # 把 samples.csv 与各数组按"dec 在前、cal 在后"重排，并重新签名
        from bundle_reader import sha256_file, signature as _sig
        task = bdir / "Agriculture_h3_f1"
        s = pd.read_csv(task / "samples.csv")
        order = np.argsort((s["segment"] == "calibration").to_numpy(), kind="stable")
        s2 = s.iloc[order].reset_index(drop=True)
        s2.to_csv(task / "samples.csv", index=False, lineterminator="\n")
        for key in ("numeric_history", "targets", "targets_standardized"):
            arr = np.load(task / f"{key}.npy")
            np.save(task / f"{key}.npy", arr[order], allow_pickle=False)
        np.save(task / "origin_index.npy",
                np.load(task / "origin_index.npy")[order], allow_pickle=False)
        for key in ("semantic", "quality", "text_available"):
            arr = np.load(task / "proxy" / f"{key}.npy")
            np.save(task / "proxy" / f"{key}.npy", arr[order], allow_pickle=False)
        sjson = json.loads(spec.read_text(encoding="utf-8"))
        inputs = {"mode": "frozen", "split_spec_sha256": sha256_file(spec), "split_spec": sjson}
        files = {p.relative_to(bdir).as_posix(): sha256_file(p)
                 for p in sorted(bdir.rglob("*"))
                 if p.is_file() and p.name != "manifest.json"}
        (bdir / "manifest.json").write_text(
            json.dumps({"inputs": inputs, "signature": _sig(inputs), "files": files}),
            encoding="utf-8")
        sig2 = _sig(inputs)

        b2 = read_frozen_bundle(bdir, "Agriculture_h3_f1", "proxy", sig2, spec)
        view = samples_order_view(b2, ["calibration", "decision"])
        segments_seen = list(view["segment"])
        wanted_order = [x for x in s2["segment"] if x in ("calibration", "decision")]
        check("交错段序下 samples_order_view 保持原行序",
              segments_seen == wanted_order)
        check("导出的段序不是简单的 cal-then-dec",
              segments_seen.index("decision") < segments_seen.index("calibration"))
        check("origin_id 与 origin_index 逐行对齐",
              all(str(o).endswith(f":o{int(i)}")
                  for o, i in zip(view["origin_id"], view["origin_index"])))

    # ---------------- 沿用：契约 / 边界 / 半段 ----------------
    print("\n--- 沿用：签名 / 边界 / 半段 ---")
    from bundle_reader import (
        BundleUnavailable, assert_grid, decision_halves, segment_bounds,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")
        bdir = make_bundle(tmp / "keep", spec)
        sig = json.loads((bdir / "manifest.json").read_text())["signature"]
        b = read_frozen_bundle(bdir, "Agriculture_h3_f1", "proxy", sig, spec)
        bounds = segment_bounds(spec, "Agriculture_h3_f1")
        check("四段边界与 spec 一致",
              bounds["train"] == (0, 20) and bounds["calibration"] == (20, 30)
              and bounds["decision"] == (30, 45) and bounds["test"] == (45, 60))
        for s in ("train", "calibration", "decision", "test"):
            assert_grid(b, s, bounds[s], 3)
        check("各段网格与 H 窗口通过", True)

        info = decision_halves(b, bounds, 3)
        check("middle = (cal_end+dec_end)//2 = 37", info["middle"] == 37)
        part = b.segment("decision")
        origins = np.asarray(part["origin_index"])
        check("前段 == origin+h<=middle",
              np.array_equal(info["first_half"]["mask"], origins + 3 <= 37))
        check("后段 == origin>=middle",
              np.array_equal(info["second_half"]["mask"], origins >= 37))
        check("半段不重叠",
              not (info["first_half"]["mask"] & info["second_half"]["mask"]).any())

        # 签名独立重算
        m = json.loads((bdir / "manifest.json").read_text())
        check("签名 == signature(inputs)",
              m["signature"] == signature(m["inputs"]))

    # ---------------- 三个入口的完整路径端到端 ----------------
    # 此前只跑过 --help 与拒绝分支；完整路径没被执行过。
    # 上一轮 predict_test 的崩溃就是靠这类端到端跑出来的。
    print("\n--- 三个入口完整路径端到端 ---")
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
                    code = module.main()
                return code
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
        check("train.py 完整路径端到端成功", code == 0)
        csvs = list(out_t.glob("*_predictions.csv"))
        check("train.py 产出预测 CSV 与 manifest",
              len(csvs) == 1 and (out_t / "N_S_Q_run_manifest.json").is_file())
        import csv as _csv
        with csvs[0].open(encoding="utf-8-sig", newline="") as fh:
            rows = list(_csv.DictReader(fh))
        # 主控 v2.load_prediction_grid 的逐条格式门槛
        check("CSV 列 == 主控 PREDICTION_COLUMNS",
              set(rows[0]) == set(MASTER_PREDICTION_COLUMNS))
        check("target_scale 全部为主控 SCALE",
              all(r["target_scale"] == MASTER_SCALE for r in rows))
        import re as _re
        check("code_commit 全部为 40 位 hex（主控要求）",
              all(_re.fullmatch(r"[0-9a-f]{40}", r["code_commit"]) for r in rows))
        check("config_sha256 全部为 64 位 hex（主控要求）",
              all(_re.fullmatch(r"[0-9a-f]{64}", r["config_sha256"]) for r in rows))
        n_origins = len({r["origin_id"] for r in rows})
        check("每起点每步恰一行",
              len(rows) == n_origins * 3
              and len({(r["origin_id"], r["step"]) for r in rows}) == len(rows))
        check("段只含 calibration/decision（不含 train/test）",
              {r["segment"] for r in rows} == {"calibration", "decision"})

        import permutation_entry as perm_entry
        out_p = tmp / "p"
        code = run_entry(perm_entry, ["permutation_entry.py", *common,
                                      "--candidate", "N+S+Q", "--nulls", "3",
                                      "--output-dir", str(out_p)])
        check("permutation_entry.py 完整路径端到端成功", code == 0)
        check("置换产出逐次 CSV 与两份 summary",
              (out_p / "null_scores.csv").is_file()
              and (out_p / "null_scores_summary.json").is_file()
              and (out_p / "permutation_summary.json").is_file())
        psum = json.loads((out_p / "permutation_summary.json").read_text(encoding="utf-8"))
        check("未满 999 次时 p 为 null（不伪填）",
              psum["p_value"] is None and psum["successful"] == 3)
        nrows = len(pd.read_csv(out_p / "null_scores.csv"))
        check("逐次日志一行一次迭代", nrows == 3)

        import audit_tables as audit_entry
        out_a = tmp / "a"
        code = run_entry(audit_entry, ["audit_tables.py", *common,
                                       "--candidate", "N+S+Q", "--output-dir", str(out_a)])
        check("audit_tables.py 完整路径端到端成功", code == 0)
        check("审计表产出两张 CSV",
              (out_a / "text_availability_audit.csv").is_file()
              and (out_a / "decision_halves_audit.csv").is_file())
        halves = pd.read_csv(out_a / "decision_halves_audit.csv")
        check("半段表含保留与剔除数", set(halves.columns) >= {"half", "n_origins", "mean_loss"})

    print(f"\n全部通过（{len(PASSED)} 项检查）")
    print("未跑（需正式 Bundle）：真实 Agriculture 训练、999 次置换、测试段预测")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1)
