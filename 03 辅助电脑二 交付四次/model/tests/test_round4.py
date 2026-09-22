"""第四轮测试（合成数据；正式 Bundle 未交付）。

覆盖主控第三次交付验收列出的 P0/P1 与任务书「必交验证」的全部负例：

  P0-1  选择期与扩展期权重**不得混比**（第三轮把扩展权重要求和选择期哈希相同）
  P0-2  路由以**文件字节 SHA256** 为锚；强制执行 route.selected；数值回退有独立路径
  P1-3  独立重算 signature(manifest.inputs)；冻结 spec 用实际字节哈希
  P1-4  正式入口传入 bounds 与 H
  P1-5  半段规则与主控一致：middle=(cal_end+dec_end)//2

用法::

    .venv/Scripts/python.exe "03 辅助电脑二 交付四次/model/tests/test_round4.py"
"""

from __future__ import annotations

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
from candidates import ALL_CANDIDATES, GATE_CANDIDATES  # noqa: E402

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

    # 真实 Bundle 的 samples.csv 里，目标窗口跨段的起点**已被数据侧剔除**。
    # 合成数据必须同样处理，否则会（正确地）触发 H 窗口拒绝，测不到想测的东西。
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


def main() -> int:
    print("第四轮测试（合成数据；正式 Bundle 未交付）")

    from bundle_reader import (
        BundleUnavailable, assert_grid, decision_halves, read_frozen_bundle,
        segment_bounds, sha256_file, signature,
    )

    # ---------------- P1-3 签名独立重算 ----------------
    print("\n--- P1-3 签名与冻结 spec 锚 ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")
        bdir = make_bundle(tmp / "good", spec)
        b = read_frozen_bundle(bdir, "Agriculture_h3_f1", "proxy",
                               json.loads((bdir / "manifest.json").read_text())["signature"], spec)
        check("签名自洽的 Bundle 可通过", b.task_id == "Agriculture_h3_f1")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")

        def tamper_inputs(root, manifest):
            # 改动 inputs 但不更新 signature —— 自洽性被破坏
            m = json.loads((root / "manifest.json").read_text())
            m["inputs"]["mode"] = "preview"
            (root / "manifest.json").write_text(json.dumps(m))

        bdir = make_bundle(tmp / "bad", spec, tamper=tamper_inputs)
        r = False
        try:
            read_frozen_bundle(bdir, "Agriculture_h3_f1", "proxy",
                               signature({"mode": "frozen"}), spec)
        except BundleUnavailable as exc:
            r = "签名不符" in str(exc)
        check("改动 inputs 而不更新 signature 被拒绝（独立重算）", r)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec_a = make_spec(tmp / "a.json")
        bdir = make_bundle(tmp / "b", spec_a)
        good_sig = json.loads((bdir / "manifest.json").read_text())["signature"]
        # 另写一份内容不同但文件名不变的 spec —— 字节哈希会变
        spec_b = make_spec(tmp / "b.json", n_rows=61)
        r = False
        try:
            read_frozen_bundle(bdir, "Agriculture_h3_f1", "proxy", good_sig, spec_b)
        except BundleUnavailable as exc:
            r = "另一份 split spec" in str(exc) or "内嵌" in str(exc)
        check("换成另一份 spec 被拒绝（字节哈希 + 内嵌对象）", r)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")
        bdir = make_bundle(tmp / "c", spec)
        sig = json.loads((bdir / "manifest.json").read_text())["signature"]
        (bdir / "Agriculture_h3_f1" / "surprise.npy").write_bytes(b"x")
        r = False
        try:
            read_frozen_bundle(bdir, "Agriculture_h3_f1", "proxy", sig, spec)
        except BundleUnavailable as exc:
            r = "清单不符" in str(exc)
        check("清单外多余文件被拒绝", r)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json", status="draft_not_for_training", approved=None)
        bdir = make_bundle(tmp / "d", spec)
        sig = json.loads((bdir / "manifest.json").read_text())["signature"]
        r = False
        try:
            read_frozen_bundle(bdir, "Agriculture_h3_f1", "proxy", sig, spec)
        except BundleUnavailable as exc:
            r = "草案" in str(exc) or "未批准" in str(exc)
        check("未冻结 spec 被拒绝", r)

    # ---------------- P1-4 边界与 H 窗口 ----------------
    print("\n--- P1-4 边界与跨段 H 窗口 ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")
        bdir = make_bundle(tmp / "e", spec)
        sig = json.loads((bdir / "manifest.json").read_text())["signature"]
        b = read_frozen_bundle(bdir, "Agriculture_h3_f1", "proxy", sig, spec)
        bounds = segment_bounds(spec, "Agriculture_h3_f1")
        check("四段边界与 spec 一致",
              bounds["train"] == (0, 20) and bounds["calibration"] == (20, 30)
              and bounds["decision"] == (30, 45) and bounds["test"] == (45, 60))
        for s in ("train", "calibration", "decision", "test"):
            assert_grid(b, s, bounds[s], 3)
        check("各段网格与 H 窗口均通过", True)

        # 跨段目标窗口：把 test 段最后一个起点挪到边界内但 origin+h 超过段末
        r = False
        try:
            assert_grid(b, "test", (45, 59), 3)
        except AssertionError as exc:
            r = "目标窗口跨出本段" in str(exc) or "边界" in str(exc)
        check("起点格式正确但跨段目标窗口被拒绝", r)

    # ---------------- P1-5 半段规则与主控一致 ----------------
    print("\n--- P1-5 半段规则 ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")
        bdir = make_bundle(tmp / "f", spec)
        sig = json.loads((bdir / "manifest.json").read_text())["signature"]
        b = read_frozen_bundle(bdir, "Agriculture_h3_f1", "proxy", sig, spec)
        bounds = segment_bounds(spec, "Agriculture_h3_f1")
        info = decision_halves(b, bounds, 3)
        check("middle = (cal_end + dec_end) // 2",
              info["middle"] == (bounds["calibration"][1] + bounds["decision"][1]) // 2)
        check("middle = (30 + 45)//2 = 37", info["middle"] == 37)
        check("半段不重叠",
              not (info["first_half"]["mask"] & info["second_half"]["mask"]).any())
        check("保留与剔除数分别记录",
              all("kept" in info[n] and "excluded" in info[n]
                  for n in ("first_half", "second_half")))

        # 前段条件 origin+h<=middle、后段条件 origin>=middle
        part = b.segment("decision")
        origins = np.asarray(part["origin_index"])
        check("前段掩码 == origin+h<=middle",
              np.array_equal(info["first_half"]["mask"], origins + 3 <= info["middle"]))
        check("后段掩码 == origin>=middle",
              np.array_equal(info["second_half"]["mask"], origins >= info["middle"]))

    # ---------------- P0-2 路由锚与强制执行 ----------------
    print("\n--- P0-2 路由锚与强制执行 ---")
    from predict_test import RouteRejected, check_route, load_route

    def route_file(path: Path, **over) -> Path:
        route = {"task_id": "Agriculture_h3_f1", "fold_id": 1, "selected": "N+S+Q",
                 "fallback": "AR-Ridge", "selection_data_segments": ["cal", "dec"]}
        route.update(over)
        path.write_text(json.dumps(route, ensure_ascii=False), encoding="utf-8")
        return path

    with tempfile.TemporaryDirectory() as tmp:
        p = route_file(Path(tmp) / "r.json")
        digest = sha256_file(p)
        check("路由以文件字节哈希通过", load_route(p, digest)["selected"] == "N+S+Q")
        # 伪造 JSON 内字段但文件字节改变 —— 字节哈希必须能检出
        r = False
        try:
            load_route(p, "0" * 64)
        except RouteRejected as exc:
            r = "字节哈希不符" in str(exc)
        check("伪造哈希（文件字节不符）被拒绝", r)

    with tempfile.TemporaryDirectory() as tmp:
        p = route_file(Path(tmp) / "r.json")
        r = False
        try:
            load_route(p, "")
        except RouteRejected:
            r = True
        check("外部哈希缺失被拒绝", r)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = route_file(tmp / "r.json")
        route = load_route(p, sha256_file(p))
        check("选中门控候选且 --candidate 相同时通过",
              check_route(route, "Agriculture_h3_f1", 1, "N+S+Q") == "N+S+Q")
        r = False
        try:
            check_route(route, "Agriculture_h3_f1", 1, "N+S+Q+SF")
        except RouteRejected as exc:
            r = "未中选候选" in str(exc)
        check("用未中选候选被拒绝", r)
        r = False
        try:
            check_route(route, "Climate_h4_f2", 1, "N+S+Q")
        except RouteRejected as exc:
            r = "任务不符" in str(exc)
        check("任务不符被拒绝", r)

        p2 = route_file(tmp / "r2.json", selected="numeric_fallback", fallback="AR-Ridge")
        route2 = load_route(p2, sha256_file(p2))
        check("numeric_fallback 时返回路由指定的回退模型",
              check_route(route2, "Agriculture_h3_f1", 1, "AR-Ridge") == "AR-Ridge")
        r = False
        try:
            check_route(route2, "Agriculture_h3_f1", 1, "Last")
        except RouteRejected as exc:
            r = "数值回退" in str(exc)
        check("数值回退被误当门控候选时被拒绝", r)

        p3 = route_file(tmp / "r3.json", selection_data_segments=["cal"])
        route3 = load_route(p3, sha256_file(p3))
        r = False
        try:
            check_route(route3, "Agriculture_h3_f1", 1, "N+S+Q")
        except RouteRejected as exc:
            r = "选择数据段" in str(exc)
        check("selection_data_segments 不正确被拒绝", r)

    # ---------------- P0-1 选择期与扩展期权重不得混比 ----------------
    print("\n--- P0-1 两阶段重拟合（不得混比）---")
    from candidates import BranchResidualCandidate
    from predict_io import weight_hash
    from train import merge_bundles

    def syn(n, seed):
        rng = np.random.default_rng(seed)
        numeric = rng.normal(size=(n, 12)).astype(np.float32)
        semantic = rng.normal(size=(n, 16)).astype(np.float32)
        quality = rng.normal(size=(n, 3)).astype(np.float32)
        mask = rng.random(n) < 0.7
        standard = numeric[:, -1, None] + rng.normal(scale=0.3, size=(n, 3)).astype(np.float32)
        return FeatureBundle(numeric_history=numeric, semantic=semantic * mask[:, None],
                             quality=quality, text_available=mask,
                             origin_index=np.arange(n, dtype=np.int64),
                             targets=standard * 5 + 20, targets_standardized=standard)

    tr, ca, de = syn(120, 1), syn(40, 2), syn(40, 3)
    model = BranchResidualCandidate("N+S+Q")
    model.fit_design(tr)
    model.select_alphas(tr, ca)

    selection_fit = merge_bundles(tr, ca)
    model.refit(selection_fit)
    sel_hash = weight_hash(model.weights)

    extended_fit = merge_bundles(tr, ca, de)
    model.refit(extended_fit)
    ext_hash = weight_hash(model.weights)

    check("扩展期权重与选择期权重不同（本就不该相同）", sel_hash != ext_hash)
    check("两个哈希都可独立记录", len(sel_hash) == 64 and len(ext_hash) == 64)
    check("样本行数分别可追溯",
          len(selection_fit.numeric_history) == 160
          and len(extended_fit.numeric_history) == 200)

    # ---------------- 数值回退模型 ----------------
    print("\n--- 数值回退模型 ---")
    from numeric_fallbacks import numeric_fallback_predict

    last = numeric_fallback_predict("Last", tr, de)
    check("Last 形状正确", last.shape == (len(de.numeric_history), 3))
    check("Last 等于窗口末值复制",
          np.allclose(last, np.repeat(de.numeric_history[:, -1:], 3, axis=1)))

    ar = numeric_fallback_predict("AR-Ridge", tr, de)
    check("AR-Ridge 形状正确且有限", ar.shape == (len(de.numeric_history), 3)
          and np.isfinite(ar).all())

    sea = numeric_fallback_predict("SeasonalNaive", tr, de, seasonal_period=12)
    check("SeasonalNaive 形状正确", sea.shape == (len(de.numeric_history), 3))

    r = False
    try:
        numeric_fallback_predict("NoSuchModel", tr, de)
    except NotImplementedError:
        r = True
    check("未实现的回退模型明确报错（不静默换模型）", r)

    # ---------------- A.4 正式入口拒绝测试（不绕过入口）----------------
    # 任务书要求的是"**正式入口**拒绝测试"。直接调 assert_grid 不算 ——
    # 第三轮的教训恰恰是"边界测试通过 ≠ 正式入口执行了检查"。
    print("\n--- A.4 正式入口拒绝（跨段窗口）---")
    import contextlib
    import io

    def rewrite_manifest(root: Path, spec_path: Path) -> str:
        """改了 Bundle 内容后重算 manifest，使签名重新自洽。"""
        s = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        inputs = {"mode": "frozen", "split_spec_sha256": sha256_file(spec_path), "split_spec": s}
        files = {p.relative_to(root).as_posix(): sha256_file(p)
                 for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        m = {"inputs": inputs, "signature": signature(inputs), "files": files}
        (root / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
        return m["signature"]

    def inject_spanning(root: Path, spec_path: Path) -> str:
        """在 train 段塞入一个目标窗口跨段的起点（origin=19, h=3 → 19+3>20）。"""
        task = root / "Agriculture_h3_f1"

        def append_first(arr):
            # concatenate(axis=0) 对 1-D（text_available）与 2-D 都成立
            return np.concatenate([arr, arr[:1]], axis=0)

        for key in ("numeric_history", "targets", "targets_standardized"):
            arr = np.load(task / f"{key}.npy")
            np.save(task / f"{key}.npy", append_first(arr), allow_pickle=False)
        for key in ("semantic", "quality", "text_available"):
            arr = np.load(task / "proxy" / f"{key}.npy")
            np.save(task / "proxy" / f"{key}.npy", append_first(arr), allow_pickle=False)
        oi = np.load(task / "origin_index.npy")
        np.save(task / "origin_index.npy", np.append(oi, 19), allow_pickle=False)
        s = pd.read_csv(task / "samples.csv")
        s.loc[len(s)] = {"task_id": "Agriculture_h3_f1", "fold_id": 1,
                         "origin_id": "Agriculture:h3:f1:o19", "origin_index": 19,
                         "segment": "train"}
        s.to_csv(task / "samples.csv", index=False, lineterminator="\n")
        return rewrite_manifest(root, spec_path)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")
        bdir = make_bundle(tmp / "span", spec)
        sig = inject_spanning(bdir, spec)

        import train as train_entry

        def run_entry(module, argv) -> int:
            old = sys.argv
            sys.argv = argv
            try:
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    code = module.main()
                return code, err.getvalue()
            except SystemExit as exc:
                return int(exc.code or 1), ""
            finally:
                sys.argv = old

        code, _ = run_entry(train_entry, [
            "train.py", "--bundle", str(bdir), "--task", "Agriculture_h3_f1",
            "--scenario", "proxy", "--signature", sig, "--split-spec", str(spec),
            "--segments", "calibration", "decision", "--candidate", "N+S+Q",
            "--output-dir", str(tmp / "out_train"),
        ])
        check("正式入口 train.py 拒绝跨段目标窗口（退出码非 0）", code != 0)

        import permutation_entry as perm_entry
        code, _ = run_entry(perm_entry, [
            "permutation_entry.py", "--bundle", str(bdir), "--task", "Agriculture_h3_f1",
            "--scenario", "proxy", "--signature", sig, "--split-spec", str(spec),
            "--candidate", "N+S+Q", "--nulls", "2",
            "--output-dir", str(tmp / "out_perm"),
        ])
        check("正式入口 permutation_entry.py 拒绝跨段目标窗口", code != 0)

        import audit_tables as audit_entry
        code, _ = run_entry(audit_entry, [
            "audit_tables.py", "--bundle", str(bdir), "--task", "Agriculture_h3_f1",
            "--scenario", "proxy", "--signature", sig, "--split-spec", str(spec),
            "--candidate", "N+S+Q", "--output-dir", str(tmp / "out_audit"),
        ])
        check("正式入口 audit_tables.py 拒绝跨段目标窗口", code != 0)

    # ---------------- A.1 路由的 spec/Bundle 锚 ----------------
    print("\n--- A.1 路由 spec/Bundle 锚 ---")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = route_file(tmp / "anchored.json", split_spec_sha256="c" * 64)
        route = load_route(p, sha256_file(p))
        r = False
        try:
            check_route(route, "Agriculture_h3_f1", 1, "N+S+Q", split_spec_sha256="d" * 64)
        except RouteRejected as exc:
            r = "split_spec_sha256" in str(exc)
        check("路由携带 spec 锚且不符时被拒绝", r)

        p2 = route_file(tmp / "ok.json", split_spec_sha256="c" * 64)
        route2 = load_route(p2, sha256_file(p2))
        name = check_route(route2, "Agriculture_h3_f1", 1, "N+S+Q", split_spec_sha256="c" * 64)
        check("路由锚相符时通过", name == "N+S+Q")
        check("已核对的锚被记录", route2["_anchors_checked"] == ["split_spec_sha256"])

        p3 = route_file(tmp / "bare.json")
        route3 = load_route(p3, sha256_file(p3))
        check_route(route3, "Agriculture_h3_f1", 1, "N+S+Q")
        check("路由无锚时记录为缺口（不假装核对过）",
              set(route3["_anchors_missing"]) == {"split_spec_sha256", "bundle_signature"})

    # ---------------- predict_test 端到端（此前从未被执行过）----------------
    # 前面的测试只覆盖 load_route / check_route 等辅助函数；入口 main() 本身没跑过。
    # 这里把 gate_candidate 与 numeric_fallback 两条路径都真跑一遍。
    print("\n--- predict_test 端到端 ---")
    from bundle_reader import read_frozen_bundle as _rfb
    from predict_io import weight_hash as _wh
    from predict_test import main as _pt_main
    from train import merge_bundles as _mb, to_feature_bundle as _tfb

    def run_predict_test(tmp: Path, spec: Path, route: dict, selection: dict,
                         candidate: str) -> tuple[int, Path, str]:
        bdir = make_bundle(tmp / "b", spec)
        sig = json.loads((bdir / "manifest.json").read_text())["signature"]
        # 占位符 @SIG@ 换成真实签名 —— 否则选择期清单的签名核对会（正确地）失败
        selection = {k: (sig if v == "@SIG@" else v) for k, v in selection.items()}
        rp = tmp / "route.json"
        rp.write_text(json.dumps(route), encoding="utf-8")
        sp = tmp / "selection.json"
        sp.write_text(json.dumps(selection), encoding="utf-8")
        out = tmp / "out"
        old = sys.argv
        sys.argv = ["predict_test.py", "--bundle", str(bdir), "--task", "Agriculture_h3_f1",
                    "--scenario", "proxy", "--signature", sig, "--split-spec", str(spec),
                    "--route", str(rp), "--route-sha256", sha256_file(rp),
                    "--selection-manifest", str(sp), "--candidate", candidate,
                    "--output-dir", str(out)]
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                code = _pt_main()
        except SystemExit as exc:
            code = int(exc.code or 1)
        finally:
            sys.argv = old
        return code, out, sig

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")
        probe = make_bundle(tmp / "probe", spec)
        psig = json.loads((probe / "manifest.json").read_text())["signature"]
        b = _rfb(probe, "Agriculture_h3_f1", "proxy", psig, spec)
        prts = {s: _tfb(b.segment(s)) for s in ("train", "calibration", "decision", "test")}
        mm = BranchResidualCandidate("N+S+Q")
        mm.fit_design(prts["train"])
        mm.select_alphas(prts["train"], prts["calibration"])
        mm.refit(_mb(prts["train"], prts["calibration"]))
        selection = {"candidate": "N+S+Q", "bundle_signature": psig,
                     "alpha_by_group": mm.alpha_by_group,
                     "weight_hash": _wh(mm.weights), "config_sha256": "cfg"}
        route = {"task_id": "Agriculture_h3_f1", "fold_id": 1, "selected": "N+S+Q",
                 "fallback": "AR-Ridge", "selection_data_segments": ["cal", "dec"]}
        code, out, _ = run_predict_test(tmp, spec, route, selection, "N+S+Q")
        check("gate_candidate 路径端到端成功（退出码 0）", code == 0)
        mpath = out / "N_S_Q_test_manifest.json"
        check("产出测试预测与清单", (out / "N_S_Q_test_predictions.csv").is_file() and mpath.is_file())
        man = json.loads(mpath.read_text(encoding="utf-8"))
        check("两个训练集合行数不同且都记录",
              man["selection_rows"] == 26 and man["extended_rows"] == 39)
        check("选择期与扩展期权重哈希不同（P0-1 的核心）",
              man["weight_hash"] != man["test_fit_weight_hash"]
              and man["weight_hash"] == _wh(mm.weights))
        check("两个训练集合的输入 SHA256 分别记录且不同",
              man["selection_input_sha256"] != man["extended_input_sha256"])
        check("模型配置完整记录",
              man["model_config"]["branches"] == ["N", "S", "Q"]
              and len(man["model_config"]["alpha_grid"]) == 7)
        check("路由无锚时记为缺口",
              set(man["route_anchors_missing"]) == {"split_spec_sha256", "bundle_signature"})
        check("预测行数 = 测试起点 × 跨度", man["n_rows"] == man["n_test_origins"] * 3)

        # 未中选候选必须被拒绝（走入口，不绕过）
        code2, _, _ = run_predict_test(tmp, spec, route, selection, "N+S+Q+SF")
        check("入口拒绝未中选候选（退出码非 0）", code2 != 0)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        spec = make_spec(tmp / "spec.json")
        route_fb = {"task_id": "Agriculture_h3_f1", "fold_id": 1,
                    "selected": "numeric_fallback", "fallback": "AR-Ridge",
                    "selection_data_segments": ["cal", "dec"]}
        code, out, _ = run_predict_test(tmp, spec, route_fb,
                                        {"candidate": "AR-Ridge",
                                         "bundle_signature": "@SIG@",
                                         "alpha_by_group": {}, "weight_hash": None}, "AR-Ridge")
        check("numeric_fallback 路径端到端成功（退出码 0）", code == 0)
        man = json.loads((out / "AR_Ridge_test_manifest.json").read_text(encoding="utf-8"))
        check("回退路径如实标记且无权重哈希",
              man["path"] == "numeric_fallback" and man["weight_hash"] is None)
        check("回退路径预测行数正确", man["n_rows"] == man["n_test_origins"] * 3)

    print(f"\n全部通过（{len(PASSED)} 项检查）")
    print("未跑（需正式 Bundle）：真实 Agriculture 训练、999 次置换、测试段预测")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1)
