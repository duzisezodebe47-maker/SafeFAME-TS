"""第三轮测试（合成数据；正式 Bundle 未交付）。

重点覆盖主控第三轮任务书 A 部分的修复项：

  A.1  读取器核对**全部且只有**清单文件；split_spec_sha256 核对；适配层同样校验
  A.2  origin_id 的**领域/跨度/折**与 task_id 一致（"索引对但领域错"必须拒绝）
  A.4  **故障注入**：第 2 次失败、第 3 次成功时，种子与损失绝不错配
  A.4  requested 与 successful 均为 999 才算 p；有失败一律不可用
  A.5  决策半段按目标时间边界切；circular-block 真正作为块长使用
  A.6  predict_test 拒绝未冻结路由 / 签名不符 / 权重哈希不符

用法::

    .venv/Scripts/python.exe "03 辅助电脑二 交付三次/model/tests/test_round3.py"
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
from candidates import (  # noqa: E402
    ALL_CANDIDATES, DIAGNOSTIC_CANDIDATES, GATE_CANDIDATES, PROTOCOL_ALPHAS,
    BranchResidualCandidate,
)
from permutation import (  # noqa: E402
    ROW_PERMUTATIONS, NullIteration, NullResult, row_permutation_null, write_null,
)
from predict_io import PredictionWriter, config_sha256  # noqa: E402

PASSED: list[str] = []


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    PASSED.append(label)
    print(f"  ok  {label}")


def synthetic_bundle(n, input_len, horizon, *, semantic_dim=768, quality_dim=10,
                     text_frac=0.7, seed=7) -> FeatureBundle:
    rng = np.random.default_rng(seed)
    numeric = rng.normal(size=(n, input_len)).astype(np.float32)
    semantic = rng.normal(size=(n, semantic_dim)).astype(np.float32)
    quality = rng.normal(size=(n, quality_dim)).astype(np.float32)
    text_available = rng.random(n) < text_frac
    base = numeric[:, -1, None] * np.ones((1, horizon), dtype=np.float32)
    standard = (base + rng.normal(scale=0.3, size=(n, horizon))).astype(np.float32)
    semantic = semantic * text_available[:, None]
    return FeatureBundle(
        numeric_history=numeric, semantic=semantic, quality=quality,
        text_available=text_available,
        origin_index=np.arange(1000, 1000 + n, dtype=np.int64),
        targets=standard * 5 + 20, targets_standardized=standard)


def fit_candidate(name, train, cal):
    from train import to_feature_bundle  # noqa: F401  (保持导入路径一致)
    model = BranchResidualCandidate(name)
    model.fit_design(train)
    model.select_alphas(train, cal)
    model.refit(FeatureBundle(
        numeric_history=np.r_[train.numeric_history, cal.numeric_history],
        semantic=np.r_[train.semantic, cal.semantic],
        quality=np.r_[train.quality, cal.quality],
        text_available=np.r_[train.text_available, cal.text_available],
        origin_index=np.r_[train.origin_index, cal.origin_index],
        targets=np.r_[train.targets, cal.targets],
        targets_standardized=np.r_[train.targets_standardized, cal.targets_standardized]))
    return model


def make_fake_bundle(root: Path, *, status="frozen", task_id="Agriculture_h3_f1",
                     n=12, horizon=3, extra_file=False):
    root.mkdir(parents=True, exist_ok=True)
    task = root / task_id
    task.mkdir(exist_ok=True)
    rng = np.random.default_rng(5)
    domain, hh, ff = task_id.split("_")[0], int(task_id.split("_h")[1].split("_")[0]), int(task_id.split("_f")[1])

    for key, val in {
        "numeric_history": rng.normal(size=(n, 6)).astype(np.float32),
        "origin_index": np.arange(100, 100 + n, dtype=np.int64),
        "targets": rng.normal(size=(n, horizon)).astype(np.float32),
        "targets_standardized": rng.normal(size=(n, horizon)).astype(np.float32),
    }.items():
        np.save(task / f"{key}.npy", val, allow_pickle=False)
    scen = task / "proxy"
    scen.mkdir(exist_ok=True)
    for key, val in {
        "semantic": rng.normal(size=(n, 16)).astype(np.float32),
        "quality": rng.normal(size=(n, 3)).astype(np.float32),
        "text_available": (rng.random(n) < 0.7),
    }.items():
        np.save(scen / f"{key}.npy", val, allow_pickle=False)

    rows = [{"task_id": task_id, "fold_id": ff,
             "origin_id": f"{domain}:h{hh}:f{ff}:o{idx}", "origin_index": int(idx),
             "segment": ("train" if i < n // 2 else "decision"),
             "target_start_time": f"2020-{(i % 12) + 1:02d}-01",
             "target_end_time": f"2020-{(i % 12) + 1:02d}-15"}
            for i, idx in enumerate(np.arange(100, 100 + n))]
    pd.DataFrame(rows).to_csv(task / "samples.csv", index=False, lineterminator="\n")

    # schema.json 必须先写 —— 契约的 data_bundle_required 把它列为必需项，
    # 因此它要在 manifest 的 files 清单里（manifest.json 自身除外，无法自列）。
    (root / "schema.json").write_text(json.dumps({"status": status}), encoding="utf-8")

    from bundle_reader import sha256_file
    files = {p.relative_to(root).as_posix(): sha256_file(p)
             for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"}
    (root / "manifest.json").write_text(json.dumps({
        "signature": "SIG-TEST", "files": files,
        "inputs": {"split_spec_sha256": "SPEC-OK", "mode": "frozen"},
    }), encoding="utf-8")

    # 多余文件必须在 manifest 写完之后再加 —— 否则它会被列进清单，
    # 就不构成"清单外"了（这正是第一次写错的地方）
    if extra_file:
        (task / "surprise_extra.npy").write_bytes(b"not-listed")
    return root


def main() -> int:
    print("第三轮测试（合成数据；正式 Bundle 未交付）")

    # ---------------- 协议与候选分类 ----------------
    print("\n--- 协议 ---")
    check("门控候选恰为两条", GATE_CANDIDATES == ("N+S+Q", "N+S+Q+SF"))
    check("N+S+Q+F 是诊断消融", "N+S+Q+F" in DIAGNOSTIC_CANDIDATES)
    check("α 为协议 7 档", PROTOCOL_ALPHAS == (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0))
    from train import RUNNABLE_SCENARIOS
    check("complete_source 不作为可运行情景",
          "complete_source" not in RUNNABLE_SCENARIOS
          and set(RUNNABLE_SCENARIOS) == {"proxy", "conservative_lag"})

    # ---------------- A.4 故障注入：种子与损失绝不错配 ----------------
    print("\n--- A.4 故障注入（第 2 次失败、第 3 次成功）---")
    # 直接构造一个含失败的 NullResult，验证 CSV 的配对关系
    result = NullResult(candidate="N+S+Q", segment="decision", requested=5, iterations=[
        NullIteration(0, 2026000, 1.0),
        NullIteration(1, 2026001, 2.0),
        NullIteration(2, 2026002, None, "RuntimeError: injected failure"),
        NullIteration(3, 2026003, 4.0),
        NullIteration(4, 2026004, 5.0),
    ])
    check("成功 4 次 / 失败 1 次", len(result.successful) == 4 and len(result.failures) == 1)
    check("失败迭代的种子被准确记录", result.failures[0].seed == 2026002)
    check("p 值因存在失败而不可用", result.p_value(0.5) is None)
    # requested=5 未达契约下限，此时原因应指向"请求次数不足"（更根本的那条）
    check("请求不足时原因指向请求次数",
          "请求次数 5" in (result.unavailable_reason() or ""))
    # 请求满 999 但有失败时，原因应指向失败
    full_with_fail = NullResult(candidate="x", segment="decision", requested=ROW_PERMUTATIONS,
                                iterations=[NullIteration(i, i, float(i))
                                            for i in range(ROW_PERMUTATIONS - 1)]
                                + [NullIteration(999, 999, None, "boom")])
    check("达契约下限但有失败时原因指向失败",
          "1 次置换失败" in (full_with_fail.unavailable_reason() or ""))

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "null_scores.csv"
        write_null(csv_path, result, observed=3.0)
        rows = pd.read_csv(csv_path)
        check("CSV 行数 = 迭代数（成功与失败都在）", len(rows) == 5)
        check("CSV 列含 status", "status" in rows.columns)
        # 关键：第 2 行（iteration=2）必须是 failed，且不携带损失
        row2 = rows[rows.iteration == 2].iloc[0]
        check("第 2 次迭代标记为 failed", row2["status"] == "failed")
        check("失败行不带损失", pd.isna(row2["loss"]))
        check("失败行带可定位错误", "injected failure" in str(row2["error"]))
        # 关键：iteration=3 的损失必须是 4.0（而非被第 2 次的失败挤位）
        row3 = rows[rows.iteration == 3].iloc[0]
        check("第 3 次迭代的损失未错配（4.0）", float(row3["loss"]) == 4.0)
        check("第 3 次迭代的种子正确", int(row3["seed"]) == 2026003)

    # ---------------- A.4 p 值门槛 ----------------
    print("\n--- A.4 p 值门槛 ---")
    full = NullResult(candidate="x", segment="decision", requested=ROW_PERMUTATIONS,
                      iterations=[NullIteration(i, i, float(i)) for i in range(ROW_PERMUTATIONS)])
    check("999 次全成功时 p 可用", full.p_value(500.0) is not None)
    short = NullResult(candidate="x", segment="decision", requested=5,
                       iterations=[NullIteration(i, i, float(i)) for i in range(5)])
    check("请求次数不足 999 时 p 不可用", short.p_value(1.0) is None)
    one_fail = NullResult(candidate="x", segment="decision", requested=ROW_PERMUTATIONS,
                          iterations=[NullIteration(i, i, float(i)) for i in range(ROW_PERMUTATIONS - 1)]
                          + [NullIteration(999, 999, None, "boom")])
    check("999 次中有 1 次失败时 p 也不可用", one_fail.p_value(1.0) is None)

    # ---------------- A.5 循环移位真正使用块长 ----------------
    print("\n--- A.5 循环移位块长 ---")
    t = synthetic_bundle(90, 8, 3, semantic_dim=12, quality_dim=3, seed=41)
    c = synthetic_bundle(30, 8, 3, semantic_dim=12, quality_dim=3, seed=42)
    d = synthetic_bundle(30, 8, 3, semantic_dim=12, quality_dim=3, seed=43)
    from permutation import circular_shift_null
    a7 = circular_shift_null("N+S+Q", t, c, d, count=3, seed=1, circular_block=7, progress_every=0)
    a2 = circular_shift_null("N+S+Q", t, c, d, count=3, seed=1, circular_block=2, progress_every=0)
    check("块长不同会产生不同的移位结果",
          not np.allclose(np.array(a7.losses), np.array(a2.losses)))
    check("块长为 1 时也接受（等价逐点循环）",
          len(circular_shift_null("N+S+Q", t, c, d, count=2, seed=1,
                                  circular_block=1, progress_every=0).successful) == 2)

    # ---------------- A.1 / A.2 Bundle 读取与网格校验 ----------------
    print("\n--- A.1 / A.2 Bundle 校验 ---")
    from bundle_reader import (
        BundleUnavailable, assert_grid, decision_halves_by_target_time, read_frozen_bundle,
    )

    with tempfile.TemporaryDirectory() as tmp:
        good = make_fake_bundle(Path(tmp) / "good")
        b = read_frozen_bundle(good, "Agriculture_h3_f1", "proxy", "SIG-TEST",
                               split_spec_sha256="SPEC-OK")
        check("冻结 Bundle 可读取", b.task_id == "Agriculture_h3_f1")
        assert_grid(b, "decision")
        check("合法网格通过", True)

    with tempfile.TemporaryDirectory() as tmp:
        r = False
        try:
            read_frozen_bundle(make_fake_bundle(Path(tmp) / "extra", extra_file=True),
                               "Agriculture_h3_f1", "proxy", "SIG-TEST")
        except BundleUnavailable as exc:
            r = "多余文件" in str(exc)
        check("A.1 清单外多余文件被拒绝", r)

    with tempfile.TemporaryDirectory() as tmp:
        r = False
        try:
            read_frozen_bundle(make_fake_bundle(Path(tmp) / "spec"),
                               "Agriculture_h3_f1", "proxy", "SIG-TEST",
                               split_spec_sha256="SPEC-WRONG")
        except BundleUnavailable as exc:
            r = "split_spec_sha256 不符" in str(exc)
        check("A.1 split_spec_sha256 不符被拒绝", r)

    with tempfile.TemporaryDirectory() as tmp:
        b = read_frozen_bundle(make_fake_bundle(Path(tmp) / "df"),
                               "Agriculture_h3_f1", "proxy", "SIG-TEST")
        # 索引数字仍然正确，但领域与折被改错
        b.samples.loc[b.samples.index[0], "origin_id"] = "Climate:h3:f9:o100"
        r = False
        try:
            assert_grid(b, "train")
        except AssertionError as exc:
            r = "领域" in str(exc) or "折号" in str(exc)
        check("A.2 索引正确但领域/折错被拒绝", r)

    with tempfile.TemporaryDirectory() as tmp:
        b = read_frozen_bundle(make_fake_bundle(Path(tmp) / "df2"),
                               "Agriculture_h3_f1", "proxy", "SIG-TEST")
        b.samples.loc[b.samples.index[0], "origin_id"] = "Agriculture:h9:f1:o100"
        r = False
        try:
            assert_grid(b, "train")
        except AssertionError as exc:
            r = "跨度" in str(exc)
        check("A.2 索引正确但跨度错被拒绝", r)

    # ---------------- A.5 决策半段按目标时间切 ----------------
    print("\n--- A.5 决策半段切法 ---")
    with tempfile.TemporaryDirectory() as tmp:
        b = read_frozen_bundle(make_fake_bundle(Path(tmp) / "halves"),
                               "Agriculture_h3_f1", "proxy", "SIG-TEST")
        masks = decision_halves_by_target_time(b, "decision")
        check("切出两个半段", set(masks) == {"first_half", "second_half"})
        check("两半段互不重叠", not (masks["first_half"] & masks["second_half"]).any())
        check("两半段均非空",
              masks["first_half"].any() and masks["second_half"].any())

    with tempfile.TemporaryDirectory() as tmp:
        b = read_frozen_bundle(make_fake_bundle(Path(tmp) / "notime"),
                               "Agriculture_h3_f1", "proxy", "SIG-TEST")
        b.samples = b.samples.drop(columns=["target_end_time"])
        r = False
        try:
            decision_halves_by_target_time(b, "decision")
        except BundleUnavailable as exc:
            r = "target_end_time" in str(exc)
        check("缺少时间列时硬失败（不退回按个数切）", r)

    # ---------------- A.6 predict_test 拒绝条件 ----------------
    print("\n--- A.6 predict_test 拒绝条件 ---")
    from predict_test import RouteNotFrozen, load_frozen_route, weight_hash

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "route.json"
        p.write_text(json.dumps({"status": "draft", "route_sha256": "abc"}), encoding="utf-8")
        r = False
        try:
            load_frozen_route(p, None)
        except RouteNotFrozen as exc:
            r = "未冻结" in str(exc)
        check("A.6 路由未冻结被拒绝", r)

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "route.json"
        p.write_text(json.dumps({"status": "frozen", "route_sha256": "abc"}), encoding="utf-8")
        r = False
        try:
            load_frozen_route(p, "different")
        except RouteNotFrozen as exc:
            r = "哈希不符" in str(exc)
        check("A.6 路由哈希不符被拒绝", r)
        check("A.6 冻结路由可通过", load_frozen_route(p, None)["status"] == "frozen")

    w1 = np.arange(12, dtype=float).reshape(3, 4)
    w2 = w1.copy()
    w2[0, 0] += 1e-9
    check("A.6 权重哈希对改动敏感", weight_hash(w1) != weight_hash(w2))
    check("A.6 权重哈希对同值稳定", weight_hash(w1) == weight_hash(w1.copy()))

    # ---------------- 保留下来的核心行为（第二轮已验，回归确认）----------------
    print("\n--- 回归：掩码 / H=1 / 契约 ---")
    train = synthetic_bundle(200, 12, 4, semantic_dim=16, quality_dim=3, seed=51)
    cal = synthetic_bundle(60, 12, 4, semantic_dim=16, quality_dim=3, seed=52)
    test = synthetic_bundle(60, 12, 4, semantic_dim=16, quality_dim=3, seed=53)
    mask = ~test.text_available
    for name in ("N+S+Q", "N+S+Q+SF"):
        m = fit_candidate(name, train, cal)
        tampered = FeatureBundle(
            numeric_history=test.numeric_history, semantic=test.semantic.copy(),
            quality=test.quality, text_available=test.text_available,
            origin_index=test.origin_index, targets=test.targets,
            targets_standardized=test.targets_standardized)
        tampered.semantic[mask] = np.random.default_rng(3).normal(
            size=(int(mask.sum()), test.semantic.shape[1]))
        check(f"{name} 改无文本行语义不影响其预测",
              np.allclose(m.predict(test)[mask], m.predict(tampered)[mask]))
        check(f"{name} 无文本行 S 贡献严格为零",
              np.allclose(m.contributions(tampered)["S"][mask], 0.0, atol=0.0))

    t1 = synthetic_bundle(120, 12, 1, semantic_dim=16, quality_dim=3, seed=61)
    c1 = synthetic_bundle(40, 12, 1, semantic_dim=16, quality_dim=3, seed=62)
    e1 = synthetic_bundle(40, 12, 1, semantic_dim=16, quality_dim=3, seed=63)
    check("H=1 返回 (n,1)",
          fit_candidate("N+S+Q", t1, c1).predict(e1).shape == (len(e1.numeric_history), 1))


    print(f"\n全部通过（{len(PASSED)} 项检查）")
    print("未跑（需正式 Bundle）：真实 Agriculture 训练、999 次置换、测试段预测")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1)
