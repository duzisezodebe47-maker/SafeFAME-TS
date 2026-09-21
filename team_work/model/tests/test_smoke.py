"""工程冒烟测试（任务书 §3 第一步）。

**本测试验证程序正确性，不作为科研结论。**

用本机现有的 v2 特征缓存（`data_processed/semantic_features/`）与 Time-MMD 原始数值，
检查：形状（含 H=1）、分支隔离、非有限值拒绝、掩码行为、确定性与持久化往返。

用法::

    .venv/Scripts/python.exe team_work/model/tests/test_smoke.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "team_work" / "model"))

from data_utils import load_time_ordered_frame  # noqa: E402

from branches import FeatureBundle  # noqa: E402
from candidates import CANDIDATES, BranchResidualCandidate  # noqa: E402

SEMANTIC_ROOT = ROOT / "data_processed" / "semantic_features"
NUMERIC_ROOT = ROOT / "references" / "external" / "Time-MMD" / "numerical"

PASSED: list[str] = []


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    PASSED.append(label)
    print(f"  ok  {label}")


def build_bundle(
    domain: str, input_len: int, horizon: int, first: int, count: int
) -> tuple[FeatureBundle, np.ndarray]:
    """构造一个小样本 FeatureBundle。真实数据，非合成。

    ``first``/``count`` 指定使用的起点区间 —— 各段必须互不重叠，
    否则会构成泄漏。本机 v2 缓存的起点下限是 52（按 input_len=52 构建）。
    """
    frame, _ = load_time_ordered_frame(NUMERIC_ROOT / domain / f"{domain}.csv")
    raw = pd.to_numeric(frame["OT"], errors="coerce").to_numpy(float)
    if np.isnan(raw).any():
        raise ValueError(f"{domain} 含缺失值")

    cache = np.load(SEMANTIC_ROOT / f"{domain}.npz")
    available = cache["origin_index"].ravel()
    positions = {int(v): i for i, v in enumerate(available)}

    origins = available[first : first + count]
    if len(origins) != count:
        raise ValueError(f"{domain} 起点不足: 需要 {count}，实得 {len(origins)}")
    if origins.min() < input_len:
        raise ValueError(f"起点 {origins.min()} 小于 input_len={input_len}")

    windows = np.stack([raw[int(o) - input_len : int(o)] for o in origins]).astype(np.float32)
    targets = np.stack([raw[int(o) : int(o) + horizon] for o in origins]).astype(np.float32)
    index = [positions[int(o)] for o in origins]

    bundle = FeatureBundle(
        x=windows,
        report=cache["report_embedding"][index].astype(np.float32),
        search=cache["search_embedding"][index].astype(np.float32),
        quality=cache["quality"][index].astype(np.float32),
        origins=origins.astype(np.int64),
    )
    return bundle, targets


def fit_candidate(names: tuple[str, ...], train: FeatureBundle, y_train, cal: FeatureBundle, y_cal):
    model = BranchResidualCandidate(names)
    model.fit_design(train)
    model.select_alphas(train, cal, y_train, y_cal, passes=1)
    fit_bundle = FeatureBundle(
        x=np.r_[train.x, cal.x],
        report=np.r_[train.report, cal.report],
        search=np.r_[train.search, cal.search],
        quality=np.r_[train.quality, cal.quality],
        origins=np.r_[train.origins, cal.origins],
    )
    model.refit(fit_bundle, np.r_[y_train, y_cal])
    return model


def run(domain: str, input_len: int, horizon: int) -> None:
    # 三段起点互不重叠（冒烟用的小切片，不构成协议划分）
    train, y_train = build_bundle(domain, input_len, horizon, 0, 400)
    cal, y_cal = build_bundle(domain, input_len, horizon, 400, 120)
    test, _ = build_bundle(domain, input_len, horizon, 520, 120)

    tag = f"{domain}/L{input_len}/H{horizon}"
    print(f"\n--- {tag} ---")

    # 1. 形状
    for name, branches in CANDIDATES.items():
        model = fit_candidate(branches, train, y_train, cal, y_cal)
        pred = model.predict(test)
        check(f"{tag} {name} 预测形状 (n,H)", pred.shape == (len(test.x), horizon))
        check(f"{tag} {name} 预测有限", np.isfinite(pred).all())
        check(f"{tag} {name} 参数已冻结", model.weights is not None)

    # 2. 分支隔离：禁用 S 时，改变语义向量不得改变输出
    model_no_s = fit_candidate(("N", "Q"), train, y_train, cal, y_cal)
    before = model_no_s.predict(test)
    perturbed = FeatureBundle(
        test.x,
        np.random.default_rng(0).normal(size=test.report.shape).astype(np.float32),
        np.random.default_rng(1).normal(size=test.search.shape).astype(np.float32),
        test.quality,
        test.origins,
    )
    check(f"{tag} 语义分支关闭时改变语义向量不影响输出",
          np.allclose(before, model_no_s.predict(perturbed), atol=0.0))

    # 3. 质量分支隔离：禁用 Q 时改变质量特征不影响输出
    model_no_q = fit_candidate(("N", "S"), train, y_train, cal, y_cal)
    before_q = model_no_q.predict(test)
    perturbed_q = FeatureBundle(test.x, test.report, test.search,
                                test.quality * 3.0 + 7.0, test.origins)
    check(f"{tag} 质量分支关闭时改变质量特征不影响输出",
          np.allclose(before_q, model_no_q.predict(perturbed_q), atol=0.0))

    # 4. 启用分支时，改变对应输入必须影响输出（否则隔离测的是"分支没接上"）
    model_full = fit_candidate(("N", "S"), train, y_train, cal, y_cal)
    before_f = model_full.predict(test)
    check(f"{tag} 语义分支启用时改变语义向量会改变输出",
          not np.allclose(before_f, model_full.predict(perturbed)))

    # 5. 非有限值拒绝
    bad = FeatureBundle(test.x.copy(), test.report.copy(), test.search.copy(),
                        test.quality.copy(), test.origins)
    bad.x[0, 0] = np.inf
    rejected = False
    try:
        model_full.predict(bad)
    except AssertionError:
        rejected = True
    check(f"{tag} 非有限输入被拒绝", rejected)

    # 6. 确定性
    m1 = fit_candidate(("N", "S", "Q"), train, y_train, cal, y_cal).predict(test)
    m2 = fit_candidate(("N", "S", "Q"), train, y_train, cal, y_cal).predict(test)
    check(f"{tag} 相同输入重复拟合输出一致", np.array_equal(m1, m2))

    # 7. 持久化往返
    # Windows 注意：np.load 返回的 NpzFile 持有文件句柄，必须显式关闭，
    # 否则后续删除该文件会 PermissionError（WinError 32）。
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "pred.npz"
        np.savez_compressed(path, pred=m1, origins=test.origins)
        with np.load(path) as restored:
            same = (np.array_equal(restored["pred"], m1)
                    and np.array_equal(restored["origins"], test.origins))
        check(f"{tag} 预测可持久化并读回", same)

    # 8. 分支贡献可追溯且加和一致
    # 容差说明：本模块全链路走 float32（与现有管线一致，见 branches.py 注）。
    # 逐分支分块矩阵乘后求和，与整体 design @ weights 相比存在 float32 舍入差，
    # 实测 ~5e-7（float32 eps 量级，非算法差异 —— 已验证 contributions 与
    # 分块乘逐位相同、predict 与手工计算逐位相同）。
    # 容差取项目既有的 1e-4 相对口径（docs/21:45）。
    contrib = model_full.contributions(test)
    total = sum(contrib.values())
    check(f"{tag} 分支贡献之和 == 预测 - Last",
          np.allclose(total, before_f - test.x[:, -1, None], rtol=1e-5, atol=1e-4))
    check(f"{tag} 贡献仅含已启用分支", set(contrib) == {"N", "S"})


def main() -> int:
    print("工程冒烟测试（验证程序，不作为科研结论）")
    print(f"数据: {SEMANTIC_ROOT.relative_to(ROOT)} + {NUMERIC_ROOT.relative_to(ROOT)}")

    # 本机 v2 缓存按 input_len=52（周频组）构建，故冒烟用 52
    run("Climate", 52, 4)     # 常规跨度
    run("Climate", 52, 1)     # H=1 —— 历史广播缺陷的回归检查
    run("Energy", 52, 12)     # 长跨度 + 另一领域

    print(f"\n全部通过（{len(PASSED)} 项检查）")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1)
