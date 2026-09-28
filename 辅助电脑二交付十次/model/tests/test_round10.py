"""第十轮测试：SocialGood 零文本安全退化契约（合成数据，**不跑正式预测**）。

先跑第九轮 160 项（内含第六、七、八轮），再补第十轮 §B 的契约测试：

  B1  训练区文本覆盖为 0 → 含 S/SF 的候选必须退化为**注册表内**的数值/质量感知候选
  B2  退化后名称 / 实际特征 / 证据字段必须一致，**不得以多模态名义参赛**
  B3  退化判定只用训练区统计：接口签名里没有测试段参数（结构上读不到测试文本/真值）
  B4  合成覆盖：全零文本、稀疏文本、时间错位、全缺失质量特征、非法边界
  B5  只写契约与测试，不生成任何 SocialGood 正式预测

用法::

    .venv/Scripts/python.exe "<交付目录>/model/tests/test_round10.py"
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent
sys.path.insert(0, str(MODEL))
sys.path.insert(0, str(HERE))

import test_round9 as r9  # noqa: E402  —— 它内部会先跑第六、七、八轮
import test_round6 as r6  # noqa: E402

check = r6.check

# 冻结 spec 登记的 SocialGood 值（第十轮任务书 B）
SG_BOUNDS = [366, 458, 641, 732]
SG_INPUT_LEN, SG_HORIZON, SG_PERIOD = 24, 3, 12
REGISTRY = ("Last", "SeasonalNaive", "AR-Ridge", "N", "N+Q", "N+S+Q", "N+S+Q+SF")


def _reject(fn, *a, **kw) -> str:
    """调用应抛 ContractViolation；返回错误信息，未抛返回空串。"""
    from degradation_contract import ContractViolation
    try:
        fn(*a, **kw)
    except ContractViolation as exc:
        return str(exc)
    return ""


def main() -> int:
    print("第十轮测试（合成数据）")
    print("\n========== 先跑第九轮 160 项 ==========")
    rc9 = r9.main()
    check("第九轮 160 项全部通过", rc9 == 0)

    from degradation_contract import (
        MIN_TRAIN_TEXT_ROWS, ContractViolation, Degradation, assert_evidence_consistent,
        expected_origins, quality_is_usable, resolve_candidate, train_text_coverage,
        validate_bounds,
    )

    print("\n--- B4a. SocialGood 登记值：边界与起点数可复算 ---")
    err = _reject(validate_bounds, SG_BOUNDS, input_len=SG_INPUT_LEN, horizon=SG_HORIZON)
    check("登记的边界 [366,458,641,732] 合法", err == "")
    counts = expected_origins(SG_BOUNDS, input_len=SG_INPUT_LEN, horizon=SG_HORIZON)
    check("复算 700 / 611 / 89（与主控第十轮口径一致）",
          counts["all"] == 700 and counts["selection_region"] == 611 and counts["test"] == 89)
    check("门控起点（cal+dec）= 271", counts["gating_origins"] == 271)

    print("\n--- B4b. 非法边界必须拒绝 ---")
    check("边界非严格递增 → 拒绝",
          _reject(validate_bounds, [366, 366, 641, 732], input_len=24, horizon=3) != "")
    check("首段非正 → 拒绝",
          _reject(validate_bounds, [0, 458, 641, 732], input_len=24, horizon=3) != "")
    check("边界个数不足 → 拒绝",
          _reject(validate_bounds, [366, 458, 641], input_len=24, horizon=3) != "")
    check("train 段容不下 input_len+H → 拒绝（26 < 24+3）",
          _reject(validate_bounds, [26, 458, 641, 732], input_len=24, horizon=3) != "")

    print("\n--- B1/B2. 全零文本 → 必须退化，且证据自洽、不得称多模态 ---")
    d0 = resolve_candidate("N+S+Q", registry=REGISTRY, train_text_rows=0,
                           train_rows=counts["train"], quality_usable=True)
    check("全零文本 + N+S+Q → 退化", d0.degraded and d0.effective != "N+S+Q")
    check("退化目标在注册表内且不含语义分支",
          d0.effective in REGISTRY and not ({"S", "SF"} & set(d0.features_actually_used)))
    check("证据字段自洽（modality 只能是 numeric / quality_aware）",
          d0.modality in ("numeric", "quality_aware") and d0.reason)
    assert_evidence_consistent({"requested_candidate": d0.requested,
                                "effective_candidate": d0.effective,
                                "degraded": d0.degraded,
                                "features_actually_used": list(d0.features_actually_used),
                                "modality": d0.modality, "reason": d0.reason})
    check("按契约校验通过（不抛异常）", True)

    d1 = resolve_candidate("N+S+Q+SF", registry=REGISTRY, train_text_rows=0,
                           train_rows=counts["train"], quality_usable=True)
    check("全零文本 + N+S+Q+SF → 退化且 effective 不含 S/SF",
          d1.degraded and not ({"S", "SF"} & set(d1.features_actually_used)))

    print("\n--- B4c. 稀疏文本 / 时间错位 ---")
    sparse = resolve_candidate("N+S+Q", registry=REGISTRY,
                               train_text_rows=MIN_TRAIN_TEXT_ROWS - 1,
                               train_rows=counts["train"], quality_usable=True)
    check(f"稀疏文本（{MIN_TRAIN_TEXT_ROWS - 1} 行 < 阈值 {MIN_TRAIN_TEXT_ROWS}）→ 退化",
          sparse.degraded)
    enough = resolve_candidate("N+S+Q", registry=REGISTRY, train_text_rows=MIN_TRAIN_TEXT_ROWS,
                               train_rows=counts["train"], quality_usable=True)
    check(f"达到阈值（{MIN_TRAIN_TEXT_ROWS} 行）→ 不退化", not enough.degraded)
    # 时间错位：文本只出现在"测试段"位置 → 只把训练区切片交给契约
    rng = np.random.default_rng(7)
    full = np.zeros(700, dtype=bool)
    full[611:] = True                     # 仅测试区有文本
    train_only = full[:611]               # 契约只接受训练区数组
    rows, total = train_text_coverage(train_only)
    check("时间错位（文本只在测试区）→ 训练区覆盖 0 → 退化",
          rows == 0 and resolve_candidate("N+S+Q", registry=REGISTRY, train_text_rows=rows,
                                          train_rows=total, quality_usable=True).degraded)
    params = set(inspect.signature(resolve_candidate).parameters)
    check("B3 接口签名里没有测试段参数（结构上读不到测试文本/真值）",
          not any("test" in p.lower() for p in params))

    print("\n--- B4d. 质量特征：全缺失 / 常量 / 可用 ---")
    q_missing = np.full((50, 10), np.nan)
    q_const = np.ones((50, 10))
    q_ok = rng.normal(size=(50, 10))
    check("质量全缺失 → 不可用", quality_is_usable(q_missing) is False)
    check("质量常量 → 不可用（常量不携带信息，不许记成 quality_aware）",
          quality_is_usable(q_const) is False)
    check("质量正常 → 可用", quality_is_usable(q_ok) is True)
    d_q = resolve_candidate("N+Q", registry=REGISTRY, train_text_rows=50,
                            train_rows=50, quality_usable=False)
    check("质量不可用时 N+Q → 退化为 N", d_q.degraded and d_q.effective == "N"
          and d_q.modality == "numeric")
    d_qq = resolve_candidate("N+S+Q", registry=REGISTRY, train_text_rows=0,
                             train_rows=50, quality_usable=False)
    check("质量不可用时不得退化到 N+Q（只能是数值）", d_qq.effective == "N")
    d_ok = resolve_candidate("N+S+Q", registry=REGISTRY, train_text_rows=50,
                             train_rows=50, quality_usable=True)
    check("文本与质量都可用 → 不退化", not d_ok.degraded and d_ok.modality == "multimodal")

    print("\n--- B1 边界情形：注册表与请求值 ---")
    check("注册表为空 → 拒绝", _reject(resolve_candidate, "N+S+Q", registry=(),
                                       train_text_rows=0, train_rows=10,
                                       quality_usable=True) != "")
    check("requested 不在注册表 → 拒绝",
          _reject(resolve_candidate, "N+X", registry=("N",), train_text_rows=0,
                  train_rows=10, quality_usable=True) != "")
    check("需要退化但注册表里没有数值候选 → 拒绝",
          _reject(resolve_candidate, "N+S+Q", registry=("N+S+Q",), train_text_rows=0,
                  train_rows=10, quality_usable=True) != "")
    d_num = resolve_candidate("N", registry=REGISTRY, train_text_rows=0, train_rows=10,
                              quality_usable=False)
    check("纯数值候选无需因文本退化", not d_num.degraded and d_num.effective == "N")

    print("\n--- B2. 交付记录自洽校验（篡改任一字段都必须报错）---")
    good = {"requested_candidate": d0.requested, "effective_candidate": d0.effective,
            "degraded": d0.degraded, "features_actually_used": list(d0.features_actually_used),
            "modality": d0.modality, "reason": d0.reason}
    for field, bad in (("effective_candidate", "N+S+Q"),
                       ("degraded", False),
                       ("features_actually_used", ["N", "S", "Q"]),
                       ("modality", "multimodal")):
        rec = dict(good); rec[field] = bad
        check(f"篡改 {field} → 拒绝", _reject(assert_evidence_consistent, rec) != "")
    rec = dict(good); rec.pop("reason")
    check("退化记录缺 reason → 拒绝", _reject(assert_evidence_consistent, rec) != "")
    rec = dict(good); rec["modality"] = "numeric"; rec["features_actually_used"] = ["N"]
    check("features 与 effective 不一致 → 拒绝", _reject(assert_evidence_consistent, rec) != "")

    print("\n--- B3. 扰动测试真值不影响退化决策 ---")
    with tempfile.TemporaryDirectory() as tmpd:
        truth = Path(tmpd) / "targets.npy"
        np.save(truth, rng.normal(size=(700, SG_HORIZON)))
        before = resolve_candidate("N+S+Q", registry=REGISTRY, train_text_rows=0,
                                   train_rows=counts["train"], quality_usable=True)
        arr = np.load(truth); arr[611:] = rng.normal(size=(89, SG_HORIZON))
        np.save(truth, arr)
        after = resolve_candidate("N+S+Q", registry=REGISTRY, train_text_rows=0,
                                  train_rows=counts["train"], quality_usable=True)
        check("改坏测试真值后退化决策与目标候选完全不变", before == after)

    print("\n--- B5. 本轮不得生成 SocialGood 正式预测 ---")
    delivery = MODEL.parent
    stray = sorted(p.name for p in delivery.rglob("*SocialGood*"))
    check("交付目录里没有任何 SocialGood 预测产物", not stray)

    total = len(r6.PASSED)
    print(f"\n全部通过（{total} 项检查：第六轮 49 + 第七轮 36 + 第八轮 46 + 第九轮 29 "
          f"+ 第十轮 {total - 160}）")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1)
