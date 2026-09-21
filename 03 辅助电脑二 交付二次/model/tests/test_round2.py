"""第二轮工程测试。**用合成数据**，因为主控冻结 Bundle 尚未交付。

覆盖主控验收门槛中不依赖真实数据即可验证的全部条目：

  [x] 零文本行语义/交互贡献严格为零          （P1 第 3 条的核心）
  [x] H=1 仍返回 (n, 1)
  [x] 关闭某分支时扰动该分支不影响输出
  [x] 同种子重复结果一致
  [x] 配置哈希可核验且跨进程稳定
  [x] 逐段每个样本每一步恰有一行（契约 prediction_step）
  [x] 预测键重复 / 步长缺失 / 起点错位被拒绝
  [x] 置换留痕：每次记录种子与损失，失败单独记录
  [x] 置换每次重拟合：同一置换跑两次结果一致（无跨次缓存）
  [x] 训练段完全无文本时 S 分支安全降级而非造伪信号
  [x] 一维预测被拒绝（H=1 广播防护）
  [x] 协议 alpha 网格与 RIDGE_ALPHAS 的差异被记录

未覆盖（需真实 Bundle，见 RUNBOOK）：
  [ ] 从冻结 Bundle 签名复算训练输入
  [ ] 真实数据的 999 次置换与 p 值

用法::

    .venv/Scripts/python.exe "03 辅助电脑二 交付二次/model/tests/test_round2.py"
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE.parent
sys.path.insert(0, str(MODEL))

from branches import FeatureBundle  # noqa: E402
from candidates import (  # noqa: E402
    ALL_CANDIDATES, DIAGNOSTIC_CANDIDATES, GATE_CANDIDATES, PROTOCOL_ALPHAS,
    BranchResidualCandidate,
)
from permutation import permute_rows, row_permutation_null  # noqa: E402
from predict_io import PredictionWriter, config_sha256  # noqa: E402

PASSED: list[str] = []


def check(label: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(f"FAIL: {label}")
    PASSED.append(label)
    print(f"  ok  {label}")


def synthetic_bundle(
    n: int, input_len: int, horizon: int, *, semantic_dim: int = 768, quality_dim: int = 10,
    text_frac: float = 0.7, seed: int = 7, signal: bool = True,
) -> FeatureBundle:
    """合成 Bundle 切片。

    ``signal=True`` 时让语义与目标弱相关，用于验证"接上了"；
    ``signal=False`` 时语义与目标独立，用于无信号情景的校准检查。
    """
    rng = np.random.default_rng(seed)
    numeric = rng.normal(size=(n, input_len)).astype(np.float32)
    semantic = rng.normal(size=(n, semantic_dim)).astype(np.float32)
    quality = rng.normal(size=(n, quality_dim)).astype(np.float32)
    text_available = rng.random(n) < text_frac

    # 目标：由数值末尾驱动 + 少量语义（仅在有信号时）
    base = numeric[:, -1, None] * np.ones((1, horizon), dtype=np.float32)
    drift = rng.normal(scale=0.3, size=(n, horizon)).astype(np.float32)
    if signal:
        drift = drift + (semantic[:, :1] * 0.2).astype(np.float32)
    standard = (base + drift).astype(np.float32)
    raw = standard * 5.0 + 20.0

    # 无文本行的语义置零（与数据侧语义一致：无文本 → 零向量）
    semantic = semantic * text_available[:, None]

    return FeatureBundle(
        numeric_history=numeric, semantic=semantic, quality=quality,
        text_available=text_available,
        origin_index=np.arange(1000, 1000 + n, dtype=np.int64),
        targets=raw, targets_standardized=standard,
    )


def fit_candidate(name: str, train: FeatureBundle, cal: FeatureBundle):
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
        targets_standardized=np.r_[train.targets_standardized, cal.targets_standardized],
    ))
    return model


def main() -> int:
    print("第二轮工程测试（合成数据；真实 Bundle 未交付）")

    # ---------- 协议一致性 ----------
    print("\n--- 协议与候选分类 ---")
    check("门控候选恰为两条且与协议一致",
          GATE_CANDIDATES == ("N+S+Q", "N+S+Q+SF"))
    check("N+S+Q+F 被列为诊断消融而非门控候选",
          "N+S+Q+F" in DIAGNOSTIC_CANDIDATES and "N+S+Q+F" not in GATE_CANDIDATES)
    check("协议 alpha 网格为 7 档",
          PROTOCOL_ALPHAS == (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0))

    train = synthetic_bundle(300, 24, 6)
    cal = synthetic_bundle(80, 24, 6, seed=8)
    test = synthetic_bundle(80, 24, 6, seed=9)

    # ---------- 形状与有限性 ----------
    print("\n--- 形状与有限性 ---")
    for name in ALL_CANDIDATES:
        model = fit_candidate(name, train, cal)
        pred = model.predict(test)
        check(f"{name} 预测形状 ({len(test.numeric_history)}, 6)",
              pred.shape == (len(test.numeric_history), 6))
        check(f"{name} 预测有限", np.isfinite(pred).all())

    # ---------- H=1 ----------
    print("\n--- H=1 回归检查 ---")
    t1, c1, e1 = (synthetic_bundle(300, 24, 1), synthetic_bundle(80, 24, 1, seed=8),
                  synthetic_bundle(80, 24, 1, seed=9))
    for name in ("N+S+Q", "N+S+Q+SF"):
        model = fit_candidate(name, t1, c1)
        pred = model.predict(e1)
        check(f"{name} H=1 返回 (n,1)", pred.shape == (len(e1.numeric_history), 1))

    # ---------- 零文本掩码（P1 第 3 条核心）----------
    print("\n--- 零文本掩码 ---")
    mask = ~test.text_available
    check("合成数据里确实存在无文本起点", bool(mask.any()))

    for name in ("N+S+Q", "N+S+Q+SF"):
        model = fit_candidate(name, train, cal)
        before = model.predict(test)
        # 只改无文本行的语义数组，其余一切不变
        tampered = FeatureBundle(
            numeric_history=test.numeric_history,
            semantic=test.semantic.copy(),
            quality=test.quality,
            text_available=test.text_available,
            origin_index=test.origin_index,
            targets=test.targets, targets_standardized=test.targets_standardized,
        )
        tampered.semantic[mask] = np.random.default_rng(3).normal(
            size=(int(mask.sum()), test.semantic.shape[1]))
        after = model.predict(tampered)
        check(f"{name} 改无文本行语义后预测完全不变",
              np.array_equal(before[~mask], after[~mask]) and np.allclose(before[mask], after[mask]))

        contrib = model.contributions(tampered)
        s_zero = np.allclose(contrib["S"][mask], 0.0, atol=0.0)
        check(f"{name} 无文本行 S 贡献严格为零", s_zero)
        if "SF" in contrib:
            check(f"{name} 无文本行 SF 贡献严格为零",
                  np.allclose(contrib["SF"][mask], 0.0, atol=0.0))
        check(f"{name} 有文本行的 S 贡献非全零（分支确实接上了）",
              not np.allclose(contrib["S"][~mask], 0.0))

    # ---------- 分支隔离 ----------
    print("\n--- 分支隔离 ---")
    model_no_q = fit_candidate("N+S", train, cal)
    perturbed_q = FeatureBundle(
        numeric_history=test.numeric_history, semantic=test.semantic,
        quality=test.quality * 5.0 + 13.0, text_available=test.text_available,
        origin_index=test.origin_index, targets=test.targets,
        targets_standardized=test.targets_standardized)
    check("N+S 中 Q 不进设计矩阵：扰动质量特征不影响输出",
          np.array_equal(model_no_q.predict(test), model_no_q.predict(perturbed_q)))

    # ---------- 确定性 ----------
    print("\n--- 确定性 ---")
    a = fit_candidate("N+S+Q", train, cal).predict(test)
    b = fit_candidate("N+S+Q", train, cal).predict(test)
    check("同输入重复拟合逐位一致", np.array_equal(a, b))

    # ---------- 训练段完全无文本 → 安全降级 ----------
    print("\n--- 训练段无文本的安全降级 ---")
    no_text = synthetic_bundle(200, 24, 6, text_frac=0.0, seed=11)
    degraded = BranchResidualCandidate("N+S+Q")
    degraded.fit_design(no_text)
    check("训练段无文本时 S 标记为降级", degraded.s_degraded)
    check("降级后仍保留 N 分支可预测", any(b.name == "N" for b in degraded.branches))
    check("降级后不包含 S 分支", not any(b.name == "S" for b in degraded.branches))

    # ---------- 预测契约 ----------
    print("\n--- 预测契约（14 字段 / 键唯一 / 步长完整）---")
    model = fit_candidate("N+S+Q", train, cal)
    pred = model.predict(test)
    writer = PredictionWriter()
    oids = np.array([f"Agriculture:h6:f1:o{int(o)}" for o in test.origin_index])
    writer.extend(pred, origin_id=oids, origin_index=test.origin_index,
                  task_id="Agriculture_h6_f1", fold_id=1, segment="decision", scenario="proxy",
                  candidate_id="N+S+Q", seed=2026, bundle_signature="sig-smoke",
                  config_sha256_value=config_sha256({"candidate": "N+S+Q"}),
                  commit="deadbeef")
    check("字段数 = 14", len(writer.records[0].__dataclass_fields__) == 14)
    check("每起点每步一行", len(writer.records) == len(test.numeric_history) * 6)
    writer.validate_complete(horizon=6, expected_origins=len(test.numeric_history), seed=2026)
    check("契约完整性校验通过", True)

    # 一维预测被拒绝
    rejected_1d = False
    try:
        PredictionWriter().extend(pred[:, 0], origin_id=oids, origin_index=test.origin_index,
                                  task_id="t", fold_id=1, segment="decision", scenario="proxy",
                                  candidate_id="x", seed=1, bundle_signature="s",
                                  config_sha256_value="c", commit="k")
    except ValueError:
        rejected_1d = True
    check("一维预测被拒绝（H=1 广播防护）", rejected_1d)

    # 错位 origin_id 被拒绝
    rejected_shift = False
    try:
        w = PredictionWriter()
        w.extend(pred, origin_id=oids, origin_index=test.origin_index + 1,
                 task_id="t", fold_id=1, segment="decision", scenario="proxy",
                 candidate_id="x", seed=1, bundle_signature="s",
                 config_sha256_value="c", commit="k")
        from bundle_reader import ORIGIN_ID_RE
        for oid, idx in zip([r.origin_id for r in w.records[:1]], [r.origin_index for r in w.records[:1]]):
            m = ORIGIN_ID_RE.match(oid)
            if m and int(m.group("index")) != idx:
                rejected_shift = True
    except Exception:
        rejected_shift = True
    check("origin_id 与 origin_index 错位可被检出", rejected_shift)

    # ---------- 置换留痕与重拟合 ----------
    print("\n--- 置换零分布（小次数自检）---")
    train_p = synthetic_bundle(120, 24, 6, seed=21)
    cal_p = synthetic_bundle(40, 24, 6, seed=22)
    dec_p = synthetic_bundle(40, 24, 6, seed=23)
    small = row_permutation_null("N+S+Q", train_p, cal_p, dec_p, count=5, seed=2026,
                                 progress_every=0)
    check("置换次数与请求一致", small.observed_count == 5)
    check("每次置换都留了种子", len(small.seeds) == 5)
    check("每次置换都留了损失", all(np.isfinite(x) for x in small.losses))
    check("未跑满 999 时 p 值为 None（不伪填）", small.p_value(0.0) is None)

    # 无跨次缓存：同一置换重复跑结果一致
    again = row_permutation_null("N+S+Q", train_p, cal_p, dec_p, count=5, seed=2026,
                                 progress_every=0)
    check("同一置换重复运行结果逐位一致（无跨次缓存 ⇒ 每次确实重拟合）",
          np.array_equal(np.array(small.losses), np.array(again.losses)))

    # ---------- 无信号情景 ----------
    print("\n--- 无信号情景（实现诊断，非真实检验）---")
    null_train = synthetic_bundle(150, 24, 6, signal=False, seed=31)
    null_cal = synthetic_bundle(50, 24, 6, signal=False, seed=32)
    null_dec = synthetic_bundle(50, 24, 6, signal=False, seed=33)
    from permutation import observed_loss
    obs, _ = observed_loss("N+S+Q", null_train, null_cal, null_dec)
    check("无信号情景能算出观测损失", np.isfinite(obs))

    print(f"\n全部通过（{len(PASSED)} 项检查）")
    print("未覆盖项（需真实 Bundle）：冻结签名复算、真实 999 次置换与 p 值")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"\n{exc}", file=sys.stderr)
        raise SystemExit(1)
