"""零文本覆盖时的**安全退化契约**（第十轮任务书 B）。

任务：`SocialGood_h3_f1`（input_len 24、horizon 3、周期 12、边界 `[366,458,641,732]`）。
主控第十轮任务的第 5 条写得很直白：**覆盖为零或过低时执行安全退化，
不以多模态名义参赛**。本模块把这条写成**可执行契约**，并逐条配套测试。

## 契约

**R1 退化义务**：训练区文本覆盖率为 0（或低于可训练下限）时，任何含语义分支
（`S` / `SF`）的候选**必须**退化为**注册表内**的数值或质量感知候选。
判定阈值与 `branches.SemanticBranch.fit` 的「训练段文本行 < 2 即不可训练」一致 ——
契约与实现用同一个下限，不允许两套标准。

**R2 证据自洽**：退化记录必须同时给出 `requested_candidate` / `effective_candidate` /
`degraded` / `features_actually_used` / `reason` / `modality`，且
· `effective_candidate` 不含 `S`/`SF`；
· `features_actually_used` 与 `effective_candidate` 的分支组成一致；
· `modality` **只能是** `numeric` 或 `quality_aware`，**不得**是 `multimodal`。
`assert_evidence_consistent()` 对任何交付记录做这项校验。

**R3 只用训练区统计**：`train_text_coverage` / `quality_is_usable` 的入参是**训练区**数组，
`resolve_candidate()` 的签名里**没有**测试段参数 —— 退化判定在结构上无法读取
测试区文本或测试真值（不是靠"我们不读"的约定）。

**R4 非法输入必须拒绝**：边界非严格递增、与 `input_len`/`horizon` 冲突、
注册表为空、`requested` 不在注册表里 —— 一律抛 `ContractViolation`，不猜、不兜底。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

TEXT_BRANCHES = ("S", "SF")
QUALITY_BRANCHES = ("Q",)
# 与 branches.SemanticBranch.fit 一致：训练段可用文本行 < 2 即不可训练
MIN_TRAIN_TEXT_ROWS = 2
ALLOWED_MODALITY = ("numeric", "quality_aware")


class ContractViolation(RuntimeError):
    """违反退化契约。**不得降级处理、不得猜测目标候选。**"""


@dataclass(frozen=True)
class Degradation:
    """一次候选解析的完整留痕（R2）。"""

    requested: str
    effective: str
    degraded: bool
    reason: str | None
    features_actually_used: tuple[str, ...]
    modality: str
    train_text_rows: int
    train_rows: int

    def as_evidence(self) -> dict:
        """转成 run manifest 里 `degradation` 字段的形状。"""
        return asdict(self)


def _branches(candidate: str) -> tuple[str, ...]:
    return tuple(part for part in str(candidate).split("+") if part)


def train_text_coverage(text_available, *, segment_mask=None) -> tuple[int, int]:
    """训练区可用文本行数与总行数（**只接受训练区数组**，R3）。

    返回 `(可用文本行数, 总行数)`；全零时第一个为 0。
    """
    mask = np.asarray(text_available).astype(bool)
    if segment_mask is not None:                     # 兼容显式传入的行掩码
        mask = mask[np.asarray(segment_mask).astype(bool)]
    return int(mask.sum()), int(mask.size)


def quality_is_usable(quality) -> bool:
    """质量特征是否可用：存在、非全缺失（非全 NaN），且**不是常量**。

    常量质量特征不携带信息，等同于不可用 —— 让它进模型只会得到 0 系数，
    却会让证据把它记成"质量感知"，那是**名不副实**。
    """
    if quality is None:
        return False
    arr = np.asarray(quality, dtype=float)
    if arr.size == 0 or arr.ndim != 2:
        return False
    if np.isnan(arr).all():
        return False
    # 至少一列有变化（忽略 NaN 后的极差 > 0）
    for col in range(arr.shape[1]):
        values = arr[:, col]
        finite = values[np.isfinite(values)]
        if finite.size >= 2 and float(np.nanmax(finite) - np.nanmin(finite)) > 0.0:
            return True
    return False


def validate_bounds(bounds, *, input_len: int, horizon: int) -> None:
    """R4：边界必须严格递增、非负、且每段放得下 H 窗口。"""
    if bounds is None or len(bounds) != 4:
        raise ContractViolation(f"边界必须是 4 个数 [train_end, cal_end, dec_end, test_end]，收到 {bounds!r}")
    values = [int(x) for x in bounds]
    if values[0] <= 0 or any(b <= a for a, b in zip(values, values[1:])):
        raise ContractViolation(f"边界必须严格递增且首段为正: {values}")
    if int(input_len) < 1 or int(horizon) < 1:
        raise ContractViolation(f"input_len / horizon 必须为正: {input_len} / {horizon}")
    for name, (lo, hi) in zip(("train", "calibration", "decision", "test"),
                              zip([0, *values[:-1]], values)):
        # 段内至少要放得下一个完整的 H 窗口（训练段还要求 input_len 的历史）
        if name == "train":
            if hi - lo < int(input_len) + int(horizon):
                raise ContractViolation(f"train 段容不下 input_len+horizon: {hi - lo}")
        elif hi - lo < int(horizon):
            raise ContractViolation(f"{name} 段容不下一个 H 窗口: {hi - lo} < {horizon}")


def expected_origins(bounds, *, input_len: int, horizon: int) -> dict:
    """按 spec 语义复算各段起点数。

    命名与主控一致（他们第十轮要核对 SocialGood 的 **700 / 611 / 89**）：
      `selection_region` = train + calibration + decision（可用于选择的全区间，611）
      `gating_origins`   = calibration + decision（真正进门控的起点，271）
      `test`             = 89
      `all`              = 700
    每个起点的可用条件是「历史窗口完整（origin ≥ input_len）」且「目标窗口不越段
    （origin + horizon ≤ 段上界）」。
    """
    values = [int(x) for x in bounds]
    edges = list(zip([0, *values[:-1]], values))
    out = {}
    for name, (lo, hi) in zip(("train", "calibration", "decision", "test"), edges):
        first = max(lo, int(input_len))
        last = hi - int(horizon)
        out[name] = max(0, last - first + 1)
    out["selection_region"] = out["train"] + out["calibration"] + out["decision"]
    out["gating_origins"] = out["calibration"] + out["decision"]
    out["all"] = out["train"] + out["calibration"] + out["decision"] + out["test"]
    return out


def resolve_candidate(requested: str, *, registry, train_text_rows: int,
                      train_rows: int, quality_usable: bool) -> Degradation:
    """按契约解析出**实际参赛**的候选（R1/R2/R3）。

    `registry` 是注册表（来自冻结 spec 的 `numeric_fallback_candidates` 与主控签发的
    候选注册表）。**退化目标必须落在注册表内**，否则拒绝。
    """
    registry = tuple(str(x) for x in registry)
    if not registry:
        raise ContractViolation("候选注册表为空 —— 拒绝猜测")
    if not str(requested):
        raise ContractViolation("requested 候选为空")
    if str(requested) not in registry:
        raise ContractViolation(f"requested 候选 {requested!r} 不在注册表 {sorted(registry)} 中")

    parts = _branches(requested)
    text_free = not (set(parts) & set(TEXT_BRANCHES))

    targets: list[str] = []
    if text_free:
        # 不含语义分支：无需因文本退化；只处理"质量不可用却带了 Q"的情形
        if set(parts) & set(QUALITY_BRANCHES) and not quality_usable:
            targets = [c for c in ("N",) if c in registry]
            reason = "训练区质量特征不可用（全缺失或常量），Q 分支无信息"
        else:
            return Degradation(str(requested), str(requested), False, None, parts,
                               "quality_aware" if set(parts) & set(QUALITY_BRANCHES) else "numeric",
                               int(train_text_rows), int(train_rows))
    else:
        degradable = (int(train_text_rows) < MIN_TRAIN_TEXT_ROWS
                      or not quality_usable and set(parts) & set(QUALITY_BRANCHES))
        if not degradable:
            return Degradation(str(requested), str(requested), False, None, parts,
                               "multimodal", int(train_text_rows), int(train_rows))
        if int(train_text_rows) < MIN_TRAIN_TEXT_ROWS:
            reason = (f"训练区可用文本行 {int(train_text_rows)} < {MIN_TRAIN_TEXT_ROWS}，"
                      f"语义分支不可训练（与 branches.SemanticBranch 同一阈值）")
        else:
            reason = "训练区质量特征不可用，保留语义分支仍可参赛"
        # 退化目标优先级：质量感知（若注册且质量可用）→ 数值
        for cand in ("N+Q", "N"):
            if cand not in registry:
                continue
            if set(_branches(cand)) & set(QUALITY_BRANCHES) and not quality_usable:
                continue
            if set(_branches(cand)) & set(TEXT_BRANCHES):
                continue
            targets = [cand]
            break

    if not targets:
        raise ContractViolation(
            f"{requested!r} 需要退化，但注册表 {sorted(registry)} 里没有可用的数值/质量感知候选")

    effective = targets[0]
    return Degradation(str(requested), effective, True, reason, _branches(effective),
                       "quality_aware" if set(_branches(effective)) & set(QUALITY_BRANCHES)
                       else "numeric",
                       int(train_text_rows), int(train_rows))


def assert_evidence_consistent(record: dict) -> None:
    """R2：校验交付记录里的退化字段自洽。任何一条不满足即抛异常。"""
    for field in ("requested_candidate", "effective_candidate", "degraded",
                  "features_actually_used", "modality"):
        if field not in record:
            raise ContractViolation(f"退化记录缺字段 {field}")
    requested = str(record["requested_candidate"])
    effective = str(record["effective_candidate"])
    features = tuple(str(x) for x in record["features_actually_used"])
    modality = str(record["modality"])
    degraded = bool(record["degraded"])

    if set(features) != set(_branches(effective)):
        raise ContractViolation(
            f"features_actually_used {features} 与 effective_candidate {effective} 不一致")
    if set(features) & set(TEXT_BRANCHES):
        raise ContractViolation(f"退化后仍启用语义分支: {features}")
    if modality not in ALLOWED_MODALITY:
        raise ContractViolation(
            f"退化后的 modality={modality!r} 非法（只能是 {ALLOWED_MODALITY}）——"
            f"不得以多模态名义参赛")
    if degraded:
        if requested == effective:
            raise ContractViolation("degraded=True 但 requested 与 effective 相同")
        if not str(record.get("reason") or "").strip():
            raise ContractViolation("degraded=True 必须给出 reason")
    else:
        if requested != effective:
            raise ContractViolation("degraded=False 但 requested 与 effective 不同")
        if modality == "multimodal" and not (set(features) & set(TEXT_BRANCHES)):
            raise ContractViolation("标为 multimodal 却没有语义分支")
