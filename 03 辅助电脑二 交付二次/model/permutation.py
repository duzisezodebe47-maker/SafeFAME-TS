"""决策段 999 次逐行错位置换（第二轮 P1 第 5 条）。

## 为什么"每次置换都重拟合"是硬要求

`docs/23` 的直接教训：v4 的频率候选在逐行置换下 p 值为 0.021，但**沿用旧交互尺度**
时该值虚低 —— 一旦每次置换都重拟交互尺度，p 值塌到 **0.085**，稳健性消失。

所以本模块**每次置换都重建全部数据依赖的量**：PCA、各分支标度器、交互尺度、
以及 α 选择。任何"缓存一次、复用 999 次"的写法都会制造假显著。

## 置换单位

按主控口径：**逐行文本与质量联合错位**。数值历史保持不动，把语义与质量的行相对
数值打乱，从而破坏"文本 ↔ 数值"的配对。train / calibration / decision 三段各自
独立置换（决策段的置换才是被计分的那次）。

## 成本

单次置换 ≈ 一次 PCA(768→24) + 若干 StandardScaler + 一次 α 坐标下降。
实测约 0.1 秒量级，999 次 × 2 个门控候选在分钟量级 —— 可接受。
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from branches import FeatureBundle  # noqa: E402
from candidates import BranchResidualCandidate  # noqa: E402

ROW_PERMUTATIONS = 999


@dataclass
class NullResult:
    """一次置换零分布的结果，含逐次留痕。"""

    candidate: str
    segment: str
    losses: list[float]
    seeds: list[int]
    failures: list[dict]
    wall_seconds: float

    @property
    def observed_count(self) -> int:
        return len(self.losses)

    def p_value(self, observed_loss: float) -> float | None:
        """单侧经验 p。**置换未跑满时返回 None，绝不外推。**"""
        if self.observed_count < ROW_PERMUTATIONS:
            return None
        arr = np.asarray(self.losses, dtype=float)
        return float((1 + np.sum(arr <= observed_loss)) / (1 + len(arr)))


def permute_rows(bundle: FeatureBundle, rng: np.random.Generator) -> FeatureBundle:
    """把语义与质量的行相对数值打乱（数值不动）。

    用一个**共享的**置换索引同时作用于 semantic / quality / text_available，
    保证文本与其可得性掩码仍然对应，只破坏与数值的配对。
    """
    order = rng.permutation(len(bundle.numeric_history))
    return FeatureBundle(
        numeric_history=bundle.numeric_history,
        semantic=bundle.semantic[order],
        quality=bundle.quality[order],
        text_available=bundle.text_available[order],
        origin_index=bundle.origin_index,
        targets=bundle.targets,
        targets_standardized=bundle.targets_standardized,
    )


def decision_loss(model: BranchResidualCandidate, decision: FeatureBundle) -> float:
    """决策段 MSE（标准化坐标）。只用于零分布，不用于任何选择。"""
    pred = model.predict(decision)
    truth = np.asarray(decision.targets_standardized, dtype=np.float64)
    return float(np.mean((pred - truth) ** 2))


def observed_loss(
    name: str, train: FeatureBundle, cal: FeatureBundle, decision: FeatureBundle
) -> tuple[float, BranchResidualCandidate]:
    """真实对齐下的决策段损失。α 只在校准段选。"""
    model = BranchResidualCandidate(name)
    model.fit_design(train)
    model.select_alphas(train, cal)
    model.refit(FeatureBundle(
        numeric_history=np.r_[train.numeric_history, cal.numeric_history],
        semantic=np.r_[train.semantic, cal.semantic],
        quality=np.r_[train.quality, cal.quality],
        text_available=np.r_[train.text_available, cal.text_available],
        origin_index=np.r_[train.origin_index, cal.origin_index],
        targets_standardized=np.r_[train.targets_standardized, cal.targets_standardized],
    ))
    return decision_loss(model, decision), model


def row_permutation_null(
    name: str,
    train: FeatureBundle,
    cal: FeatureBundle,
    decision: FeatureBundle,
    count: int = ROW_PERMUTATIONS,
    seed: int = 2026,
    progress_every: int = 100,
) -> NullResult:
    """跑 `count` 次逐行错位置换。**每次置换重建 PCA / 尺度 / 交互 / α。**

    任何一次置换内部抛错都被记录到 `failures` 而不是吞掉 ——
    任务书要求"保存每次置换的…失败处理"，未跑满不得伪填 p 值。
    """
    losses: list[float] = []
    seeds: list[int] = []
    failures: list[dict] = []
    started = time.perf_counter()

    for k in range(count):
        local_seed = seed * 1000 + k
        rng = np.random.default_rng(local_seed)
        seeds.append(local_seed)
        try:
            p_train = permute_rows(train, rng)
            p_cal = permute_rows(cal, rng)
            p_dec = permute_rows(decision, rng)

            model = BranchResidualCandidate(name)
            model.fit_design(p_train)              # ← 重拟合 PCA / 尺度
            model.select_alphas(p_train, p_cal)    # ← 重选 α
            model.refit(FeatureBundle(
                numeric_history=np.r_[p_train.numeric_history, p_cal.numeric_history],
                semantic=np.r_[p_train.semantic, p_cal.semantic],
                quality=np.r_[p_train.quality, p_cal.quality],
                text_available=np.r_[p_train.text_available, p_cal.text_available],
                origin_index=np.r_[p_train.origin_index, p_cal.origin_index],
                targets_standardized=np.r_[p_train.targets_standardized, p_cal.targets_standardized],
            ))
            losses.append(decision_loss(model, p_dec))
        except Exception as exc:  # noqa: BLE001 — 失败必须留痕
            failures.append({"iteration": k, "seed": local_seed,
                             "error": f"{type(exc).__name__}: {exc}"})

        if progress_every and (k + 1) % progress_every == 0:
            print(f"    {name}: {k + 1}/{count}  (失败 {len(failures)})", file=sys.stderr)

    return NullResult(candidate=name, segment="decision", losses=losses, seeds=seeds,
                      failures=failures, wall_seconds=round(time.perf_counter() - started, 2))


def circular_shift_null(
    name: str, train: FeatureBundle, cal: FeatureBundle, decision: FeatureBundle,
    count: int = ROW_PERMUTATIONS, seed: int = 2026, block: int = 7,
) -> NullResult:
    """循环移位敏感性诊断。**只作诊断，不回写主门控。**"""
    losses: list[float] = []
    seeds: list[int] = []
    failures: list[dict] = []
    started = time.perf_counter()

    for k in range(count):
        local_seed = seed * 1000 + k
        rng = np.random.default_rng(local_seed)
        seeds.append(local_seed)
        try:
            def shift(bundle: FeatureBundle) -> FeatureBundle:
                n = len(bundle.numeric_history)
                offset = int(rng.integers(1, max(2, n)))
                order = np.roll(np.arange(n), offset)
                return FeatureBundle(
                    numeric_history=bundle.numeric_history,
                    semantic=bundle.semantic[order],
                    quality=bundle.quality[order],
                    text_available=bundle.text_available[order],
                    origin_index=bundle.origin_index,
                    targets=bundle.targets,
                    targets_standardized=bundle.targets_standardized,
                )

            p_train, p_cal, p_dec = shift(train), shift(cal), shift(decision)
            model = BranchResidualCandidate(name)
            model.fit_design(p_train)
            model.select_alphas(p_train, p_cal)
            model.refit(FeatureBundle(
                numeric_history=np.r_[p_train.numeric_history, p_cal.numeric_history],
                semantic=np.r_[p_train.semantic, p_cal.semantic],
                quality=np.r_[p_train.quality, p_cal.quality],
                text_available=np.r_[p_train.text_available, p_cal.text_available],
                origin_index=np.r_[p_train.origin_index, p_cal.origin_index],
                targets_standardized=np.r_[p_train.targets_standardized, p_cal.targets_standardized],
            ))
            losses.append(decision_loss(model, p_dec))
        except Exception as exc:  # noqa: BLE001
            failures.append({"iteration": k, "seed": local_seed,
                             "error": f"{type(exc).__name__}: {exc}"})

    return NullResult(candidate=name, segment="decision-circular", losses=losses,
                      seeds=seeds, failures=failures,
                      wall_seconds=round(time.perf_counter() - started, 2))


def write_null(path: Path, result: NullResult, observed: float) -> None:
    """写出零分布。`predictions.csv` 与 `null_scores.csv` 严格分离。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "candidate,segment,iteration,seed,loss\n"
    body = "".join(
        f"{result.candidate},{result.segment},{i},{result.seeds[i]},{loss}\n"
        for i, loss in enumerate(result.losses)
    )
    path.write_text(header + body, encoding="utf-8", newline="\n")

    summary = {
        "candidate": result.candidate,
        "segment": result.segment,
        "requested": ROW_PERMUTATIONS,
        "completed": result.observed_count,
        "failures": len(result.failures),
        "observed_loss": observed,
        # 未跑满时 p 为 null —— 任务书要求"不伪填 p 值"
        "p_value": result.p_value(observed),
        "p_value_note": ("置换未跑满，p 值不可用" if result.observed_count < ROW_PERMUTATIONS
                         else "经验单侧 p"),
        "wall_seconds": result.wall_seconds,
        "failure_detail": result.failures[:10],
    }
    (path.parent / f"{path.stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
