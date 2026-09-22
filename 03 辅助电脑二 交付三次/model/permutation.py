"""决策段逐行联合错位置换（第三轮 A.4 / A.5 修复版）。

## 为什么"每次置换都重拟合"是硬要求

`docs/23` 的教训：v4 频率候选在逐行置换下 p=0.021，但沿用旧交互尺度时该值虚低 ——
每次置换都重拟尺度后 p 塌到 0.085。故本模块**每次置换重建全部数据依赖量**：
PCA、各分支标度器、交互尺度、α 选择。

## 第三轮修复：逐次记录，不再错配

第二轮 `seeds` 包含**失败**迭代，而 `losses` 只含**成功**迭代；写 CSV 时用
`seeds[i]` 配 `losses[i]` 会**错配**（第 2 次失败、第 3 次成功时，第 2 个损失会被
安到第 3 个种子上）。主控在第三轮任务书 A.4 点出了这一点。

现在改为 `NullIteration` 三元组逐条记录 `(iteration, seed, loss | error)`，
CSV 一行一次迭代，成功与失败各自留痕。

**p 值门槛**：契约要求 `requested == 999` **且** `successful == 999` 才计算；
任何失败先补跑，否则报"不可用"，绝不外推。
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from branches import FeatureBundle  # noqa: E402
from candidates import BranchResidualCandidate  # noqa: E402

ROW_PERMUTATIONS = 999


@dataclass
class NullIteration:
    """一次置换的完整留痕 —— 成功记损失，失败记原因，**种子始终对应得上**。"""

    iteration: int
    seed: int
    loss: float | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.loss is not None


@dataclass
class NullResult:
    candidate: str
    segment: str
    iterations: list[NullIteration] = field(default_factory=list)
    requested: int = ROW_PERMUTATIONS
    wall_seconds: float = 0.0

    @property
    def successful(self) -> list[NullIteration]:
        return [it for it in self.iterations if it.ok]

    @property
    def failures(self) -> list[NullIteration]:
        return [it for it in self.iterations if not it.ok]

    @property
    def losses(self) -> list[float]:
        return [float(it.loss) for it in self.successful]

    def p_value(self, observed_loss: float) -> float | None:
        """单侧经验 p。

        **要求 requested 达到契约下限 999，且全部 999 次都成功。**
        任何失败（或请求次数不足）都返回 None —— 由调用方决定补跑还是报告不可用。
        """
        if self.requested < ROW_PERMUTATIONS:
            return None
        if len(self.successful) != self.requested:
            return None
        arr = np.asarray(self.losses, dtype=float)
        return float((1 + np.sum(arr <= observed_loss)) / (1 + len(arr)))

    def unavailable_reason(self) -> str | None:
        """p 值不可用时的可定位原因；可用时返回 None。"""
        if self.requested < ROW_PERMUTATIONS:
            return f"请求次数 {self.requested} 低于契约下限 {ROW_PERMUTATIONS}"
        if self.failures:
            return (f"{len(self.failures)} 次置换失败"
                    f"（首次: iteration={self.failures[0].iteration} "
                    f"{self.failures[0].error}）；须先补跑，不得据此计算 p")
        return None


def permute_rows(bundle: FeatureBundle, rng: np.random.Generator) -> FeatureBundle:
    """把语义与质量的行相对数值打乱（数值不动）。

    同一个置换索引同时作用于 semantic / quality / text_available，
    保证文本与其可得性掩码仍对应，只破坏与数值的配对。
    """
    order = rng.permutation(len(bundle.numeric_history))
    return FeatureBundle(
        numeric_history=bundle.numeric_history,
        semantic=bundle.semantic[order], quality=bundle.quality[order],
        text_available=bundle.text_available[order],
        origin_index=bundle.origin_index, targets=bundle.targets,
        targets_standardized=bundle.targets_standardized,
    )


def shift_rows(bundle: FeatureBundle, offset: int) -> FeatureBundle:
    """循环移位：整段平移 offset，保持同一块内的时序结构。"""
    n = len(bundle.numeric_history)
    order = np.roll(np.arange(n), offset)
    return FeatureBundle(
        numeric_history=bundle.numeric_history,
        semantic=bundle.semantic[order], quality=bundle.quality[order],
        text_available=bundle.text_available[order],
        origin_index=bundle.origin_index, targets=bundle.targets,
        targets_standardized=bundle.targets_standardized,
    )


def decision_loss(model: BranchResidualCandidate, decision: FeatureBundle) -> float:
    """决策段 MSE（标准化坐标）。只用于零分布，不用于任何选择。"""
    pred = model.predict(decision)
    truth = np.asarray(decision.targets_standardized, dtype=np.float64)
    return float(np.mean((pred - truth) ** 2))


def _fit(name: str, train: FeatureBundle, cal: FeatureBundle) -> BranchResidualCandidate:
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


def observed_loss(
    name: str, train: FeatureBundle, cal: FeatureBundle, decision: FeatureBundle
) -> tuple[float, BranchResidualCandidate]:
    """真实对齐下的决策段损失。α 只在校准段选。"""
    model = _fit(name, train, cal)
    return decision_loss(model, decision), model


def _run(
    name: str, segment: str, train: FeatureBundle, cal: FeatureBundle,
    decision: FeatureBundle, count: int, seed: int, transform,
    progress_every: int,
) -> NullResult:
    """公共置换循环。`transform(bundle, rng, k)` 返回被置换后的切片。"""
    iterations: list[NullIteration] = []
    started = time.perf_counter()
    for k in range(count):
        local_seed = seed * 1000 + k
        rng = np.random.default_rng(local_seed)
        loss: float | None = None
        error: str | None = None
        try:
            p_train = transform(train, rng, k)
            p_cal = transform(cal, rng, k)
            p_dec = transform(decision, rng, k)
            model = _fit(name, p_train, p_cal)          # ← 重拟合 PCA/尺度/交互/α
            loss = decision_loss(model, p_dec)
        except Exception as exc:  # noqa: BLE001 — 失败留痕，不吞
            error = f"{type(exc).__name__}: {exc}"
        # 无论成败都记一条，种子与结果一一对应
        iterations.append(NullIteration(iteration=k, seed=local_seed, loss=loss, error=error))
        if progress_every and (k + 1) % progress_every == 0:
            print(f"    {name}: {k + 1}/{count} 失败 {len([i for i in iterations if not i.ok])}",
                  file=sys.stderr)
    return NullResult(candidate=name, segment=segment, iterations=iterations,
                      requested=count, wall_seconds=round(time.perf_counter() - started, 2))


def row_permutation_null(
    name: str, train: FeatureBundle, cal: FeatureBundle, decision: FeatureBundle,
    count: int = ROW_PERMUTATIONS, seed: int = 2026, progress_every: int = 100,
) -> NullResult:
    """逐行联合错位置换。每次重建 PCA / 尺度 / 交互 / α。"""
    return _run(name, "decision", train, cal, decision, count, seed,
                lambda b, rng, k: permute_rows(b, rng), progress_every)


def circular_shift_null(
    name: str, train: FeatureBundle, cal: FeatureBundle, decision: FeatureBundle,
    count: int = ROW_PERMUTATIONS, seed: int = 2026, circular_block: int = 7,
    progress_every: int = 100,
) -> NullResult:
    """循环移位诊断。**只作诊断，不回写主门控。**

    `circular_block` 是**真实的移位块长度**：每次移位取
    `offset = circular_block * (1 + rng.integers(0, n // circular_block))`，
    再按 `shift_rows` 整段循环平移。任务书 A.5 要求"参数叫 circular-block 就真正使用块长度"。
    """
    if circular_block < 1:
        raise ValueError("circular_block 必须为正整数")

    def transform(bundle: FeatureBundle, rng: np.random.Generator, _k: int) -> FeatureBundle:
        n = len(bundle.numeric_history)
        blocks = max(1, n // circular_block)
        offset = int(circular_block * (1 + rng.integers(0, blocks)))
        return shift_rows(bundle, offset % max(1, n))

    return _run(name, "decision-circular", train, cal, decision, count, seed,
                transform, progress_every)


def write_null(path: Path, result: NullResult, observed: float) -> None:
    """写出零分布。**一行一次迭代**，成功与失败各自留痕，种子绝不错配。

    `predictions.csv` 与 `null_scores.csv` 严格分离。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["iteration,seed,status,loss,error"]
    for it in result.iterations:
        status = "ok" if it.ok else "failed"
        loss = "" if it.loss is None else repr(float(it.loss))
        error = "" if not it.error else it.error.replace(",", ";")
        lines.append(f"{it.iteration},{it.seed},{status},{loss},{error}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    reason = result.unavailable_reason()
    summary = {
        "candidate": result.candidate,
        "segment": result.segment,
        "requested": result.requested,
        "successful": len(result.successful),
        "failed": len(result.failures),
        "contract_minimum": ROW_PERMUTATIONS,
        "observed_loss": observed,
        "p_value": result.p_value(observed),
        "p_value_note": reason or "经验单侧 p（requested 与 successful 均达 999）",
        "wall_seconds": result.wall_seconds,
        "failure_detail": [{"iteration": it.iteration, "seed": it.seed, "error": it.error}
                           for it in result.failures[:10]],
    }
    (path.parent / f"{path.stem}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
