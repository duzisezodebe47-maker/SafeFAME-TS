"""第二轮候选定义。

## 候选分类（严格按主控 `split_spec_v2.json`）

**门控候选**（`gate_candidates`，最多两条，可进入正式门控）::

    N+S+Q       当前逐分支惩罚版本
    N+S+Q+SF    语义 × 数值谱显式交互版本（本轮新增）

**诊断消融**（保留同一输入、同一划分、同一种子，用于反证，**不进门槛**）::

    N           纯数值
    N+Q         数值 + 质量
    N+S         数值 + 语义（Q 不进设计矩阵，只用掩码）
    N+F         数值 + 纯数值谱
    N+S+Q+F     历史命名；F 是纯数值谱统计，**不得称为交互版本**

## 与第一轮的差异

1. 设计矩阵改为消费主控冻结 Bundle 的字段（`numeric_history` / `semantic` /
   `quality` / `text_available`）。
2. `N+S` 中 Q **不进入**回归设计，仅用于掩码传播（主控第二轮口径第 3 条）。
3. 训练段无文本时 `S` 不可训练：该候选降级为纯 `N` 并**如实记录**，
   不构造伪信号（主控第二轮口径第 4 条）。

## α 网格

第三轮任务书正文曾写「沿用仓库 `RIDGE_ALPHAS`」（9 档），与 `split_spec_v2.json`
的 `alpha_grid`（7 档）不一致。**主控第四轮已确认以 7 档为准**，此处不再列为待裁定项。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from branches import (  # noqa: E402
    Branch, FeatureBundle, FrequencyBranch, NumericBranch, QualityBranch,
    SemanticBranch, SpectralInteractionBranch,
)

# 协议冻结的 7 档网格（split_spec_v2.json: alpha_grid）—— 主控第四轮已确认，无需再裁定
PROTOCOL_ALPHAS: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)

# 门控候选 —— 只有这两条可进正式门控
GATE_CANDIDATES: tuple[str, ...] = ("N+S+Q", "N+S+Q+SF")

# 诊断消融 —— 用于反证，不进门槛
DIAGNOSTIC_CANDIDATES: tuple[str, ...] = ("N", "N+Q", "N+S", "N+F", "N+S+Q+F")

ALL_CANDIDATES: tuple[str, ...] = GATE_CANDIDATES + DIAGNOSTIC_CANDIDATES

# 分支组成。S 由 SemanticBranch 提供，SF 由 SpectralInteractionBranch 提供。
_COMPOSITION: dict[str, tuple[str, ...]] = {
    "N": ("N",),
    "N+Q": ("N", "Q"),
    "N+S": ("N", "S"),          # Q 不入设计矩阵
    "N+F": ("N", "F"),
    "N+S+Q": ("N", "S", "Q"),
    "N+S+Q+F": ("N", "S", "Q", "F"),
    "N+S+Q+SF": ("N", "S", "Q", "SF"),
}


def _group_ridge(
    design: np.ndarray, target: np.ndarray, groups: list[str], alpha_by_group: dict[str, float]
) -> np.ndarray:
    """带逐组惩罚的岭回归。正规方程在 float64 下求解。

    第一轮的教训：同一设计矩阵在 float32 下 `alpha=1e-4` 时 cholesky 失败
    （`potrf` 报非正定），float64 下成功。设计矩阵可保持 float32，Gram 必须升精度。
    """
    penalty = np.array([alpha_by_group[g] for g in groups], dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    gram = design.T @ design
    gram.flat[:: len(gram) + 1] += penalty
    rhs = design.T @ target
    try:
        return cho_solve(cho_factor(gram, lower=True, check_finite=False), rhs, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise AssertionError(
            f"Gram 非正定，cholesky 失败（分支: {sorted(set(groups))}，设计矩阵 {design.shape}）。"
            f"通常意味着零方差或高度共线的列 —— 检查 SafeScaler 是否丢弃了退化列。"
        ) from exc


class BranchResidualCandidate:
    """``pred = Last + Σ_b X_b w_b``，逐分支独立 α。

    对线性模型而言拼接等价于相加，所以"分支隔离"只能体现在**正则化几何**上 ——
    逐分支 α 允许各分支不同的收缩强度，这是单 α 做不到的可检验差异。
    """

    def __init__(self, name: str, uniform_alpha: bool = False) -> None:
        if name not in _COMPOSITION:
            raise ValueError(f"未知候选: {name}")
        self.name = name
        self.uniform_alpha = uniform_alpha
        self.branch_names = _COMPOSITION[name]
        self.branches: list[Branch] = []
        self.s_degraded = False        # 训练段无文本时 S 降级
        self.alpha_by_group: dict[str, float] = {}
        self.weights: np.ndarray | None = None
        self._groups: list[str] = []

    # ---------- 分支装配 ----------

    def _build_branches(self, train: FeatureBundle) -> list[Branch]:
        """按组成装配并 **fit 全部**分支。训练段无文本时安全降级 S/SF。"""
        # 语义分支先建先 fit —— SF 依赖它，且要先知道是否可训练
        semantic: SemanticBranch | None = None
        if "S" in self.branch_names or "SF" in self.branch_names:
            semantic = SemanticBranch()
            semantic.fit(train)

        branches: list[Branch] = []
        for name in self.branch_names:
            if name == "N":
                branch = NumericBranch()
            elif name == "Q":
                branch = QualityBranch()
            elif name == "F":
                branch = FrequencyBranch()
            elif name == "S":
                if semantic.untrainable:
                    self.s_degraded = True    # 记录，不伪装
                    continue
                branch = semantic
            elif name == "SF":
                if semantic.untrainable:
                    self.s_degraded = True
                    continue
                branch = SpectralInteractionBranch(semantic)
            else:
                raise ValueError(f"未知分支: {name}")
            branch.fit(train)                 # ← 每个分支都在训练段 fit
            branches.append(branch)
        return branches

    # ---------- 设计矩阵 ----------

    def fit_design(self, train: FeatureBundle) -> None:
        train.validate_finite()
        self.branches = self._build_branches(train)

    def design(self, bundle: FeatureBundle) -> np.ndarray:
        bundle.validate_finite()
        chunks: list[np.ndarray] = []
        self._groups = []
        for branch in self.branches:
            block = branch.transform(bundle)
            if block.ndim != 2:
                raise ValueError(f"分支 {branch.name} 输出形状异常: {block.shape}")
            chunks.append(block)
            self._groups.extend([branch.name] * block.shape[1])
        if not chunks:
            raise ValueError("候选没有任何可用分支")
        design = np.c_[tuple(chunks)]
        if not np.isfinite(design).all():
            raise AssertionError("设计矩阵含非有限值")
        return design

    # ---------- 目标 ----------

    @staticmethod
    def residual_target(bundle: FeatureBundle) -> np.ndarray:
        """残差目标 = 标准化目标 - Last 锚点。用标准化坐标（协议 `target_scale`）。"""
        if bundle.targets_standardized is None:
            raise ValueError("Bundle 缺少 targets_standardized")
        y = np.asarray(bundle.targets_standardized, dtype=np.float64)
        if y.ndim != 2:
            raise ValueError(f"目标必须是 (n, H)，收到 {y.shape}")
        if len(y) != len(bundle.numeric_history):
            raise ValueError(f"目标与特征长度不一致: {len(y)} vs {len(bundle.numeric_history)}")
        return y - bundle.numeric_history[:, -1, None]

    @staticmethod
    def _last_anchor(bundle: FeatureBundle) -> np.ndarray:
        return np.asarray(bundle.numeric_history[:, -1, None], dtype=np.float64)

    @staticmethod
    def _as_matrix(raw: np.ndarray, n: int) -> np.ndarray:
        return np.asarray(raw).reshape(n, -1)

    # ---------- 逐分支 α 选择 ----------

    def select_alphas(self, train: FeatureBundle, cal: FeatureBundle, passes: int = 2) -> None:
        """校准段坐标下降选 α。**测试段从不参与。**"""
        design_train = self.design(train)
        target_train = self.residual_target(train)
        design_cal = self.design(cal)
        last_cal = self._last_anchor(cal)
        y_cal = np.asarray(cal.targets_standardized, dtype=np.float64)

        active = [b.name for b in self.branches]
        current = {n: PROTOCOL_ALPHAS[len(PROTOCOL_ALPHAS) // 2] for n in active}
        for _ in range(passes):
            for name in active:
                best, best_mse = current[name], float("inf")
                for alpha in PROTOCOL_ALPHAS:
                    trial = dict(current, **{name: float(alpha)})
                    w = _group_ridge(design_train, target_train, self._groups, trial)
                    pred = last_cal + self._as_matrix(design_cal @ w, len(design_cal))
                    mse = float(np.mean((pred - y_cal) ** 2))
                    if mse < best_mse:
                        best, best_mse = float(alpha), mse
                current[name] = best
        self.alpha_by_group = current

    def refit(self, fit: FeatureBundle) -> None:
        """按已冻结的 α 在给定段重拟合。置换与正式重拟合都走这里。"""
        if not self.alpha_by_group:
            raise RuntimeError("必须先 select_alphas 才能 refit")
        design = self.design(fit)
        target = self.residual_target(fit)
        self.weights = _group_ridge(design, target, self._groups, self.alpha_by_group)

    # ---------- 预测与贡献 ----------

    def predict(self, bundle: FeatureBundle) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("必须先 refit 才能 predict")
        design = self.design(bundle)
        return self._last_anchor(bundle) + self._as_matrix(design @ self.weights, len(design))

    def contributions(self, bundle: FeatureBundle) -> dict[str, np.ndarray]:
        """逐分支贡献 ``X_b w_b``。用于可追溯性与"无文本贡献为零"的检验。"""
        if self.weights is None:
            raise RuntimeError("必须先 refit 才能取分支贡献")
        design = self.design(bundle)
        out: dict[str, np.ndarray] = {}
        start = 0
        for branch in self.branches:
            width = branch.width
            out[branch.name] = design[:, start : start + width] @ self.weights[start : start + width, :]
            start += width
        return out

    @property
    def n_parameters(self) -> int:
        return 0 if self.weights is None else int(self.weights.size)
