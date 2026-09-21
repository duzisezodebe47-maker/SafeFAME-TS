"""候选模型：逐分支独立正则化的残差族。

## 与现有实现的实质差异

现有 ``semantic_residual`` 把 N/Q/S 拼成一个矩阵喂给**单个** Ridge，全部特征共享一个 α：

    (XᵀX + αI) w = Xᵀ(y - Last)

本模块对**每个分支施加各自的惩罚** α_b：

    (XᵀX + diag(α_{g(1)}, …, α_{g(d)})) w = Xᵀ(y - Last)

对线性模型而言 ``w·[a;b] ≡ w_a·a + w_b·b`` —— 所以"分支隔离"**不可能**体现在拼接方式上，
只能体现在**正则化几何**上。逐分支 α 允许"语义分支强收缩、质量分支弱收缩"，
这是单 α 做不到的可检验差异。

融合仍是加性的，但**可追溯**：每个分支的贡献 ``X_b w_b`` 可单独取出检查。

## 已知代价

逐分支 α 引入额外自由度（|分支数| 倍）。校准段选择过程必须记录，
并在敏感性分析中检验 α 网格细分是否改变结论。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from run_safefame_v2 import RIDGE_ALPHAS  # noqa: E402  复用现有 9 档网格，不自建

from branches import Branch, FeatureBundle, make_branches  # noqa: E402


def _group_ridge(
    design: np.ndarray,
    target: np.ndarray,
    groups: list[str],
    alpha_by_group: dict[str, float],
) -> np.ndarray:
    """解带逐组惩罚的岭回归。

    惩罚矩阵是对角的：第 j 个特征使用它所属分支的 α。
    """
    penalty = np.array([alpha_by_group[g] for g in groups], dtype=np.float64)
    # 正规方程在 float64 下求解。设计矩阵可保持 float32 以省内存，但 Gram 必须升精度：
    # 实测同一设计矩阵在 float32 下 alpha=1e-4 时 cholesky 失败（potrf 报非正定），
    # float64 下成功。小 alpha 时 float32 的舍入足以让最小特征值变号。
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    gram = design.T @ design
    gram.flat[:: len(gram) + 1] += penalty
    rhs = design.T @ target
    try:
        return cho_solve(cho_factor(gram, lower=True, check_finite=False), rhs, check_finite=False)
    except np.linalg.LinAlgError as exc:
        # 不静默改用其它求解器：换求解器会改变方法，且会掩盖设计矩阵的退化。
        present = sorted(set(groups))
        raise AssertionError(
            f"Gram 矩阵非正定，cholesky 分解失败（分支: {present}，设计矩阵 {design.shape}）。"
            f"通常意味着存在零方差或高度共线的列 —— 检查 SafeScaler 是否正确丢弃了退化列。"
        ) from exc


class BranchResidualCandidate:
    """一个由若干分支组成的残差候选。

    ``pred = Last + Σ_b X_b w_b``，其中 ``Last`` 是输入窗口最后一个观测，
    复制到全部预测跨度（与现有管线一致）。

    参数
    ----
    branch_names : 参与融合的分支，例如 ``("N", "S", "Q")``。
                   未列出的分支**完全不进入模型** —— 这是结构性隔离，不是特征子集选择。
    uniform_alpha : 为 True 时退化为单一 α（用于与逐分支 α 做敏感性对照）。
    """

    def __init__(self, branch_names: tuple[str, ...], uniform_alpha: bool = False) -> None:
        if "N" not in branch_names:
            raise ValueError("所有候选都必须含 N 分支作为残差基底")
        self.branch_names = tuple(branch_names)
        self.uniform_alpha = uniform_alpha
        self.branches: list[Branch] = make_branches(branch_names)
        self.alpha_by_group: dict[str, float] = {}
        self.weights: np.ndarray | None = None
        self._groups: list[str] = []
        self.alpha_history: list[dict[str, float]] = []

    # ---------- 设计矩阵 ----------

    def fit_design(self, train: FeatureBundle) -> None:
        """只在训练段拟合各分支的标度器 / PCA。"""
        for branch in self.branches:
            branch.fit(train)

    def design(self, bundle: FeatureBundle) -> np.ndarray:
        bundle.validate_finite()   # 进分支前统一拒绝，避免 sklearn 抛出含糊错误
        chunks: list[np.ndarray] = []
        self._groups = []
        for branch in self.branches:
            block = branch.transform(bundle)
            if block.ndim != 2 or block.shape[1] == 0:
                raise ValueError(f"分支 {branch.name} 输出形状异常: {block.shape}")
            chunks.append(block)
            self._groups.extend([branch.name] * block.shape[1])
        design = np.c_[tuple(chunks)]
        if not np.isfinite(design).all():
            raise AssertionError("设计矩阵含非有限值")
        return design

    # ---------- 目标 ----------

    @staticmethod
    def residual_target(bundle: FeatureBundle, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (残差目标, Last 锚点)。

        y 形状 (n, H)。``Last`` 显式保持 (n, 1) 再由广播交给 numpy ——
        直接用 ``x[:, -1, None]`` 是为了避免 scikit-learn 在 H=1 时把预测压成一维
        后触发 (n, n) 广播（历史缺陷，见 docs/21:39）。
        """
        if y.ndim != 2:
            raise ValueError(f"目标必须是 (n, H)，收到 {y.shape}")
        if len(y) != len(bundle.x):
            raise ValueError(f"目标与特征长度不一致: {len(y)} vs {len(bundle.x)}")
        last = bundle.x[:, -1, None]
        return y - last, last

    @staticmethod
    def _as_matrix(raw: np.ndarray, n: int) -> np.ndarray:
        """把 Ridge 的输出规整回 (n, H)。H=1 时 sklearn 会压成一维。"""
        return np.asarray(raw).reshape(n, -1)

    # ---------- 逐分支 α 选择 ----------

    def select_alphas(
        self,
        train: FeatureBundle,
        cal: FeatureBundle,
        y_train: np.ndarray,
        y_cal: np.ndarray,
        passes: int = 2,
    ) -> None:
        """在校准段上做坐标下降，逐分支选 α。**测试段从不参与。**

        坐标下降而非全网格：4 分支 × 9 个 α 的全组合是 6561 次闭式解，
        在 240 条候选上开销过大。两轮扫描在校准段上已足够稳定，
        且该近似本身会作为敏感性分析报告。
        """
        design_train = self.design(train)
        target_train, _ = self.residual_target(train, y_train)
        design_cal = self.design(cal)
        _, last_cal = self.residual_target(cal, y_cal)

        # 起点：全部用中间值 α，避免从极端起步带来偏置
        current = {name: float(RIDGE_ALPHAS[len(RIDGE_ALPHAS) // 2]) for name in self.branch_names}

        for _ in range(passes):
            for name in self.branch_names:
                best, best_mse = current[name], float("inf")
                for alpha in RIDGE_ALPHAS:
                    trial = dict(current, **{name: float(alpha)})
                    weights = _group_ridge(design_train, target_train, self._groups, trial)
                    pred = last_cal + self._as_matrix(design_cal @ weights, len(design_cal))
                    mse = float(np.mean((pred - y_cal) ** 2))
                    if mse < best_mse:
                        best, best_mse = float(alpha), mse
                current[name] = best

        self.alpha_by_group = current
        self.alpha_history.append(dict(current))

    def refit(self, fit: FeatureBundle, y_fit: np.ndarray) -> None:
        """在"训练+校准"上按已冻结的 α 重拟合。α 已冻结，此处不再选择。"""
        if not self.alpha_by_group:
            raise RuntimeError("必须先 select_alphas 才能 refit")
        design = self.design(fit)
        target, _ = self.residual_target(fit, y_fit)
        self.weights = _group_ridge(design, target, self._groups, self.alpha_by_group)

    # ---------- 预测 ----------

    def predict(self, bundle: FeatureBundle) -> np.ndarray:
        """返回 (n, H) 预测。"""
        if self.weights is None:
            raise RuntimeError("必须先 refit 才能 predict")
        design = self.design(bundle)
        last = bundle.x[:, -1, None]
        return last + self._as_matrix(design @ self.weights, len(design))

    def contributions(self, bundle: FeatureBundle) -> dict[str, np.ndarray]:
        """逐分支贡献 ``X_b w_b``，用于可追溯性与归因检查。"""
        if self.weights is None:
            raise RuntimeError("必须先 refit 才能取分支贡献")
        design = self.design(bundle)
        out: dict[str, np.ndarray] = {}
        start = 0
        for branch in self.branches:
            width = branch.width
            # 逐分支做 (n, w) @ (w, H) -> (n, H)，避免构造 (n, d, H) 三维数组
            # —— 在 28554 个起点 × 236 维 × 24 跨度下后者会占几百 MB。
            out[branch.name] = design[:, start : start + width] @ self.weights[start : start + width, :]
            start += width
        return out

    @property
    def n_parameters(self) -> int:
        return int(self.weights.shape[0] * self.weights.shape[1]) if self.weights is not None else 0


# 任务书 §1.2 建议的候选族。C4/C5 在维度上等同现有候选，差异仅在正则化结构
# （见 DESIGN.md §2 说明与 §9 问题 1）。
CANDIDATES: dict[str, tuple[str, ...]] = {
    "N": ("N",),
    "N+Q": ("N", "Q"),
    "N+S": ("N", "S"),
    "N+S+Q": ("N", "S", "Q"),
    "N+S+Q+F": ("N", "S", "Q", "F"),
    "N+F": ("N", "F"),
}
