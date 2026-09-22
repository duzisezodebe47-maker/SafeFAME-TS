"""第二轮分支定义：消费主控冻结 Bundle，修复第一轮的两处 P1 阻断。

相对第一轮（`03 辅助电脑二 交付一次/model/branches.py`）的改动：

1. **新增 `text_available` 可得性掩码**（第一轮 P1 第 3 条）。
   第一轮的 `SemanticBranch` 只把零向量喂进 PCA，即使对零向量做中心化/标准化，
   仍可能产出非零贡献 —— 主控独立验收时点出了这一点。现在掩码从数据侧
   Bundle 一路传到贡献层：无文本起点令 S 的**贡献**严格为零，N 分支继续预测。

2. **新增 `SpectralInteractionBranch`（SF）**（第一轮 P1 第 4 条）。
   第一轮的 `F` 是 10 维纯数值谱统计，不含语义；协议规定它只能是诊断消融。
   真正的交互候选 SF = 训练段 PCA 语义（≤24 维）× 10 维数值谱统计的外积展平。

3. **α 网格改用协议冻结值**（见 `candidates.py` 注释中的冲突说明）。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from run_famets_selective import frequency_statistics  # noqa: E402

SEMANTIC_COMPONENTS = 24


@dataclass
class FeatureBundle:
    """主控冻结 Bundle 的一个切片。

    字段名与 `prediction_contract_v2.json` 的 `data_bundle_required` 对齐。
    `text_available` 是逐起点的布尔掩码 —— 数据侧 Bundle 已提供，不是本模块推断的。
    """

    numeric_history: np.ndarray   # (n, L)
    semantic: np.ndarray          # (n, 768)
    quality: np.ndarray           # (n, 10)
    text_available: np.ndarray    # (n,) bool
    origin_index: np.ndarray      # (n,)
    targets: np.ndarray = field(default=None)             # (n, H) 原始尺度
    targets_standardized: np.ndarray = field(default=None)  # (n, H) 训练段标准化

    def __post_init__(self) -> None:
        n = len(self.numeric_history)
        for name in ("semantic", "quality", "text_available", "origin_index"):
            got = len(getattr(self, name))
            if got != n:
                raise ValueError(f"FeatureBundle 长度不一致: numeric_history={n}, {name}={got}")
        self.text_available = np.asarray(self.text_available).astype(bool)
        if self.targets is not None and len(self.targets) != n:
            raise ValueError("targets 长度与特征不一致")

    def validate_finite(self) -> None:
        for name in ("numeric_history", "semantic", "quality"):
            array = getattr(self, name)
            if not np.isfinite(array).all():
                count = int((~np.isfinite(array)).sum())
                raise AssertionError(f"{name} 含 {count} 个非有限值")

    def slice(self, index: np.ndarray) -> "FeatureBundle":
        return FeatureBundle(
            numeric_history=self.numeric_history[index],
            semantic=self.semantic[index],
            quality=self.quality[index],
            text_available=self.text_available[index],
            origin_index=self.origin_index[index],
            targets=None if self.targets is None else self.targets[index],
            targets_standardized=(None if self.targets_standardized is None
                                  else self.targets_standardized[index]),
        )


class SafeScaler:
    """在训练段 fit 的标准化器，丢弃零方差列。

    第一轮的教训保留：`StandardScaler` 对恒定列输出全 0，会使设计矩阵出现全零列、
    Gram 奇异、cholesky 直接失败。**该判定必须每次置换重做**，否则破坏置换交换性。
    """

    ZERO_VARIANCE = 1e-8

    def __init__(self) -> None:
        self.keep: np.ndarray | None = None
        self.scaler = StandardScaler()
        self.n_dropped = 0

    def fit(self, values: np.ndarray) -> "SafeScaler":
        values = np.asarray(values, dtype=np.float64)
        self.keep = values.std(axis=0) > self.ZERO_VARIANCE
        self.n_dropped = int((~self.keep).sum())
        if not self.keep.any():
            raise ValueError("全部列零方差，无法构建设计矩阵")
        self.scaler.fit(values[:, self.keep])
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.keep is None:
            raise RuntimeError("SafeScaler 必须先 fit")
        return self.scaler.transform(np.asarray(values, dtype=np.float64)[:, self.keep])

    @property
    def width(self) -> int:
        return 0 if self.keep is None else int(self.keep.sum())


class Branch(Protocol):
    name: str
    def fit(self, bundle: FeatureBundle) -> None: ...
    def transform(self, bundle: FeatureBundle) -> np.ndarray: ...
    @property
    def width(self) -> int: ...


class NumericBranch:
    """N：OT 单变量历史（协议 `numeric_features: ["OT"]`）。"""

    name = "N"

    def __init__(self) -> None:
        self.scaler = SafeScaler()

    def fit(self, bundle: FeatureBundle) -> None:
        self.scaler.fit(bundle.numeric_history)

    def transform(self, bundle: FeatureBundle) -> np.ndarray:
        return self.scaler.transform(bundle.numeric_history)

    @property
    def width(self) -> int:
        return self.scaler.width


class QualityBranch:
    """Q：质量特征。**不含语义**。注意 `N+S` 候选中 Q 不进入设计矩阵。"""

    name = "Q"

    def __init__(self) -> None:
        self.scaler = SafeScaler()

    def fit(self, bundle: FeatureBundle) -> None:
        self.scaler.fit(bundle.quality)

    def transform(self, bundle: FeatureBundle) -> np.ndarray:
        return self.scaler.transform(bundle.quality)

    @property
    def width(self) -> int:
        return self.scaler.width


class SemanticBranch:
    """S：语义向量，受 `text_available` 掩码控制。

    **掩码语义（第一轮 P1 第 3 条）**：无文本起点上，本分支的**贡献严格为零**，
    且不得污染 PCA 的拟合。实现方式：
      - `fit` 只用 `text_available` 为真的行拟合 PCA 与标度器 —— 避免对零语义做
        均值中心化后产生伪信号（主控明确要求）；
      - 若训练段完全无文本，标记 `untrainable`，贡献恒为零并安全退化；
      - `transform` 后按掩码清零，使下游的贡献和严格为 0。
    """

    name = "S"

    def __init__(self, components: int = SEMANTIC_COMPONENTS) -> None:
        self.components = components
        self.pca: PCA | None = None
        self.scaler = SafeScaler()
        self.untrainable = False
        self.n_train_rows = 0

    def fit(self, bundle: FeatureBundle) -> None:
        mask = bundle.text_available
        self.n_train_rows = int(mask.sum())
        if self.n_train_rows < 2:
            # 训练段完全无文本：PCA 无从拟合，安全退化而不是造一个伪信号
            self.untrainable = True
            self.pca = None
            return
        self.untrainable = False
        embeddings = bundle.semantic[mask]
        count = min(self.components, len(embeddings) - 1, embeddings.shape[1])
        self.pca = PCA(n_components=count, svd_solver="full").fit(embeddings)
        self.scaler.fit(self.pca.transform(embeddings))

    def transform(self, bundle: FeatureBundle) -> np.ndarray:
        if self.untrainable:
            raise RuntimeError("训练段无文本，S 分支不可训练；应由候选层跳过该分支")
        if self.pca is None:
            raise RuntimeError("SemanticBranch 必须先 fit")
        projected = self.scaler.transform(self.pca.transform(bundle.semantic))
        # 无文本起点贡献严格为零
        return projected * bundle.text_available[:, None]

    @property
    def width(self) -> int:
        return 0 if self.pca is None else self.scaler.width


class FrequencyBranch:
    """F：纯数值谱统计（10 维）。**协议规定它只是诊断消融，不是门控候选。**"""

    name = "F"

    def __init__(self) -> None:
        self.scaler = SafeScaler()

    def fit(self, bundle: FeatureBundle) -> None:
        self.scaler.fit(frequency_statistics(bundle.numeric_history))

    def transform(self, bundle: FeatureBundle) -> np.ndarray:
        return self.scaler.transform(frequency_statistics(bundle.numeric_history))

    @property
    def width(self) -> int:
        return self.scaler.width


class SpectralInteractionBranch:
    """SF：语义 × 数值谱的显式交互（第一轮 P1 第 4 条，本轮新增）。

    定义：``flatten(PCA_{≤24}(semantic) ⊗ frequency_10(numeric_history))``，
    即逐元素外积展平，最多 24 × 10 = 240 维。

    与历史 `frequency_residual` 的结构**不同**（后者是 24×8 的交互且语义 PCA 未受掩码约束），
    报告时按"新交互候选"命名，不得声称等价。

    **每次置换必须重新拟合交互尺度** —— docs/23 的频率候选正是因为沿用旧尺度，
    逐行 p 从 0.021 塌到 0.085。
    """

    name = "SF"

    def __init__(self, semantic: SemanticBranch, spectral_dims: int = 10) -> None:
        self.semantic = semantic
        self.spectral_dims = spectral_dims
        self.spectral_scaler = SafeScaler()
        self.interaction_scaler = SafeScaler()
        self.n_interaction = 0

    def fit(self, bundle: FeatureBundle) -> None:
        spectral = frequency_statistics(bundle.numeric_history)
        self.spectral_scaler.fit(spectral)
        interaction = self._interaction(bundle)
        self.n_interaction = interaction.shape[1]
        self.interaction_scaler.fit(interaction)

    def _interaction(self, bundle: FeatureBundle) -> np.ndarray:
        semantic = self.semantic.transform(bundle)
        spectral = self.spectral_scaler.transform(frequency_statistics(bundle.numeric_history))
        dims = min(self.spectral_dims, spectral.shape[1])
        outer = semantic[:, :, None] * spectral[:, None, :dims]
        return outer.reshape(len(semantic), -1)

    def transform(self, bundle: FeatureBundle) -> np.ndarray:
        interaction = self._interaction(bundle)
        scaled = self.interaction_scaler.transform(interaction)
        # 无文本起点的交互贡献同样严格为零
        return scaled * bundle.text_available[:, None]

    @property
    def width(self) -> int:
        return self.interaction_scaler.width


BASIC_BRANCHES: dict[str, type] = {
    "N": NumericBranch, "Q": QualityBranch, "S": SemanticBranch, "F": FrequencyBranch,
}
