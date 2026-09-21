"""分支定义与逐分支预处理。

设计要点（见 DESIGN.md §1.2）：

- 每个分支有**自己的**标度器，只在训练段 fit，分支之间互不影响。
  这是与现有 ``ResidualBuilder`` 的关键差异：后者把 N/Q/S 拼成一个矩阵共用一套尺度，
  导致置换检验时某一分支的尺度重拟合会牵动其它分支。
- 分支是**可插拔**的：候选由分支名列表声明，缺省的分支不参与融合。
- 语义分支使用 PCA，且 PCA 只在训练段 fit，避免测试段信息泄漏。

与现有实现的关系：复用 ``run_famets_selective.frequency_statistics`` 与
``run_famets_selective.semantic_rows`` 的特征定义，不重新发明。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from run_famets_selective import frequency_statistics  # noqa: E402

# 语义分支降维后的宽度。与现有 semantic_residual 的 PCA(24) 保持一致，
# 否则维度差异会让"公平对比"无法辩护。
SEMANTIC_COMPONENTS = 24


@dataclass
class FeatureBundle:
    """一个数据切片的全部可用特征。字段名按主控契约冻结后可能调整。"""

    x: np.ndarray          # (n, L)     数值历史窗口
    report: np.ndarray     # (n, 384)   报告句向量
    search: np.ndarray     # (n, 384)   搜索句向量
    quality: np.ndarray    # (n, 10)    文本质量特征
    origins: np.ndarray    # (n,)       预测起点索引，用于追溯与错位检测

    def __post_init__(self) -> None:
        n = len(self.x)
        for name in ("report", "search", "quality", "origins"):
            got = len(getattr(self, name))
            if got != n:
                raise ValueError(f"FeatureBundle 长度不一致: x={n}, {name}={got}")

    def validate_finite(self) -> None:
        """在任何分支处理前统一拒绝非有限值。

        不依赖 sklearn 的隐式报错：那会给出含糊的 "value too large for dtype"，
        且不同分支报错时机不一致。
        """
        for name in ("x", "report", "search", "quality"):
            array = getattr(self, name)
            if not np.isfinite(array).all():
                count = int((~np.isfinite(array)).sum())
                raise AssertionError(f"FeatureBundle.{name} 含 {count} 个非有限值")

    def slice(self, index: np.ndarray) -> "FeatureBundle":
        return FeatureBundle(
            x=self.x[index],
            report=self.report[index],
            search=self.search[index],
            quality=self.quality[index],
            origins=self.origins[index],
        )


class Branch(Protocol):
    """分支协议：能 fit，能 transform，能报告自己的宽度。"""

    name: str

    def fit(self, bundle: FeatureBundle) -> None: ...
    def transform(self, bundle: FeatureBundle) -> np.ndarray: ...
    @property
    def width(self) -> int: ...


class SafeScaler:
    """在训练段 fit 的标准化器，**丢弃零方差列**。

    为什么必须丢：``StandardScaler`` 对恒定列输出全 0，使设计矩阵出现全零列，
    Gram 矩阵随之奇异 —— 最小特征值可为负，``cho_factor`` 直接抛
    ``LinAlgError: potrf``。这在 Climate 的质量特征上**实际发生过**
    （质量块含恒定列，96 列的设计矩阵秩只有 89）。

    丢弃判定只在训练段做，且**每次置换必须重新判定**（见 DESIGN.md §5.2）：
    置换会改变哪些列恒定，沿用上一次的判定会破坏置换交换性。
    """

    ZERO_VARIANCE = 1e-8

    def __init__(self) -> None:
        self.keep: np.ndarray | None = None
        self.scaler = StandardScaler()
        self.n_dropped = 0

    def fit(self, values: np.ndarray) -> "SafeScaler":
        spread = values.std(axis=0)
        self.keep = spread > self.ZERO_VARIANCE
        self.n_dropped = int((~self.keep).sum())
        if not self.keep.any():
            raise ValueError("全部列都是零方差，无法构建设计矩阵")
        self.scaler.fit(values[:, self.keep])
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.keep is None:
            raise RuntimeError("SafeScaler 必须先 fit")
        return self.scaler.transform(values[:, self.keep])

    @property
    def width(self) -> int:
        return 0 if self.keep is None else int(self.keep.sum())


class NumericBranch:
    """N：数值历史窗口。"""

    name = "N"

    def __init__(self) -> None:
        self.scaler = SafeScaler()

    def fit(self, bundle: FeatureBundle) -> None:
        self.scaler.fit(bundle.x)

    def transform(self, bundle: FeatureBundle) -> np.ndarray:
        return self.scaler.transform(bundle.x)

    @property
    def width(self) -> int:
        return self.scaler.width


class QualityBranch:
    """Q：文本覆盖率、时距、缺失状态等质量特征。**不含语义向量**。"""

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
    """S：语义向量。质量信息**不进入**本分支。"""

    name = "S"

    def __init__(self, components: int = SEMANTIC_COMPONENTS) -> None:
        self.components = components
        self.pca: PCA | None = None
        self.scaler = SafeScaler()

    def fit(self, bundle: FeatureBundle) -> None:
        embeddings = np.c_[bundle.report, bundle.search]
        count = min(self.components, len(embeddings) - 1, embeddings.shape[1])
        if count < 1:
            raise ValueError(f"语义分支样本不足: n={len(embeddings)}")
        self.pca = PCA(n_components=count, svd_solver="full").fit(embeddings)
        # PCA 分量可能退化为零方差（当嵌入矩阵秩不足时），同样需要丢弃
        self.scaler.fit(self.pca.transform(embeddings))

    def transform(self, bundle: FeatureBundle) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("SemanticBranch 必须先 fit")
        embeddings = np.c_[bundle.report, bundle.search]
        return self.scaler.transform(self.pca.transform(embeddings))

    @property
    def width(self) -> int:
        return self.scaler.width


class FrequencyBranch:
    """F：谱统计量。定义复用 run_famets_selective.frequency_statistics。"""

    name = "F"

    def __init__(self) -> None:
        self.scaler = SafeScaler()

    def fit(self, bundle: FeatureBundle) -> None:
        self.scaler.fit(frequency_statistics(bundle.x))

    def transform(self, bundle: FeatureBundle) -> np.ndarray:
        return self.scaler.transform(frequency_statistics(bundle.x))

    @property
    def width(self) -> int:
        return self.scaler.width


BRANCH_FACTORY: dict[str, type] = {
    "N": NumericBranch,
    "Q": QualityBranch,
    "S": SemanticBranch,
    "F": FrequencyBranch,
}


def make_branches(names: tuple[str, ...]) -> list[Branch]:
    unknown = set(names) - set(BRANCH_FACTORY)
    if unknown:
        raise ValueError(f"未知分支: {sorted(unknown)}")
    return [BRANCH_FACTORY[name]() for name in names]
