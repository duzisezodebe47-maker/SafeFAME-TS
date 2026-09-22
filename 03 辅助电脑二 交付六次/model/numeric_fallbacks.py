"""主控数值基线的**逐行对齐实现**（第五轮 A.2）。

## 为什么要重写

第四轮的 `numeric_fallbacks.py` 是**另一个模型**：末 8 个滞后作特征、在训练段内部
按 80/20 选 α。主控已经固定了实现 —— `team_work/main/src/team_eval/v2.py`
的 `numeric_baselines`，并在第五轮任务书 A.2 里点名要求统一：

> 当前 `numeric_fallbacks.py` 的末 8 滞后 + train 内部 80/20 选 α 是另一个模型。
> 按主控固定实现改写，并用相同合成 Bundle 在 calibration/decision/test 比对
> 逐起点逐步输出和 α；偏差需在数值容差内。

本模块是主控实现的**逐行等价移植**，差异只有：

  - 主控读 `bundle["samples"]/["arrays"]`，本模块读 `FeatureBundle`；
  - 主控用 `np.linalg.solve`，本模块同用（不做 cholesky 改写，避免引入数值差异）。

## 主控口径（逐条对应）

  输入      ：**全部数值历史窗口** `x`（不是末 N 个滞后）
  Last      ：`x[:, -1]` 复制到 H 步
  Seasonal  ：`x[:, -P + (step % P)]`，step = 0..H-1
  AR-Ridge  ：**训练段**去中心化拟合，**校准段**选 7 档 α，
              预测 `(x - x_mean) @ weights + y_mean`（**全部行**，含测试段）
  目标尺度  ：标准化
  α 平局    ：按 (loss, alpha) 取最小 —— 与主控 `min(candidates, key=...)` 一致

**不读测试真值做任何拟合或选择**：α 只用校准段，拟合只用训练段。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from branches import FeatureBundle  # noqa: E402
from candidates import PROTOCOL_ALPHAS  # noqa: E402

SUPPORTED = ("Last", "SeasonalNaive", "AR-Ridge")


class BaselineError(RuntimeError):
    """主控数值基线的前置条件不满足。**不得用近似实现替代。**"""


def numeric_baselines(
    features: FeatureBundle,
    segments: np.ndarray,
    fit_targets: np.ndarray,
    alpha_grid: tuple[float, ...] = PROTOCOL_ALPHAS,
    seasonal_period: int = 12,
) -> dict:
    """主控 `team_eval/v2.numeric_baselines` 的逐行等价实现（第六轮修复 2）。

    **真值与特征在接口上分离**（任务书要求「接口中确实不存在真值字段」）::

        features     : 推理视图 —— `targets` / `targets_standardized` 必须为 None
        fit_targets  : (n, H)；**仅 train/calibration 行**有值，其余行必须为 NaN

    这样测试段的真值**根本传不进本函数**，而不是靠"我们不读它"的约定。
    算法本身与主控逐行一致：全部历史窗口作输入、训练段去中心化拟合、
    校准段选 7 档 α、`(loss, alpha)` 平局规则、逐 H 直接输出标准化尺度。
    """
    if features.targets is not None or features.targets_standardized is not None:
        raise BaselineError(
            "numeric_baselines 只接受推理视图：features 的 targets/targets_standardized 必须为 None")
    x = np.asarray(features.numeric_history, dtype=float)
    segments = np.asarray(segments)
    fit_targets = np.asarray(fit_targets, dtype=float)
    if fit_targets.shape[0] != x.shape[0]:
        raise BaselineError(f"fit_targets 行数 {fit_targets.shape[0]} != 特征行数 {x.shape[0]}")

    train = segments == "train"
    cal = segments == "calibration"
    allowed = train | cal
    # 真值只允许出现在 train/cal —— 测试段掺进真值必须硬失败
    if not np.isnan(fit_targets[~allowed]).all():
        raise BaselineError("fit_targets 在 train/calibration 之外含真值 —— 真值不得越界")
    if np.isnan(fit_targets[allowed]).any():
        raise BaselineError("fit_targets 在 train/calibration 行不得为 NaN")
    if train.sum() < 2 or not cal.any():
        raise BaselineError("数值基线缺少 train/calibration 行")
    if seasonal_period < 1 or seasonal_period > x.shape[1]:
        raise BaselineError(
            f"seasonal_period={seasonal_period} 超出历史窗口长度 {x.shape[1]}")
    if not alpha_grid or any(not np.isfinite(float(a)) or float(a) <= 0 for a in alpha_grid):
        raise BaselineError("α 网格必须为正的有限值")

    horizon = fit_targets.shape[1]
    y = fit_targets                      # 只有 train/cal 行被真正使用
    last = np.repeat(x[:, -1:], horizon, axis=1)
    seasonal = np.stack(
        [x[:, -seasonal_period + (step % seasonal_period)] for step in range(horizon)],
        axis=1)

    x_mean, y_mean = x[train].mean(axis=0), y[train].mean(axis=0)
    xc, yc = x[train] - x_mean, y[train] - y_mean
    gram, cross = xc.T @ xc, xc.T @ yc

    candidates: dict[float, tuple[float, np.ndarray]] = {}
    for alpha in sorted(set(map(float, alpha_grid))):
        weights = np.linalg.solve(gram + alpha * np.eye(x.shape[1]), cross)
        prediction = (x - x_mean) @ weights + y_mean
        loss = float(np.mean((prediction[cal] - y[cal]) ** 2))
        candidates[alpha] = (loss, prediction)

    # 平局按 α 值取最小 —— 与主控 min(candidates, key=lambda a: (loss, a)) 一致
    chosen_alpha = min(candidates, key=lambda a: (candidates[a][0], a))
    predictions = {
        "Last": last,
        "SeasonalNaive": seasonal,
        "AR-Ridge": candidates[chosen_alpha][1],
    }
    for candidate, prediction in predictions.items():
        if prediction.shape != (x.shape[0], horizon) or not np.isfinite(prediction).all():
            raise BaselineError(f"数值基线预测非法: {candidate}")

    return {
        "predictions": predictions,
        "ridge_alpha": chosen_alpha,
        "ridge_calibration_mse": candidates[chosen_alpha][0],
        "fit_rows": int(train.sum()),
        "calibration_rows": int(cal.sum()),
        "alpha_grid": sorted(candidates),
        "target_scale": "train_only_standardized_OT",
    }


def fit_targets_with_nan(bundle: FrozenBundleLike, segments: np.ndarray) -> np.ndarray:
    """构造 `fit_targets`：只保留 train/calibration 行的真值，其余为 NaN。"""
    y = np.asarray(bundle.targets_standardized, dtype=float).copy()
    allowed = (np.asarray(segments) == "train") | (np.asarray(segments) == "calibration")
    y[~allowed] = np.nan
    return y


def numeric_fallback_predict(
    name: str, features: FeatureBundle, segments: np.ndarray, fit_targets: np.ndarray,
    *, seasonal_period: int = 12, alpha_grid: tuple[float, ...] = PROTOCOL_ALPHAS,
) -> np.ndarray:
    """按名字取主控基线的预测。不支持的模型**明确报错**，不静默换模型。"""
    result = numeric_baselines(features, segments, fit_targets, alpha_grid, seasonal_period)
    if name not in result["predictions"]:
        raise NotImplementedError(
            f"数值回退模型 {name!r} 未实现。已实现: {SUPPORTED} 与 candidates.py 的 'N'。")
    return result["predictions"][name]
