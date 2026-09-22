"""主控路由指定的数值回退模型（第四轮 A.1 要求"数值回退也要有明确测试路径"）。

`split_spec_v2.json` 的 `numeric_fallback_candidates` 是::

    ["Last", "SeasonalNaive", "AR-Ridge", "N"]

其中 `N` 已在 `candidates.py` 里实现（逐分支惩罚 Ridge 的纯数值分支）。
本模块补齐另外三个 —— 主控路由 `selected == "numeric_fallback"` 时，
测试入口必须能**按路由指定的那个模型**出预测，而不是含糊地跑一个别的。

全部只用训练段信息拟合，**不读测试真值**。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from branches import FeatureBundle  # noqa: E402
from candidates import PROTOCOL_ALPHAS  # noqa: E402

SUPPORTED = ("Last", "SeasonalNaive", "AR-Ridge")


def last_forecast(bundle: FeatureBundle) -> np.ndarray:
    """Last：把窗口最后一个观测复制到整个跨度。"""
    horizon = np.asarray(bundle.targets_standardized).shape[1]
    return np.repeat(bundle.numeric_history[:, -1:], horizon, axis=1).astype(np.float64)


def seasonal_naive_forecast(bundle: FeatureBundle, seasonal_period: int) -> np.ndarray:
    """SeasonalNaive：取上一个季节周期的同相位观测。"""
    horizon = np.asarray(bundle.targets_standardized).shape[1]
    history = np.asarray(bundle.numeric_history, dtype=np.float64)
    if history.shape[1] < seasonal_period:
        raise ValueError(
            f"SeasonalNaive 需要至少 {seasonal_period} 步历史，实得 {history.shape[1]}")
    tail = history[:, -seasonal_period:]
    offsets = np.arange(horizon) % seasonal_period
    return tail[:, offsets]


def ar_ridge_forecast(
    train: FeatureBundle, target: FeatureBundle, lags: int = 8, horizon: int | None = None,
) -> np.ndarray:
    """AR-Ridge：以滞后值为特征的自回归 Ridge，逐跨度直接预测（direct multi-step）。

    α 在**训练段**以留一式的内部划分选择 —— 不使用校准段之外的任何信息，
    更不接触测试真值。
    """
    y_train = np.asarray(train.targets_standardized, dtype=np.float64)
    h = int(horizon if horizon is not None else y_train.shape[1])
    x_train = np.asarray(train.numeric_history, dtype=np.float64)
    x_test = np.asarray(target.numeric_history, dtype=np.float64)
    if x_train.shape[1] < lags or x_test.shape[1] < lags:
        raise ValueError(f"AR-Ridge 需要至少 {lags} 步历史")

    def design(matrix: np.ndarray) -> np.ndarray:
        return np.stack([matrix[:, -(lag + 1)] for lag in range(lags)], axis=1)

    fit_x, query_x = design(x_train), design(x_test)
    anchor_train = x_train[:, -1, None]
    anchor_query = x_test[:, -1, None]

    # 训练段内部再切一刀用于选 α —— 校准段留给主控的选择期，不在这里用。
    # 目标保持 (n, H) 二维：Ridge 原生支持多输出，展平会与特征行数不一致。
    cut = int(len(fit_x) * 0.8)
    if cut < lags + 2:
        alpha = PROTOCOL_ALPHAS[len(PROTOCOL_ALPHAS) // 2]
    else:
        best, best_mse = None, float("inf")
        for candidate_alpha in PROTOCOL_ALPHAS:
            model = Ridge(alpha=candidate_alpha, solver="cholesky").fit(
                fit_x[:cut], y_train[:cut] - anchor_train[:cut])
            pred = anchor_train[cut:] + model.predict(fit_x[cut:])
            mse = float(np.mean((pred - y_train[cut:]) ** 2))
            if mse < best_mse:
                best, best_mse = candidate_alpha, mse
        alpha = best

    model = Ridge(alpha=alpha, solver="cholesky").fit(fit_x, y_train - anchor_train)
    return anchor_query + model.predict(query_x).reshape(len(query_x), h)


def numeric_fallback_predict(
    name: str, train: FeatureBundle, target: FeatureBundle, *, seasonal_period: int = 12,
) -> np.ndarray:
    """按名字出预测。不支持的模型**明确报错**，不静默换成别的。"""
    if name == "Last":
        return last_forecast(target)
    if name == "SeasonalNaive":
        return seasonal_naive_forecast(target, seasonal_period)
    if name == "AR-Ridge":
        return ar_ridge_forecast(train, target)
    raise NotImplementedError(
        f"数值回退模型 {name!r} 未实现。已实现: {SUPPORTED} 与 candidates.py 的 'N'。")
