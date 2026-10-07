"""与 src/run_famets_selective.py 的十维谱统计逐行相同的轻量实现。"""
import numpy as np


def frequency_statistics(x):
    spectrum = np.abs(np.fft.rfft(x, axis=1))
    power = spectrum**2
    total = power.sum(1, keepdims=True).clip(1e-8)
    normalized = power / total
    bins = normalized.shape[1]
    edges = np.linspace(0, bins, 5, dtype=int)
    bands = np.column_stack([normalized[:, edges[i]:edges[i+1]].sum(1) for i in range(4)])
    frequencies = np.linspace(0.0, 1.0, bins)
    centroid = (normalized * frequencies).sum(1)
    entropy = -(normalized * np.log(normalized.clip(1e-12))).sum(1) / np.log(max(2, bins))
    dominant = np.argmax(normalized[:, 1:], axis=1) + 1 if bins > 1 else np.zeros(len(x))
    dominant = dominant / max(1, bins - 1)
    concentration = normalized.max(1)
    slope = x[:, -1] - x[:, 0]
    volatility = np.diff(x, axis=1).std(1)
    return np.c_[bands, centroid, entropy, dominant, concentration, slope, volatility].astype(np.float32)
