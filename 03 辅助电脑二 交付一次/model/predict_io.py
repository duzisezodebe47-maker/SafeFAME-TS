"""逐起点预测的持久化与契约。

任务书 §1.3 要求每条预测至少带以下字段：

    task_id, fold_id, origin_id, candidate_id, seed, y_pred,
    配置哈希, 特征哈希, 代码提交

本模块只负责**写出**预测；测试期真实目标由主控评测器关联，
训练脚本不得读取测试真值来决定保留哪个候选。
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# 契约字段顺序固定，便于主控按列读取
CONTRACT_FIELDS = (
    "task_id",
    "fold_id",
    "origin_id",
    "candidate_id",
    "seed",
    "horizon",
    "y_pred",
    "config_hash",
    "feature_hash",
    "code_commit",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def hash_config(config: dict) -> str:
    """配置哈希：键排序后序列化，保证同一配置在任何机器上哈希一致。"""
    blob = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return sha256_bytes(blob.encode("utf-8"))


def hash_features(bundle) -> str:
    """特征哈希：对参与本次训练的特征数组取摘要。

    只散列**值**，不含形状以外的元数据，使同一特征在任何机器上哈希一致。
    """
    digest = hashlib.sha256()
    for name in ("x", "report", "search", "quality", "origins"):
        array = np.ascontiguousarray(getattr(bundle, name))
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def code_commit(repo_root: Path) -> str:
    """当前代码提交。取不到时返回 'unknown' 而不是编造。"""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass
class PredictionRecord:
    """一行预测。字段与 CONTRACT_FIELDS 一致。"""

    task_id: str
    fold_id: int
    origin_id: int
    candidate_id: str
    seed: int
    horizon: int
    y_pred: float
    config_hash: str
    feature_hash: str
    code_commit: str


@dataclass
class PredictionSet:
    """一个候选在指定任务/折/种子下的全部逐起点预测。"""

    records: list[PredictionRecord] = field(default_factory=list)

    def extend_grid(
        self,
        origins: np.ndarray,
        predictions: np.ndarray,
        *,
        task_id: str,
        fold_id: int,
        candidate_id: str,
        seed: int,
        config_hash: str,
        feature_hash: str,
        commit: str,
    ) -> None:
        """把 (n, H) 的预测展平成逐起点、逐跨度的行。

        ``predictions`` 必须是二维 (n, H)。H=1 时 scikit-learn 风格的库
        会把结果压成一维，本函数显式拒绝，避免历史上出现过的 (n, n) 广播。
        """
        if predictions.ndim != 2:
            raise ValueError(f"预测必须是 (n, H)，收到 {predictions.shape}")
        if predictions.shape[0] != len(origins):
            raise ValueError(f"预测行数 {predictions.shape[0]} != 起点数 {len(origins)}")
        if not np.isfinite(predictions).all():
            raise AssertionError("预测含非有限值，拒绝写出")

        for row, origin in enumerate(origins):
            for step in range(predictions.shape[1]):
                self.records.append(
                    PredictionRecord(
                        task_id=task_id,
                        fold_id=int(fold_id),
                        origin_id=int(origin),
                        candidate_id=candidate_id,
                        seed=int(seed),
                        horizon=step + 1,
                        y_pred=float(predictions[row, step]),
                        config_hash=config_hash,
                        feature_hash=feature_hash,
                        code_commit=commit,
                    )
                )

    def write(self, path: Path) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CONTRACT_FIELDS)
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))
        return len(self.records)


def validate_alignment(origins: np.ndarray, expected: np.ndarray, label: str) -> None:
    """拒绝错位的起点 ID。

    任务书 §5 要求"任一预测行缺 ID、错位…时给出明确失败，不静默丢弃"。
    """
    if len(origins) != len(expected):
        raise AssertionError(f"{label}: 起点数不一致 {len(origins)} != {len(expected)}")
    if not np.array_equal(np.asarray(origins), np.asarray(expected)):
        mismatch = int(np.sum(np.asarray(origins) != np.asarray(expected)))
        raise AssertionError(f"{label}: 起点 ID 错位（{mismatch} 处不匹配）")
    if len(set(int(o) for o in origins)) != len(origins):
        raise AssertionError(f"{label}: 起点 ID 重复")
