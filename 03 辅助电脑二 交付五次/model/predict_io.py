"""预测输出契约（第二轮 P1 第 1 条）。

字段与 `prediction_contract_v2.json` 的 `prediction_csv_required` **逐字对齐**::

    task_id, fold_id, origin_id, origin_index, segment, scenario, candidate_id,
    seed, step, y_pred, target_scale, bundle_signature, config_sha256, code_commit

与第一轮的差异（主控验收指出的协议不一致）：

| | 第一轮 | 本轮 |
|---|---|---|
| 字段数 | 10 | **14**（新增 segment / scenario / target_scale / origin_index） |
| `origin_id` | 整数 | **字符串** `Domain:hH:fF:o<index>` |
| 段 | 只有 test | **四段**，测试预测另行发布 |
| 步长 | 1-based（同） | 1-based（同） |

契约的 `prediction_key` = `[task_id, fold_id, segment, origin_id, step, candidate_id, seed]`，
`prediction_step` 要求**每个起点、每个候选、每个种子的每一步 1..H 恰好出现一次**。
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

CONTRACT_FIELDS = (
    "task_id", "fold_id", "origin_id", "origin_index", "segment", "scenario",
    "candidate_id", "seed", "step", "y_pred", "target_scale",
    "bundle_signature", "config_sha256", "code_commit",
)

PREDICTION_KEY = ("task_id", "fold_id", "segment", "origin_id", "step", "candidate_id", "seed")

TARGET_SCALE = "train_only_standardized_OT"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_sha256(config: dict) -> str:
    """配置哈希：键排序序列化，保证跨机器一致。"""
    return sha256_text(json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def weight_hash(weights) -> str:
    """权重哈希：形状 + 字节。便携、与机器无关，供 A.6 的权重一致性核对。"""
    array = np.ascontiguousarray(weights, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def code_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass
class PredictionRecord:
    task_id: str
    fold_id: int
    origin_id: str
    origin_index: int
    segment: str
    scenario: str
    candidate_id: str
    seed: int
    step: int
    y_pred: float
    target_scale: str
    bundle_signature: str
    config_sha256: str
    code_commit: str


class PredictionWriter:
    """收集逐起点×步长预测并写出契约 CSV。

    `extend` 拒绝一维预测 —— H=1 时某些库会把结果压成一维，第一轮的
    (n, n) 广播缺陷正是从这里进来的。
    """

    def __init__(self) -> None:
        self.records: list[PredictionRecord] = []

    def extend(
        self, predictions: np.ndarray, *, origin_id: np.ndarray, origin_index: np.ndarray,
        task_id: str, fold_id: int, segment: str, scenario: str, candidate_id: str,
        seed: int, bundle_signature: str, config_sha256_value: str, commit: str,
    ) -> None:
        if predictions.ndim != 2:
            raise ValueError(f"预测必须是 (n, H)，收到 {predictions.shape}；"
                             f"一维预测会触发 H=1 广播缺陷")
        if predictions.shape[0] != len(origin_id):
            raise ValueError(f"预测行数 {predictions.shape[0]} != 起点数 {len(origin_id)}")
        if not np.isfinite(predictions).all():
            raise AssertionError("预测含非有限值，拒绝写出")

        # `segment` 可以是单个字符串，也可以是**逐行**的段名数组 ——
        # 后者用于按 Bundle 行序导出多个段（主控的 grid 校验是**有序**比较）
        if isinstance(segment, str):
            segments_per_row = [segment] * predictions.shape[0]
        else:
            segments_per_row = [str(x) for x in segment]
        if len(segments_per_row) != predictions.shape[0]:
            raise ValueError(
                f"segment 数组长度 {len(segments_per_row)} != 预测行数 {predictions.shape[0]}")

        for row in range(predictions.shape[0]):
            for step in range(predictions.shape[1]):
                self.records.append(PredictionRecord(
                    task_id=task_id, fold_id=int(fold_id), origin_id=str(origin_id[row]),
                    origin_index=int(origin_index[row]), segment=segments_per_row[row],
                    scenario=scenario,
                    candidate_id=candidate_id, seed=int(seed), step=step + 1,
                    y_pred=float(predictions[row, step]), target_scale=TARGET_SCALE,
                    bundle_signature=bundle_signature, config_sha256=config_sha256_value,
                    code_commit=commit,
                ))

    def write(self, path: Path) -> int:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # lineterminator="\n"：csv 默认 CRLF，但仓库 .gitattributes 对 *.csv 声明 eol=lf。
        # 不指定会让提交字节 ≠ 产出字节 —— 本项目此前哈希对不上的同类成因。
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CONTRACT_FIELDS, lineterminator="\n")
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))
        return len(self.records)

    def validate_complete(self, expected_origins: int, horizon: int, seed: int) -> None:
        """契约 `prediction_step`：每个起点每一步恰好一次。"""
        if not self.records:
            raise AssertionError("没有预测记录")
        keys = [(r.task_id, r.fold_id, r.segment, r.origin_id, r.step, r.candidate_id, r.seed)
                for r in self.records]
        if len(keys) != len(set(keys)):
            raise AssertionError("预测键重复")
        n_origins = len({r.origin_id for r in self.records})
        if n_origins != expected_origins:
            raise AssertionError(f"起点数 {n_origins} != 期望 {expected_origins}")
        steps = {r.step for r in self.records}
        if steps != set(range(1, horizon + 1)):
            raise AssertionError(f"步长集合 {sorted(steps)} != 1..{horizon}")
        if any(r.seed != seed for r in self.records):
            raise AssertionError("记录里混入了其它种子")
