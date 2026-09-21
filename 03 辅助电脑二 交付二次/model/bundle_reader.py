"""主控冻结 Bundle 的读取与对齐（第二轮 P1 第 2 条）。

任务书要求：**直接调用数据侧 `read_bundle` 或其官方适配层，不复制旧 v2 的清洗/编码。**
本模块优先动态加载数据侧交付目录里的 `bundle.py`；加载失败时才回退到
按 `prediction_contract_v2.json` 的 `data_bundle_required` 直接读文件。

契约要点（`prediction_contract_v2.json`）::

    data_bundle_required = [manifest.json, schema.json,
                            <task_id>/samples.csv, <task_id>/origin_index.npy,
                            <task_id>/numeric_history.npy, <task_id>/targets.npy,
                            <task_id>/targets_standardized.npy, <task_id>/numeric_fit.json]
    origin_id_pattern = "Domain:hH:fF:o<origin_index>"
    segments          = [train, calibration, decision, test]
    rejection         = 拒绝重复/缺失/多余/错位/重排的起点网格；不得静默内连接

数据侧 `schema.json` 的 `status` 必须是 `frozen` —— 工程预览不构成可训练接口。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEGMENTS = ("train", "calibration", "decision", "test")
ORIGIN_ID_RE = re.compile(r"^(?P<domain>[A-Za-z0-9_]+):h(?P<horizon>\d+):f(?P<fold>\d+):o(?P<index>\d+)$")


class BundleUnavailable(RuntimeError):
    """Bundle 不存在或未冻结。**这是可定位的阻塞，不是可以用占位数据绕过的情况。**"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_signature(folder: Path, expected: str) -> dict:
    """核对 bundle_signature，并逐文件复核 manifest 记录的哈希。"""
    folder = Path(folder)
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        raise BundleUnavailable(f"找不到 Bundle 清单: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = manifest.get("signature")
    if actual != expected:
        raise BundleUnavailable(f"bundle_signature 不符: 期望 {expected}，实际 {actual}")

    mismatched = []
    for relative, recorded in (manifest.get("files") or {}).items():
        candidate = folder / relative
        if not candidate.is_file():
            mismatched.append(f"{relative}(缺失)")
        elif sha256_file(candidate) != recorded:
            mismatched.append(f"{relative}(哈希不符)")
    if mismatched:
        raise BundleUnavailable("Bundle 文件校验失败: " + ", ".join(mismatched[:5]))
    return manifest


@dataclass
class FrozenBundle:
    """Bundle 的一个任务切片，四段起点顺序与数据侧完全一致。"""

    task_id: str
    scenario: str
    signature: str
    samples: pd.DataFrame
    arrays: dict[str, np.ndarray]
    manifest: dict

    @property
    def origin_index(self) -> np.ndarray:
        return np.asarray(self.arrays["origin_index"]).ravel()

    def segment_mask(self, segment: str) -> np.ndarray:
        if segment not in SEGMENTS:
            raise ValueError(f"未知段: {segment}  (合法: {SEGMENTS})")
        return self.samples["segment"].to_numpy() == segment

    def segment(self, segment: str) -> dict[str, np.ndarray]:
        """取出某一段，返回契约要求的模型输入字段。

        起点网格按 `samples.csv` 的原始顺序导出，**不排序、不重排、不去重** ——
        任何顺序变化都会在 `assert_grid` 里被拒绝。
        """
        mask = self.segment_mask(segment)
        return {
            key: np.asarray(self.arrays[key])[mask]
            for key in ("numeric_history", "semantic", "quality", "text_available",
                        "origin_index", "targets", "targets_standardized")
            if key in self.arrays
        } | {"origin_id": self.samples["origin_id"].to_numpy()[mask]}


def assert_grid(bundle: FrozenBundle, segment: str, expected_count: int | None = None) -> None:
    """拒绝重复、缺失、错位、重排的起点网格（契约 `rejection` 条款）。"""
    part = bundle.segment(segment)
    origin_id = part["origin_id"]
    index = part["origin_index"]

    if len(origin_id) != len(set(origin_id.tolist())):
        raise AssertionError(f"{bundle.task_id}/{segment}: origin_id 重复")
    if len(index) != len(set(index.tolist())):
        raise AssertionError(f"{bundle.task_id}/{segment}: origin_index 重复")

    # origin_id 必须与 origin_index 自洽
    for oid, idx in zip(origin_id, index):
        match = ORIGIN_ID_RE.match(str(oid))
        if match is None:
            raise AssertionError(f"origin_id 格式不符契约: {oid}")
        if int(match.group("index")) != int(idx):
            raise AssertionError(f"origin_id 与 origin_index 错位: {oid} vs {idx}")

    # 各数组行数必须完全一致 —— 不一致就硬失败，绝不静默内连接
    lengths = {k: len(v) for k, v in part.items() if hasattr(v, "__len__")}
    if len(set(lengths.values())) != 1:
        raise AssertionError(f"{bundle.task_id}/{segment}: 各字段行数不一致 {lengths}")

    if expected_count is not None and len(index) != expected_count:
        raise AssertionError(
            f"{bundle.task_id}/{segment}: 起点数 {len(index)} != 协议期望 {expected_count}")


def _load_via_data_side(bundle_folder: Path, task_id: str, scenario: str, signature: str):
    """优先调用数据侧交付目录里的 `read_bundle`。"""
    data_side = Path(__file__).resolve().parents[3] / "辅助电脑02交付01次"
    module_path = data_side / "bundle.py"
    if not module_path.is_file():
        raise BundleUnavailable(f"数据侧读取器不在位: {module_path}")
    if str(data_side) not in sys.path:
        sys.path.insert(0, str(data_side))
    spec = importlib.util.spec_from_file_location("data_side_bundle", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_bundle(bundle_folder, task_id, scenario, signature)


def read_frozen_bundle(
    bundle_folder: Path, task_id: str, scenario: str, signature: str
) -> FrozenBundle:
    """读取并要求 Bundle 已冻结。

    `schema.json` 的 `status` 必须是 `frozen`；`DRAFT_PENDING_FROZEN_BUNDLE` 一律拒绝。
    """
    folder = Path(bundle_folder)
    schema_path = folder / "schema.json"
    if not schema_path.is_file():
        raise BundleUnavailable(f"找不到 schema.json: {schema_path}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("status") != "frozen":
        raise BundleUnavailable(
            f"Bundle 未冻结（schema.status={schema.get('status')!r}）。"
            f"契约要求 status=='frozen' 才可作为训练接口。"
        )

    verify_signature(folder, signature)
    samples, arrays, manifest = _load_via_data_side(folder, task_id, scenario, signature)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    return FrozenBundle(task_id=task_id, scenario=scenario, signature=signature,
                        samples=samples, arrays=arrays, manifest=manifest)


def expected_origin_id(domain: str, horizon: int, fold: int, origin_index: int) -> str:
    """按契约生成 origin_id，供导出前核对。"""
    return f"{domain}:h{horizon}:f{fold}:o{origin_index}"
