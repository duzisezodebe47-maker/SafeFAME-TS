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
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEGMENTS = ("train", "calibration", "decision", "test")
ORIGIN_ID_RE = re.compile(r"^(?P<domain>[A-Za-z0-9_]+):h(?P<horizon>\d+):f(?P<fold>\d+):o(?P<index>\d+)$")
TASK_ID_RE = re.compile(r"^(?P<domain>[A-Za-z0-9_]+)_h(?P<horizon>\d+)_f(?P<fold>\d+)$")


class BundleUnavailable(RuntimeError):
    """Bundle 不存在或未冻结。**这是可定位的阻塞，不是可以用占位数据绕过的情况。**"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_signature(folder: Path, expected: str, split_spec_sha256: str | None = None) -> dict:
    """核对 bundle_signature，并复核 manifest 记录的哈希。

    第三轮 A.1 要求：核对 **全部且只有** 清单所列文件 —— 既要检出缺失/哈希不符，
    也要检出**清单外的多余文件**（多余文件意味着 Bundle 与签名描述的内容不一致）。

    `split_spec_sha256` 非空时，还需与 `manifest.inputs.split_spec_sha256` 一致，
    确保模型侧用的是主控冻结的那份划分协议。
    """
    folder = Path(folder)
    manifest_path = folder / "manifest.json"
    if not manifest_path.is_file():
        raise BundleUnavailable(f"找不到 Bundle 清单: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = manifest.get("signature")
    if actual != expected:
        raise BundleUnavailable(f"bundle_signature 不符: 期望 {expected}，实际 {actual}")

    listed = manifest.get("files")
    if not isinstance(listed, dict) or not listed:
        raise BundleUnavailable("manifest 缺少 files 清单，无法核对完整性")

    problems: list[str] = []
    for relative, recorded in listed.items():
        candidate = folder / relative
        if not candidate.is_file():
            problems.append(f"{relative}(缺失)")
        elif sha256_file(candidate) != recorded:
            problems.append(f"{relative}(哈希不符)")

    # 只在清单所列文件 —— 多余文件同样是签名不符
    on_disk = {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}
    extra = sorted(on_disk - set(listed) - {"manifest.json"})
    if extra:
        problems.append(f"清单外多余文件 {len(extra)} 个: {extra[:3]}")

    if problems:
        raise BundleUnavailable("Bundle 文件校验失败: " + "; ".join(problems[:5]))

    if split_spec_sha256 is not None:
        recorded_spec = (manifest.get("inputs") or {}).get("split_spec_sha256")
        if recorded_spec != split_spec_sha256:
            raise BundleUnavailable(
                f"split_spec_sha256 不符: 期望 {split_spec_sha256}，manifest 记录 {recorded_spec}"
                "（模型侧用的划分协议与 Bundle 的不一致）")
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

    # origin_id 必须与 task_id、origin_index 三方自洽（第三轮 A.2）
    task_match = TASK_ID_RE.match(bundle.task_id)
    if task_match is None:
        raise AssertionError(f"task_id 格式不符契约 Domain_hH_fF: {bundle.task_id}")
    want_domain = task_match.group("domain")
    want_horizon = int(task_match.group("horizon"))
    want_fold = int(task_match.group("fold"))

    for oid, idx in zip(origin_id, index):
        match = ORIGIN_ID_RE.match(str(oid))
        if match is None:
            raise AssertionError(f"origin_id 格式不符契约: {oid}")
        # 索引对但领域/跨度/折错 —— 同样必须拒绝
        if match.group("domain") != want_domain:
            raise AssertionError(f"origin_id 领域与 task_id 不符: {oid} vs {bundle.task_id}")
        if int(match.group("horizon")) != want_horizon:
            raise AssertionError(f"origin_id 跨度与 task_id 不符: {oid} vs {bundle.task_id}")
        if int(match.group("fold")) != want_fold:
            raise AssertionError(f"origin_id 折号与 task_id 不符: {oid} vs {bundle.task_id}")
        if int(match.group("index")) != int(idx):
            raise AssertionError(f"origin_id 与 origin_index 错位: {oid} vs {idx}")

    # 各数组行数必须完全一致 —— 不一致就硬失败，绝不静默内连接
    lengths = {k: len(v) for k, v in part.items() if hasattr(v, "__len__")}
    if len(set(lengths.values())) != 1:
        raise AssertionError(f"{bundle.task_id}/{segment}: 各字段行数不一致 {lengths}")

    if expected_count is not None and len(index) != expected_count:
        raise AssertionError(
            f"{bundle.task_id}/{segment}: 起点数 {len(index)} != 协议期望 {expected_count}")


def _data_side_dir() -> Path:
    # parents[2] = 仓库根（本文件在 <repo>/03 辅助电脑二 交付二次/model/ 下）。
    # 数据侧交付目录**不在本仓库内**（它在孤儿分支 3218151885-creator 上），
    # 因此除下列位置外，还允许通过 DATA_SIDE_DIR 环境变量指定。
    return Path(os.environ.get(
        "DATA_SIDE_DIR",
        Path(__file__).resolve().parents[2] / "辅助电脑02交付01次",
    ))


def _load_direct(bundle_folder: Path, task_id: str, scenario: str):
    """官方适配层：按 `prediction_contract_v2.json` 的 `data_bundle_required` 直接读文件。

    数据侧 `bundle.py` 在孤儿分支上、不一定随仓库分发，所以必须有这条不依赖它的路径。
    字段缺失一律硬失败 —— 不代填、不静默跳过。
    """
    folder = Path(bundle_folder)
    task_dir = folder / task_id
    scenario_dir = task_dir / scenario
    if not task_dir.is_dir():
        raise BundleUnavailable(f"找不到任务目录: {task_dir}")
    if not scenario_dir.is_dir():
        raise BundleUnavailable(f"找不到情景目录: {scenario_dir}")

    shared = ("numeric_history", "origin_index", "targets", "targets_standardized")
    per_scenario = ("semantic", "quality", "text_available")
    arrays: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for key in shared:
        path = task_dir / f"{key}.npy"
        if not path.is_file():
            missing.append(f"{task_id}/{key}.npy")
        else:
            arrays[key] = np.load(path, allow_pickle=False)
    for key in per_scenario:
        path = scenario_dir / f"{key}.npy"
        if not path.is_file():
            missing.append(f"{task_id}/{scenario}/{key}.npy")
        else:
            arrays[key] = np.load(path, allow_pickle=False)
    if missing:
        raise BundleUnavailable("Bundle 缺少契约要求的字段: " + ", ".join(missing))

    samples_path = task_dir / "samples.csv"
    if not samples_path.is_file():
        raise BundleUnavailable(f"找不到 {task_id}/samples.csv")
    samples = pd.read_csv(samples_path)
    for column in ("task_id", "fold_id", "origin_id", "origin_index", "segment"):
        if column not in samples.columns:
            raise BundleUnavailable(f"samples.csv 缺少契约列: {column}")

    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return samples, arrays, manifest


def _load_via_data_side(bundle_folder: Path, task_id: str, scenario: str, signature: str):
    """优先调用数据侧交付目录里的 `read_bundle`；不可用时回退到官方适配层。"""
    module_path = _data_side_dir() / "bundle.py"
    if not module_path.is_file():
        return _load_direct(bundle_folder, task_id, scenario)
    data_side = module_path.parent
    if str(data_side) not in sys.path:
        sys.path.insert(0, str(data_side))
    spec = importlib.util.spec_from_file_location("data_side_bundle", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_bundle(bundle_folder, task_id, scenario, signature)


def read_frozen_bundle(
    bundle_folder: Path, task_id: str, scenario: str, signature: str,
    split_spec_sha256: str | None = None,
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

    # 无论走数据侧 read_bundle 还是适配层直读，**同一套校验先跑**
    # （第三轮 A.1：不得只在一条路径上校验、另一条静默放行）
    verify_signature(folder, signature, split_spec_sha256=split_spec_sha256)
    samples, arrays, manifest = _load_via_data_side(folder, task_id, scenario, signature)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    return FrozenBundle(task_id=task_id, scenario=scenario, signature=signature,
                        samples=samples, arrays=arrays, manifest=manifest)


def expected_origin_id(domain: str, horizon: int, fold: int, origin_index: int) -> str:
    """按契约生成 origin_id，供导出前核对。"""
    return f"{domain}:h{horizon}:f{fold}:o{origin_index}"


def decision_halves_by_target_time(
    bundle: FrozenBundle, segment: str = "decision", time_column: str = "target_end_time",
) -> dict[str, np.ndarray]:
    """按**目标时间中点**把决策段切成两个不重叠半段（第三轮 A.5）。

    主控明确要求不得按起点个数对半切。做法：

    1. 取该段 `time_column` 的中位数作为分界；
    2. 前段 = 目标窗口**完全**落在分界之前，后段 = **完全**落在分界之后；
    3. 跨分界的 H 窗口**两边都不收**（剔除），保证两个半段的目标时间不重叠。

    返回 `{half_name: 布尔掩码}`，掩码索引对应 `bundle.segment(segment)` 的行序。
    时间列缺失时**硬失败** —— 不退回按个数切。
    """
    if time_column not in bundle.samples.columns:
        raise BundleUnavailable(
            f"samples.csv 缺少 {time_column}，无法按目标时间切半段。"
            f"任务书 A.5 禁止按起点个数对半切，故此处不提供回退。")

    mask = bundle.segment_mask(segment)
    times = np.asarray(bundle.samples.loc[mask, time_column].to_numpy())
    if times.size == 0:
        raise BundleUnavailable(f"{bundle.task_id}/{segment} 为空，无法切半段")

    ordered = np.sort(times)
    midpoint = ordered[len(ordered) // 2]

    # 逐起点取该窗口的起止，只有整窗落在某一侧才收入该半段
    start_column = time_column.replace("end", "start")
    if start_column in bundle.samples.columns:
        starts = np.asarray(bundle.samples.loc[mask, start_column].to_numpy())
    else:
        starts = times  # 无起始列时按单点时间处理

    first = starts < midpoint
    second = times > midpoint
    if not first.any() or not second.any():
        raise AssertionError(
            f"{bundle.task_id}/{segment}: 按 {time_column} 中点切分后有一半为空，"
            f"请核对协议的目标边界设置")
    return {"first_half": first, "second_half": second}
