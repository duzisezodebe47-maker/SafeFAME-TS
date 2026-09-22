"""主控冻结 Bundle 的读取与校验（第四轮返修版）。

## 相对第三轮的修复

**P1-3：清单输入锚可伪造。**第三轮只比对 `manifest["signature"]` 与外部传入的期望值，
没有**独立重算**签名。主控 `team_eval/v2.verify_bundle` 的规则是::

    signature(manifest["inputs"]) == manifest["signature"] == expected_signature

本模块改用**完全相同的** `signature()` 实现（`sort_keys=True, ensure_ascii=False,
allow_nan=False` 的 JSON 序列化后取 SHA256），并把冻结 spec 的**实际文件字节哈希**、
嵌入的 spec 对象、`inputs.mode` 一并强制核对。

**P1-4：边界参数未在主入口传入。**`assert_grid` 现要求传入该段的半开边界与 H 窗口约束，
正式入口从冻结 spec 读取并传入。

**P1-5：半段切法与主控不一致。**第三轮按已保留起点的时间中位数切；主控门控按
`middle = (cal_end + dec_end) // 2` 的**行索引**中点切，前段仅 `origin + h <= middle`、
后段仅 `origin >= middle`，跨界目标窗口剔除。本模块改用主控规则。

**分段坐标**（主控 `bounds` 语义，`v2.verify_bundle` 已校验其严格递增）::

    bounds = [train_end, cal_end, dec_end, test_end]
    train       = [0,            bounds[0])
    calibration = [bounds[0],    bounds[1])
    decision    = [bounds[1],    bounds[2])
    test        = [bounds[2],    bounds[3]]
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
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class BundleUnavailable(RuntimeError):
    """Bundle 不存在、未冻结或校验不通过。**不得用占位数据绕过。**"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def signature(value: object) -> str:
    """与主控 `team_eval/v2.signature` **逐字一致**的签名算法。

    任何参数差异都会让两边算出不同的值，所以这里刻意不做"改进"。
    """
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BundleUnavailable(f"JSON 无效: {path}") from exc
    if not isinstance(value, dict):
        raise BundleUnavailable(f"期望 JSON 对象: {path}")
    return value


def verify_bundle(folder: Path, expected_signature: str, spec_path: Path) -> dict:
    """与主控 `v2.verify_bundle` 同一规则：数据侧封印 + 主控冻结协议 + 每个记录的字节。

    与主控实现的差异只有一处：主控对不合规项抛 `EvidenceError`，本函数抛
    `BundleUnavailable`；校验顺序与判据完全相同。
    """
    folder = Path(folder)
    spec = _read_json(Path(spec_path))
    if spec.get("status") != "frozen" or not spec.get("approved_by"):
        raise BundleUnavailable("split spec 仍是草案或未批准；不接受正式 Bundle")

    manifest = _read_json(folder / "manifest.json")
    recorded = manifest.get("signature")
    # 独立重算：签名必须等于 signature(inputs)，而不是只信外部传入的字符串
    if recorded != expected_signature or signature(manifest.get("inputs")) != expected_signature:
        raise BundleUnavailable(
            f"Bundle 签名不符：manifest.signature={recorded}，"
            f"signature(inputs)={signature(manifest.get('inputs'))}，期望={expected_signature}")

    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise BundleUnavailable("manifest.inputs 缺失或不是对象")
    spec_digest = sha256_file(spec_path)
    if inputs.get("mode") != "frozen":
        raise BundleUnavailable(f"Bundle 非冻结模式（inputs.mode={inputs.get('mode')!r}）")
    if inputs.get("split_spec_sha256") != spec_digest:
        raise BundleUnavailable(
            f"Bundle 由另一份 split spec 构建：inputs.split_spec_sha256="
            f"{inputs.get('split_spec_sha256')}，本次冻结 spec 的字节哈希={spec_digest}")
    if inputs.get("split_spec") != spec:
        raise BundleUnavailable("Bundle 内嵌的 split spec 与主控冻结文件不同")

    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise BundleUnavailable("Bundle 缺少文件清单")
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob("*")
              if p.is_file() and p.name != "manifest.json"}
    if actual != set(files):
        missing = sorted(set(files) - actual)
        extra = sorted(actual - set(files))
        raise BundleUnavailable(f"Bundle 清单不符: 缺失={missing[:3]} 多余={extra[:3]}")

    for relative, expected_hash in files.items():
        candidate = (folder / relative).resolve()
        if not candidate.is_relative_to(folder.resolve()):
            raise BundleUnavailable(f"清单路径越界: {relative}")
        if not HEX64.match(str(expected_hash)):
            raise BundleUnavailable(f"清单哈希格式非法: {relative}")
        if not candidate.is_file():
            raise BundleUnavailable(f"清单文件缺失: {relative}")
        if sha256_file(candidate) != expected_hash:
            raise BundleUnavailable(f"文件哈希不符: {relative}")
    return manifest


def segment_bounds(spec_path: Path, task_id: str) -> dict[str, tuple[int, int]]:
    """从冻结 spec 取四段的半开边界。

    `bounds = [train_end, cal_end, dec_end, test_end]` →
    train `[0, t)`、calibration `[t, c)`、decision `[c, d)`、test `[d, e]`。
    """
    spec = _read_json(Path(spec_path))
    match = TASK_ID_RE.match(task_id)
    if match is None:
        raise BundleUnavailable(f"task_id 格式不符: {task_id}")
    tasks = [t for t in (spec.get("tasks") or [])
             if f"{t.get('domain')}_h{t.get('horizon')}_f{t.get('fold_id')}" == task_id]
    if len(tasks) != 1:
        raise BundleUnavailable(f"冻结 spec 中 {task_id} 不是唯一登记项（{len(tasks)} 个）")
    bounds = [int(x) for x in tasks[0]["bounds"]]
    if not (0 < bounds[0] < bounds[1] < bounds[2] < bounds[3]):
        raise BundleUnavailable(f"冻结 spec 的 bounds 非严格递增: {bounds}")
    return {
        "train": (0, bounds[0]),
        "calibration": (bounds[0], bounds[1]),
        "decision": (bounds[1], bounds[2]),
        "test": (bounds[2], bounds[3]),
    }


@dataclass
class FrozenBundle:
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
        """取出某一段。起点网格按 `samples.csv` 原始顺序导出，**不排序、不重排、不去重**。"""
        mask = self.segment_mask(segment)
        return {key: np.asarray(self.arrays[key])[mask]
                for key in ("numeric_history", "semantic", "quality", "text_available",
                            "origin_index", "targets", "targets_standardized")
                if key in self.arrays} | {"origin_id": self.samples["origin_id"].to_numpy()[mask]}


def assert_grid(
    bundle: FrozenBundle, segment: str, bounds: tuple[int, int],
    horizon: int, expected_count: int | None = None,
) -> dict:
    """契约级网格校验：领域/H/折/索引/重复/边界/H 窗口。

    `bounds` 与 `horizon` **必填** —— 第三轮的教训是"边界测试通过"并不代表
    "正式入口执行了边界检查"，所以这里不给默认值。

    返回统计字典（起点数、各段边界、跨界窗口数），供审计表记录。
    """
    part = bundle.segment(segment)
    origin_id = part["origin_id"]
    index = np.asarray(part["origin_index"], dtype=np.int64)
    task_match = TASK_ID_RE.match(bundle.task_id)
    if task_match is None:
        raise AssertionError(f"task_id 格式不符契约 Domain_hH_fF: {bundle.task_id}")
    want_domain = task_match.group("domain")
    want_h = int(task_match.group("horizon"))
    want_fold = int(task_match.group("fold"))
    if want_h != int(horizon):
        raise AssertionError(f"传入的 horizon {horizon} 与 task_id {bundle.task_id} 不符")

    if len(origin_id) != len(set(map(str, origin_id))):
        raise AssertionError(f"{bundle.task_id}/{segment}: origin_id 重复")
    if len(index) != len(set(index.tolist())):
        raise AssertionError(f"{bundle.task_id}/{segment}: origin_index 重复")

    for oid, idx in zip(origin_id, index):
        m = ORIGIN_ID_RE.match(str(oid))
        if m is None:
            raise AssertionError(f"origin_id 格式不符契约: {oid}")
        if m.group("domain") != want_domain:
            raise AssertionError(f"origin_id 领域与 task_id 不符: {oid} vs {bundle.task_id}")
        if int(m.group("horizon")) != want_h:
            raise AssertionError(f"origin_id 跨度与 task_id 不符: {oid} vs {bundle.task_id}")
        if int(m.group("fold")) != want_fold:
            raise AssertionError(f"origin_id 折号与 task_id 不符: {oid} vs {bundle.task_id}")
        if int(m.group("index")) != int(idx):
            raise AssertionError(f"origin_id 与 origin_index 错位: {oid} vs {idx}")

    lengths = {k: len(v) for k, v in part.items() if hasattr(v, "__len__")}
    if len(set(lengths.values())) != 1:
        raise AssertionError(f"{bundle.task_id}/{segment}: 各字段行数不一致 {lengths}")

    lo, hi = int(bounds[0]), int(bounds[1])
    outside = index[(index < lo) | (index >= hi)]
    if outside.size:
        raise AssertionError(
            f"{bundle.task_id}/{segment}: {outside.size} 个起点落在该段边界 "
            f"[{lo}, {hi}) 之外，例如 {outside[:3].tolist()}")

    # H 窗口约束：目标窗口必须完整落在该段内（主控 target_boundary = disjoint_half_open）
    spanning = index[index + int(horizon) > hi]
    if spanning.size:
        raise AssertionError(
            f"{bundle.task_id}/{segment}: {spanning.size} 个起点的目标窗口跨出本段 "
            f"（origin + h > {hi}），例如 {spanning[:3].tolist()}")

    if expected_count is not None and len(index) != expected_count:
        raise AssertionError(
            f"{bundle.task_id}/{segment}: 起点数 {len(index)} != 期望 {expected_count}")

    return {"segment": segment, "n_origins": int(len(index)),
            "bounds": [lo, hi], "horizon": int(horizon),
            "index_min": int(index.min()) if index.size else None,
            "index_max": int(index.max()) if index.size else None}


def decision_halves(bundle: FrozenBundle, bounds: dict[str, tuple[int, int]], horizon: int) -> dict:
    """决策段两个不重叠半段，规则与主控 `team_eval/core.py` 一致。

    `middle = (cal_end + dec_end) // 2`
    前段：`origin + h <= middle`
    后段：`origin >= middle`
    跨界目标窗口两边都不收。

    **不按保留行的中位时间自行改变分界** —— 第三轮那么做，有样本剔除时会与主控不一致。
    返回 `{half_name: {"mask": 布尔数组, "kept": n, "excluded": n}}`。
    """
    cal_end = int(bounds["calibration"][1])
    dec_end = int(bounds["decision"][1])
    middle = (cal_end + dec_end) // 2

    part = bundle.segment("decision")
    origins = np.asarray(part["origin_index"], dtype=np.int64)
    first = origins + int(horizon) <= middle
    second = origins >= middle
    overlap = first & second
    if overlap.any():
        raise AssertionError(
            f"{bundle.task_id}: 半段重叠 {int(overlap.sum())} 个 —— 说明 h 或 middle 取值有误")

    return {
        "first_half": {"mask": first, "kept": int(first.sum()),
                       "excluded": int(len(origins) - first.sum())},
        "second_half": {"mask": second, "kept": int(second.sum()),
                        "excluded": int(len(origins) - second.sum())},
        "middle": middle, "cal_end": cal_end, "dec_end": dec_end,
        "rule": "middle=(cal_end+dec_end)//2; first: origin+h<=middle; second: origin>=middle",
    }


# ---------- 读取 ----------

def _data_side_dir() -> Path:
    # parents[2] = 仓库根。数据侧交付不在仓库内（在孤儿分支上），可用 DATA_SIDE_DIR 指定。
    return Path(os.environ.get(
        "DATA_SIDE_DIR",
        Path(__file__).resolve().parents[2] / "辅助电脑02交付01次"))


def _load_direct(bundle_folder: Path, task_id: str, scenario: str):
    """官方适配层：按 `data_bundle_required` 直读，不依赖数据侧 `bundle.py`。

    数据侧 `bundle.py` 在孤儿分支上、不一定随仓库分发，所以必须有这条路径。
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
        (arrays.__setitem__(key, np.load(path, allow_pickle=False))
         if path.is_file() else missing.append(f"{task_id}/{key}.npy"))
    for key in per_scenario:
        path = scenario_dir / f"{key}.npy"
        (arrays.__setitem__(key, np.load(path, allow_pickle=False))
         if path.is_file() else missing.append(f"{task_id}/{scenario}/{key}.npy"))
    if missing:
        raise BundleUnavailable("Bundle 缺少契约要求的字段: " + ", ".join(missing))

    samples_path = task_dir / "samples.csv"
    if not samples_path.is_file():
        raise BundleUnavailable(f"找不到 {task_id}/samples.csv")
    samples = pd.read_csv(samples_path)
    for column in ("task_id", "fold_id", "origin_id", "origin_index", "segment"):
        if column not in samples.columns:
            raise BundleUnavailable(f"samples.csv 缺少契约列: {column}")
    manifest = _read_json(folder / "manifest.json")
    return samples, arrays, manifest


def _load_via_data_side(bundle_folder: Path, task_id: str, scenario: str, signature_value: str):
    module_path = _data_side_dir() / "bundle.py"
    if not module_path.is_file():
        print(f"NOTE: 数据侧读取器不在位（{module_path}），改用官方适配层直读；"
              f"签名与逐文件哈希校验由 verify_bundle 统一执行。", file=sys.stderr)
        return _load_direct(bundle_folder, task_id, scenario)
    data_side = module_path.parent
    if str(data_side) not in sys.path:
        sys.path.insert(0, str(data_side))
    spec = importlib.util.spec_from_file_location("data_side_bundle", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_bundle(bundle_folder, task_id, scenario, signature_value)


def read_frozen_bundle(
    bundle_folder: Path, task_id: str, scenario: str, expected_signature: str,
    spec_path: Path,
) -> FrozenBundle:
    """读取并要求 Bundle 已冻结。

    `spec_path` **必填** —— 冻结协议的**文件路径**，本函数自己算它的字节哈希，
    不接受调用方传一个"声称的"哈希字符串（那样无法防伪造）。
    """
    folder = Path(bundle_folder)
    spec_path = Path(spec_path)
    if not spec_path.is_file():
        raise BundleUnavailable(f"找不到冻结协议文件: {spec_path}")

    schema_path = folder / "schema.json"
    if not schema_path.is_file():
        raise BundleUnavailable(f"找不到 schema.json: {schema_path}")
    schema = _read_json(schema_path)
    if schema.get("status") != "frozen":
        raise BundleUnavailable(
            f"Bundle 未冻结（schema.status={schema.get('status')!r}）；"
            f"契约要求 status=='frozen' 才可作为训练接口")

    manifest = verify_bundle(folder, expected_signature, spec_path)
    samples, arrays, _ = _load_via_data_side(folder, task_id, scenario, expected_signature)
    arrays = {k: np.asarray(v) for k, v in arrays.items()}
    return FrozenBundle(task_id=task_id, scenario=scenario, signature=expected_signature,
                        samples=samples, arrays=arrays, manifest=manifest)


def expected_origin_id(domain: str, horizon: int, fold: int, origin_index: int) -> str:
    return f"{domain}:h{horizon}:f{fold}:o{origin_index}"
