r"""Climate 选择期**隔离输入包**读取器（第八次交付）。

## 为什么有这条路径

第八次任务书 §五：**优先消费辅助电脑02提供的选择期隔离包**；若该包尚未交付，
可先实现代码与合成测试，正式证据必须在固定包上重跑。包已交付（Release 附件
`climate-isolated-input-v8-20260924`，ZIP SHA256 `b028052e…b314b`），故正式证据走这条路径。

## 包与正式 Bundle 的关系（本模块**独立复核**，不靠声称）

    selection_fit/   956 行 = train 581 + calibration 124 + decision 251
                     含建模特征与这三段的真值
    test_features/   124 行 = test；**只有** origin/网格/缺失掩码与文本特征
                     **没有 targets\*，也没有 numeric_history / frequency**

也就是说，"测试真值不参与选择期"不是"读进来再抹掉"，而是**包里根本没有**。
`selection_fit_commitments.json` 给出每张表的 `full_bundle_file_sha256` 与
`selection_slice_npy_sha256`；给到正式 Bundle 时本模块逐位重算比对
（切片 == Bundle 对应行、测试特征 == Bundle 测试行），把"包是 Bundle 的忠实切片"
这一条也变成可核对项。

## 与既有代码的接口

返回的仍是 `bundle_reader.FrozenBundle`（samples + arrays + signature + isolation），
所以 `train.py` / `permutation_entry.py` / `audit_boundaries.py` 一行预测逻辑都不用改。
test 段以 **NaN 占位**（占位值不是真值，且本模块断言其为 NaN），仅供边界审计
检查索引与特征结构 —— 第八次不生成任何 test 预测。
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bundle_reader import FrozenBundle, sha256_file  # noqa: E402

SELECTION_PARTS = ("train", "calibration", "decision")
TARGET_KEYS = ("targets", "targets_standardized", "targets_raw")
# test 包不提供、以 NaN 占位的键（占位值不是真值；断言其为 NaN）
TEST_ABSENT_KEYS = ("numeric_history", "numeric_history_raw", "frequency")
# 两侧都提供、需按行拼接的键
SCENARIO_KEYS = ("semantic", "quality", "text_available", "source_available")
SHARED_KEYS = ("origin_index", "numeric_missing")
# 包里**绝不允许**出现在 test_features 下的键：出现即说明隔离被破坏。
# 注意要与**文件名**比较，所以必须带 `.npy` 后缀（第一版漏了后缀，
# 这条结构性断言等于从没生效过 —— 是合成反例把它试出来的）。
FORBIDDEN_IN_TEST = tuple(f"{k}.npy" for k in
                          TARGET_KEYS + ("numeric_history", "numeric_history_raw",
                                         "frequency"))


class PackageUnavailable(RuntimeError):
    """隔离包缺失、损坏或与 Bundle/spec 锚不符。**不得降级到别的输入。**"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_package_files(package_dir: Path) -> dict:
    """按包的 `MANIFEST.json` 逐个文件复核字节哈希。缺文件或哈希不符即拒绝。"""
    manifest = _load_json(package_dir / "MANIFEST.json")
    missing, mismatch = [], []
    for entry in manifest.get("files", []):
        rel = entry["relative_path"]
        path = package_dir / rel
        if not path.is_file():
            missing.append(rel)
            continue
        if _sha256(path) != entry["sha256"]:
            mismatch.append(rel)
    if missing or mismatch:
        raise PackageUnavailable(
            f"隔离包文件不符：缺 {missing[:3]}… 哈希不符 {mismatch[:3]}…")
    return {"manifest": manifest, "files_checked": len(manifest.get("files", [])),
            "self_exclusion": manifest.get("self_exclusion")}


def verify_anchors(package_dir: Path, expected_signature: str, split_spec: Path) -> dict:
    """包声明的 Bundle 签名与冻结 spec 哈希必须与本轮固定输入一致；
    并核对包内 `anchors/formal_bundle_manifest.json` 确实是那份 Bundle 清单。"""
    manifest = _load_json(package_dir / "MANIFEST.json")
    spec_sha = sha256_file(split_spec)
    if manifest.get("bundle_signature") != expected_signature:
        raise PackageUnavailable(
            f"包声明的 Bundle 签名不符: {manifest.get('bundle_signature')} != {expected_signature}")
    if manifest.get("split_spec_sha256") != spec_sha:
        raise PackageUnavailable(
            f"包声明的 spec 哈希不符: {manifest.get('split_spec_sha256')} != {spec_sha}")
    anchor = package_dir / "anchors" / "formal_bundle_manifest.json"
    anchor_info = {}
    if anchor.is_file():
        anchor_json = _load_json(anchor)
        anchor_info = {
            "formal_bundle_manifest_sha256": _sha256(anchor),
            "anchor_signature": anchor_json.get("signature"),
        }
        if anchor_json.get("signature") != expected_signature:
            raise PackageUnavailable("包内 anchors/formal_bundle_manifest.json 的签名不符")
    return {"bundle_signature": expected_signature, "split_spec_sha256": spec_sha,
            **anchor_info}


def verify_commitments(package_dir: Path) -> dict:
    """按 `selection_fit_commitments.json` 复核每张选择期切片的形状/dtype/字节哈希。"""
    commitments = _load_json(package_dir / "selection_fit_commitments.json")
    checked, bad = 0, []
    for name, meta in commitments.items():
        path = package_dir / "selection_fit" / name
        if not path.is_file():
            bad.append(f"{name}: 缺文件")
            continue
        if _sha256(path) != meta["selection_slice_npy_sha256"]:
            bad.append(f"{name}: 切片哈希不符")
            continue
        arr = np.load(path, mmap_mode="r")
        if list(arr.shape) != list(meta["selection_shape"]) or str(arr.dtype) != meta["dtype"]:
            bad.append(f"{name}: 形状/dtype 不符")
            continue
        checked += 1
    if bad:
        raise PackageUnavailable(f"selection_fit 承诺校验失败: {bad[:3]}")
    return {"checked": checked, "keys": sorted(commitments)}


def assert_test_side_is_truth_free(package_dir: Path) -> dict:
    """**结构隔离断言**：test_features 下不得出现任何真值或可重建真值的数组。

    数据侧明确说明：逐起点数值历史与频率衍生特征被主动扣留（滚动窗口重叠足以
    重建约 123/127 个测试时间点的真值）。这里把"扣留"变成**断言**：
    出现 `targets*` / `numeric_history*` / `frequency` 即判失败。
    """
    test_dir = package_dir / "test_features"
    if not test_dir.is_dir():
        raise PackageUnavailable("隔离包缺少 test_features/")
    present = {p.name for p in test_dir.rglob("*.npy")}
    violated = sorted(n for n in present if n in FORBIDDEN_IN_TEST)
    if violated:
        raise PackageUnavailable(
            f"test_features 下出现真值/可重建真值的数组: {violated} —— 隔离被破坏")
    return {"test_files": sorted(present), "forbidden_present": [],
            "withheld": list(TEST_ABSENT_KEYS)}


def verify_against_formal_bundle(package_dir: Path, formal_bundle: Path,
                                 scenario: str, task_id: str) -> dict:
    """**独立可追溯性**：把包里的每张选择期切片与 test 特征逐位对照正式 Bundle。

    数据侧 README 说明"单靠严格隔离包不能重新证明切片来自原 Bundle"——
    但本侧**持有正式 Bundle**，所以这里把这条也验掉，而不是照信。
    """
    task_dir = Path(formal_bundle) / task_id
    scen_dir = task_dir / scenario
    samples = pd.read_csv(task_dir / "samples.csv")
    sel_mask = samples["segment"].isin(SELECTION_PARTS).to_numpy()
    test_mask = (samples["segment"] == "test").to_numpy()
    if samples["segment"].isin(SELECTION_PARTS).sum() == 0 or test_mask.sum() == 0:
        raise PackageUnavailable("正式 Bundle 的段划分异常，无法用于对照")

    mismatches: list[str] = []
    checked = 0

    def compare(pkg_path: Path, bundle_path: Path, mask, label: str) -> None:
        """对照时**只索引 mask 选中的行**，且 Bundle 侧一律 mmap。

        为什么不能用普通 `np.load`：Bundle 的 `targets*` 是**单表含 test 行**，
        普通 load 会把 test 真值整表读进内存 —— 第八轮读文件审计会（正确地）判失败。
        `mmap_mode="r"` + 布尔索引只会读选中行的页，test 行从不 materialize。
        """
        nonlocal checked
        if not pkg_path.is_file() or not bundle_path.is_file():
            return
        pkg_arr = np.load(pkg_path)
        view = np.load(bundle_path, mmap_mode="r")
        bun_arr = np.asarray(view[mask])
        del view
        if pkg_arr.shape != bun_arr.shape or not np.array_equal(pkg_arr, bun_arr):
            mismatches.append(label)
        checked += 1

    for name in ("numeric_history.npy", "numeric_history_raw.npy", "numeric_missing.npy",
                 "targets.npy", "targets_raw.npy", "targets_standardized.npy",
                 "frequency.npy", "origin_index.npy"):
        compare(package_dir / "selection_fit" / name, task_dir / name, sel_mask,
                f"selection_fit/{name}")
    for name in SCENARIO_KEYS:
        compare(package_dir / "selection_fit" / scenario / f"{name}.npy",
                scen_dir / f"{name}.npy", sel_mask, f"selection_fit/{scenario}/{name}.npy")
        compare(package_dir / "test_features" / scenario / f"{name}.npy",
                scen_dir / f"{name}.npy", test_mask, f"test_features/{scenario}/{name}.npy")
    compare(package_dir / "test_features" / "numeric_missing.npy",
            task_dir / "numeric_missing.npy", test_mask, "test_features/numeric_missing.npy")
    compare(package_dir / "test_features" / "origin_index.npy",
            task_dir / "origin_index.npy", test_mask, "test_features/origin_index.npy")

    # 行序也要对：包内 origin_id 顺序必须等于 Bundle 对应段的顺序
    meta = pd.read_csv(package_dir / "selection_fit" / "metadata.csv")
    if meta["origin_id"].tolist() != samples[sel_mask]["origin_id"].tolist():
        mismatches.append("selection_fit/metadata.csv 行序")
    tmeta = pd.read_csv(package_dir / "test_features" / "metadata.csv")
    if tmeta["origin_id"].tolist() != samples[test_mask]["origin_id"].tolist():
        mismatches.append("test_features/metadata.csv 行序")

    if mismatches:
        raise PackageUnavailable(f"隔离包与正式 Bundle 逐位对照不符: {mismatches}")
    return {"compared_arrays": checked, "rows_selection": int(sel_mask.sum()),
            "rows_test": int(test_mask.sum()), "bitwise_identical": True}


def load_isolated_package(
    package_dir: Path, task_id: str, scenario: str, split_spec: Path,
    *, expected_signature: str, formal_bundle: Path | None = None,
) -> FrozenBundle:
    """读取隔离包并组装成 `FrozenBundle`（test 段真值以 NaN 占位，从不来自数据）。"""
    package_dir = Path(package_dir)
    if not (package_dir / "MANIFEST.json").is_file():
        raise PackageUnavailable(f"找不到隔离包清单: {package_dir / 'MANIFEST.json'}")

    integrity = verify_package_files(package_dir)
    anchors = verify_anchors(package_dir, expected_signature, split_spec)
    commitments = verify_commitments(package_dir)
    truth_free = assert_test_side_is_truth_free(package_dir)
    traceability = None
    if formal_bundle is not None:
        traceability = verify_against_formal_bundle(package_dir, Path(formal_bundle),
                                                   scenario, task_id)

    sel_meta = pd.read_csv(package_dir / "selection_fit" / "metadata.csv")
    test_meta = pd.read_csv(package_dir / "test_features" / "metadata.csv")
    for frame, label in ((sel_meta, "selection_fit"), (test_meta, "test_features")):
        for col in ("origin_id", "origin_index", "segment"):
            if col not in frame.columns:
                raise PackageUnavailable(f"{label}/metadata.csv 缺列 {col}")
    if set(sel_meta["segment"]) - set(SELECTION_PARTS):
        raise PackageUnavailable(
            f"selection_fit 含非选择期段: {sorted(set(sel_meta['segment']) - set(SELECTION_PARTS))}")
    if set(test_meta["segment"]) != {"test"}:
        raise PackageUnavailable("test_features 只能含 test 段")

    n_sel, n_test = len(sel_meta), len(test_meta)
    samples = pd.concat(
        [sel_meta[["origin_id", "origin_index", "segment"]],
         test_meta[["origin_id", "origin_index", "segment"]]],
        ignore_index=True)

    arrays: dict[str, np.ndarray] = {}
    shrunk: list[str] = []

    def stack(sel_path: Path, test_path: Path | None, key: str,
              *, placeholder: bool) -> None:
        if not Path(sel_path).is_file():
            # 该键不在本包的选择期部分里（不同任务的数组集不同）——跳过，不代填
            return
        sel_arr = np.load(sel_path)
        if test_path is not None and Path(test_path).is_file():
            test_arr = np.load(test_path)
        elif placeholder:
            test_arr = np.full((n_test, *sel_arr.shape[1:]), np.nan, dtype=sel_arr.dtype)
            shrunk.append(key)
        else:
            return
        if sel_arr.shape[0] != n_sel or test_arr.shape[0] != n_test:
            raise PackageUnavailable(f"{key}: 行数与 metadata 不符")
        arrays[key] = np.concatenate([sel_arr, test_arr], axis=0)

    sel_dir = package_dir / "selection_fit"
    test_dir = package_dir / "test_features"

    for key in ("numeric_history", "numeric_history_raw", "numeric_missing",
                "targets", "targets_raw", "targets_standardized", "frequency",
                "origin_index"):
        stack(sel_dir / f"{key}.npy", test_dir / f"{key}.npy", key,
              placeholder=key not in SHARED_KEYS)
    for key in SCENARIO_KEYS:
        stack(sel_dir / scenario / f"{key}.npy", test_dir / scenario / f"{key}.npy",
              key, placeholder=True)

    # test 段真值必须是 NaN（占位值由本模块生成，不是任何来源的真值）
    test_rows = (samples["segment"] == "test").to_numpy()
    for key in TARGET_KEYS:
        if key in arrays and not np.isnan(arrays[key][test_rows]).all():
            raise PackageUnavailable(f"{key} 的 test 行不是占位 NaN —— 拒绝")
    for key in TEST_ABSENT_KEYS:
        if key in arrays:
            present = ~np.isnan(arrays[key][test_rows])
            if present.any():
                raise PackageUnavailable(f"{key} 的 test 行含非 NaN 值 —— 拒绝")

    isolation = {
        "mode": "isolated_package",
        "package_dir_name": package_dir.name,
        "test_truth_present": False,
        "test_truth_mechanism": "包内不含 test 真值（结构隔离），test 行以 NaN 占位",
        "test_absent_keys_placeholder_nan": sorted(shrunk),
        "integrity": integrity,
        "anchors": anchors,
        "commitments": commitments,
        "test_side": truth_free,
        "traceability_vs_formal_bundle": traceability,
    }
    return FrozenBundle(task_id=task_id, scenario=scenario, signature=expected_signature,
                        samples=samples, arrays=arrays,
                        manifest=integrity["manifest"], isolation=isolation)


# --------------------------------------------------------------------------
# 三个入口共用的输入选择：--bundle（正式 Bundle）或 --input-package（隔离包）
# --------------------------------------------------------------------------

def add_input_args(parser) -> None:
    """给入口加互斥的两个输入：正式 Bundle 或选择期隔离包。

    §五 优先隔离包；`--formal-bundle` 只在用隔离包时可选，用于把
    "包是 Bundle 的忠实切片"逐位复核（不给也照跑，只是少一层可追溯性）。
    """
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bundle", type=Path, help="主控冻结 Bundle 目录")
    group.add_argument("--input-package", type=Path,
                       help="辅助电脑02 的选择期隔离包目录（第八轮优先）")
    parser.add_argument("--formal-bundle", type=Path, default=None,
                        help="配合 --input-package：逐位复核切片是否来自该 Bundle")


def open_input(args, task_id: str, scenario: str, signature: str, split_spec: Path):
    """按参数打开输入；返回 `(bundle, input_kind)`。"""
    if getattr(args, "input_package", None) is not None:
        bundle = load_isolated_package(
            args.input_package, task_id, scenario, split_spec,
            expected_signature=signature,
            formal_bundle=getattr(args, "formal_bundle", None))
        # 隔离包是**选择期**输入：没有 test 特征，任何测试预测都必须被拒
        bundle.isolation["test_prediction_supported"] = False
        bundle.isolation["test_prediction_note"] = (
            "包内不含 test 数值历史/频率特征，结构上无法（也不允许）生成 test 预测")
        return bundle, "isolated_package"
    from bundle_reader import read_frozen_bundle
    bundle = read_frozen_bundle(args.bundle, task_id, scenario, signature, split_spec,
                                isolate_test="strict")
    return bundle, "formal_bundle"
