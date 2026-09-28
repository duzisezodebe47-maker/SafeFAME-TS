"""Bundle 读取通道探针：区分「完整性哈希」与「当作数据载入」（第七轮补充证据）。

## 为什么需要它

第七轮任务书要求 `test_isolation_audit.json` 给出「监测方法与读文件清单」，
主控验收第 4 条要「审计读文件清单，确认模型侧未读取 test 真值」。

`test_isolation_audit.json` 已经给出了完整读清单，但其 `first_via` 一律是
`pathlib.Path.open`：因为 `verify_bundle` **先**对清单里每个文件做字节哈希，
于是"首次接触通道"永远是 pathlib，看不出某个文件到底是**只被哈希**还是
**被当作数据 np.load 了**。这正是主控要判的那件事。

本探针**只监测 Bundle 读取这一步**（`read_frozen_bundle`，与预测入口调用它的
参数完全一致），逐文件记录**各通道的调用次数**：

    pathlib.Path.open   读字节（清单完整性哈希、或本模块自己的 b"rb" 读取）
    numpy.load          当作数组载入
    pandas.read_csv     CSV 载入

因此它**不运行预测、不产生任何预测输出**，不影响
`replay_check.json` / `test_isolation_audit.json` 及其锚点。

输出 `bundle_read_channels.json`：逐文件通道计数 + 分组的「数据载入清单」
与「仅被哈希清单」，并对含 test 段真值的表给出说明。
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bundle_reader import read_frozen_bundle  # noqa: E402
from predict_io import warn_if_dirty  # noqa: E402
from runtime_profile import cpu_seconds, script_entry_snapshot  # noqa: E402
from train import RUNNABLE_SCENARIOS  # noqa: E402

REPO_ROOT = HERE.parents[1]

TRUTH_TABLE_NAMES = ("targets.npy", "targets_standardized.npy", "targets_raw.npy",
                     "target_time.npy")

LOG: dict[str, dict] = {}
_ORIG: dict[str, object] = {}


def _record(path, channel: str) -> None:
    try:
        key = str(Path(path).resolve())
    except (TypeError, ValueError):
        return
    entry = LOG.setdefault(key, {"path": key, "channels": {}, "total": 0})
    entry["channels"][channel] = entry["channels"].get(channel, 0) + 1
    entry["total"] += 1


def _install() -> None:
    import builtins
    real_open, real_path_open = builtins.open, Path.open
    real_load, real_read_csv = np.load, __import__("pandas").read_csv
    _ORIG.update(open=real_open, path_open=real_path_open, load=real_load,
                 read_csv=real_read_csv)

    def open_wrap(file, *a, **k):
        if isinstance(file, (str, Path)):
            _record(file, "builtins.open")
        return real_open(file, *a, **k)

    def path_open_wrap(self, *a, **k):
        mode = str(a[0] if a else k.get("mode", "r"))
        _record(self, "pathlib.Path.open(write)" if "w" in mode and "r" not in mode
                else "pathlib.Path.open")
        return real_path_open(self, *a, **k)

    def load_wrap(file, *a, **k):
        if isinstance(file, (str, Path)):
            _record(file, "numpy.load")
        return real_load(file, *a, **k)

    def read_csv_wrap(file, *a, **k):
        if isinstance(file, (str, Path)):
            _record(file, "pandas.read_csv")
        return real_read_csv(file, *a, **k)

    builtins.open = open_wrap
    Path.open = path_open_wrap
    np.load = load_wrap
    import pandas
    pandas.read_csv = read_csv_wrap


def _remove() -> None:
    import builtins
    builtins.open = _ORIG["open"]
    Path.open = _ORIG["path_open"]
    np.load = _ORIG["load"]
    import pandas
    pandas.read_csv = _ORIG["read_csv"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scenario", required=True, choices=RUNNABLE_SCENARIOS)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--split-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    cpu_started = cpu_seconds()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    _install()
    try:
        bundle = read_frozen_bundle(args.bundle, args.task, args.scenario,
                                    args.signature, args.split_spec, isolate_test=True)
    finally:
        _remove()

    entries = sorted(LOG.values(), key=lambda e: e["path"])
    for e in entries:
        e["name"] = Path(e["path"]).name
        e["under"] = str(Path(e["path"]).parent.relative_to(Path(args.bundle).resolve())) \
            if str(e["path"]).startswith(str(Path(args.bundle).resolve())) else None
        e["contains_test_truth"] = e["name"] in TRUTH_TABLE_NAMES

    data_loaded = [e for e in entries
                   if any(c in e["channels"] for c in ("numpy.load", "pandas.read_csv"))]
    hashed_only = [e for e in entries
                   if set(e["channels"]) <= {"pathlib.Path.open", "builtins.open"}]

    target_task = args.task
    target_scenario = args.scenario
    scoped = [e for e in data_loaded
              if e["under"] and e["under"].startswith(target_task)]
    scoped_truth = [e for e in scoped if e["contains_test_truth"]]

    report = {
        "task": args.task, "scenario": args.scenario,
        "purpose": "区分「完整性哈希」与「当作数据载入」的通道 —— 本探针不运行预测",
        "note_no_prediction": "只调用 read_frozen_bundle，与 predict_test.py 的调用参数"
                              "一致（isolate_test=True）；不产生任何预测输出。",
        "bundle_signature": bundle.signature,
        "files_touched": len(entries),
        "data_loaded_files": [
            {"name": e["name"], "under": e["under"], "channels": e["channels"],
             "contains_test_truth": e["contains_test_truth"]} for e in data_loaded],
        "hashed_only_count": len(hashed_only),
        "target_task_data_loads": [
            {"name": e["name"], "channels": e["channels"],
             "contains_test_truth": e["contains_test_truth"]} for e in scoped],
        "test_truth_tables_loaded_as_data": [
            {"name": e["name"], "channels": e["channels"],
             "why": "该 .npy 是**单表含全部段**（train/cal/dec/test 同行）。"
                    "清单完整性哈希与本任务的 train/cal/dec 拟合都必须打开它；"
                    "载入后 test 段行立即被置 NaN（isolate_test=True），"
                    "不进入模型 —— 见 test_isolation_audit.json 的断言与"
                    " tests/test_round7.py 的扰动实验。"} for e in scoped_truth],
        "full_channel_table": entries,
        "code_provenance": warn_if_dirty(REPO_ROOT, HERE),
        "runtime": script_entry_snapshot(started, cpu_started),
    }
    (out / "bundle_read_channels.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    print(json.dumps({"files_touched": len(entries),
                      "data_loaded": len(data_loaded),
                      "hashed_only": len(hashed_only),
                      "target_task_data_loads": [e["name"] for e in scoped],
                      "test_truth_loaded_as_data": [e["name"] for e in scoped_truth]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
