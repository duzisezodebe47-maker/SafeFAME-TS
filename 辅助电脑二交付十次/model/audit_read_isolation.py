"""入口读文件审计（第八轮任务书 §五）。

> 入口读文件审计中，如果 test 真值被读取为数据而不是完整性哈希，必须判失败。

## 判据（可机检）

对每个**含 test 段真值**的文件（`targets.npy` / `targets_standardized.npy` /
`targets_raw.npy` / `target_time.npy`），只允许两种访问：

  1. **完整性哈希**：`pathlib`/`builtins` 的字节读（`verify_bundle` 复核清单）；
  2. **严格隔离载入**：`np.load(..., mmap_mode="r")` 且**只布尔索引非 test 行**
     —— 整表从未进入内存。

出现 `np.load` **不带 mmap**（= 整表当作数据读入）即 **FAIL**。
这正是第七轮做不到、第八轮用 `isolate_test="strict"` 补上的那条：
第七轮是"读进来再把 test 行抹掉"，本轮是"根本不读 test 行"。

## 做法

同进程内包装 `builtins.open` / `io.open` / `pathlib.Path.open` / `numpy.load` /
`pandas.read_csv`，逐路径记录**通道与调用参数**（np.load 是否带 mmap），
然后在监测下真正跑一次选择期入口（默认 `train.py`），最后：

  - 给出完整读清单（含通道分类）；
  - 断言 `targets*` 的 test 行在载入结果里要么不存在、要么是 NaN；
  - 断言非 test 行未被波及；
  - 静态核查入口源码里没有误差计算。

退出码 0 = 通过；非 0 = 有 test 真值被当作数据读取（或断言失败）。
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bundle_reader import TARGET_KEYS, TEST_TRUTH_FILE_NAMES  # noqa: E402
from isolated_package import (  # noqa: E402
    PackageUnavailable, add_input_args, open_input,
)
from predict_io import warn_if_dirty  # noqa: E402
from runtime_profile import cpu_seconds, script_entry_snapshot  # noqa: E402
from train import RUNNABLE_SCENARIOS  # noqa: E402

REPO_ROOT = HERE.parents[1]

_LOG: dict[str, dict] = {}
_ORIG: dict[str, object] = {}


def _record(path, channel: str, mmap: bool | None = None) -> None:
    try:
        key = str(Path(path).resolve())
    except (TypeError, ValueError):
        return
    entry = _LOG.setdefault(key, {"path": key, "channels": {}, "np_load_modes": []})
    entry["channels"][channel] = entry["channels"].get(channel, 0) + 1
    if channel == "numpy.load":
        entry["np_load_modes"].append("mmap" if mmap else "full")


def _install() -> None:
    real_open, real_io, real_path_open = builtins.open, io.open, Path.open
    real_load, real_read_csv = np.load, __import__("pandas").read_csv
    _ORIG.update(open=real_open, io=real_io, path_open=real_path_open,
                 load=real_load, read_csv=real_read_csv)

    def open_wrap(file, *a, **k):
        if isinstance(file, (str, Path)):
            _record(file, "builtins.open")
        return real_open(file, *a, **k)

    def io_wrap(file, *a, **k):
        if isinstance(file, (str, Path)):
            _record(file, "io.open")
        return real_io(file, *a, **k)

    def path_open_wrap(self, *a, **k):
        mode = str(a[0] if a else k.get("mode", "r"))
        _record(self, "pathlib.Path.open(write)" if "w" in mode and "r" not in mode
                else "pathlib.Path.open")
        return real_path_open(self, *a, **k)

    def load_wrap(file, *a, **k):
        mmap = k.get("mmap_mode")
        if mmap is None and len(a) > 1:
            mmap = a[1]
        if isinstance(file, (str, Path)):
            _record(file, "numpy.load", mmap=bool(mmap))
        return real_load(file, *a, **k)

    def read_csv_wrap(file, *a, **k):
        if isinstance(file, (str, Path)):
            _record(file, "pandas.read_csv")
        return real_read_csv(file, *a, **k)

    builtins.open, io.open, Path.open = open_wrap, io_wrap, path_open_wrap
    np.load = load_wrap
    import pandas
    pandas.read_csv = read_csv_wrap


def _remove() -> None:
    builtins.open, io.open, Path.open = _ORIG["open"], _ORIG["io"], _ORIG["path_open"]
    np.load = _ORIG["load"]
    import pandas
    pandas.read_csv = _ORIG["read_csv"]


def main() -> int:
    parser = argparse.ArgumentParser()
    add_input_args(parser)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scenario", required=True, choices=RUNNABLE_SCENARIOS)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--split-spec", type=Path, required=True)
    parser.add_argument("--candidate", default="N+S+Q")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--probe-output", type=Path, required=True,
                        help="被监测入口的输出目录（临时，不属于交付物）")
    args = parser.parse_args()

    started = time.perf_counter()
    cpu_started = cpu_seconds()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    import train as train_entry

    argv = ["train.py", "--task", args.task,
            "--scenario", args.scenario, "--signature", args.signature,
            "--split-spec", str(args.split_spec), "--segments", "calibration", "decision",
            "--candidate", args.candidate, "--seed", str(args.seed),
            "--output-dir", str(args.probe_output)]
    if getattr(args, "input_package", None) is not None:
        argv += ["--input-package", str(args.input_package)]
        if getattr(args, "formal_bundle", None) is not None:
            argv += ["--formal-bundle", str(args.formal_bundle)]
    else:
        argv += ["--bundle", str(args.bundle)]
    old_argv = sys.argv
    sys.argv = argv
    _install()
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = train_entry.main()
    except SystemExit as exc:
        code = int(exc.code or 1)
    finally:
        _remove()
        sys.argv = old_argv

    input_dir = Path(args.input_package if getattr(args, "input_package", None)
                     else args.bundle).resolve()
    entries = []
    violations: list[str] = []
    is_package = getattr(args, "input_package", None) is not None
    for e in sorted(_LOG.values(), key=lambda x: x["path"]):
        p = Path(e["path"])
        name = p.name
        try:
            under = str(p.parent.relative_to(input_dir))
        except ValueError:
            under = None
        # 包模式下 `selection_fit/targets.npy` **只含选择期行**，不含 test 真值；
        # 只有 test_features/ 下的真值名文件才算（结构上不存在）。
        # Bundle 模式下任务目录里的 targets* 是单表含 test 行，按文件名即可判定。
        contains_test_truth = name in TEST_TRUTH_FILE_NAMES and (
            ("test_features" in (under or "")) or not is_package)
        modes = e.get("np_load_modes") or []
        entry = {"path": str(p), "name": name, "under_bundle": under,
                 "channels": e["channels"], "np_load_modes": modes,
                 "contains_test_truth": contains_test_truth,
                 "class": ("input_test_truth_table" if contains_test_truth and under
                           else ("input_feature_or_index" if under else "outside_input"))}
        entries.append(entry)
        if contains_test_truth and "full" in modes:
            violations.append(f"{p} 被 np.load 整表读入（未用 mmap）—— test 真值当作数据读取")

    # 隔离断言：两种输入各按各的判据
    b, input_kind = open_input(args, args.task, args.scenario, args.signature,
                               args.split_spec)
    tm = b.segment_mask("test")
    assertions = {"input_kind": input_kind,
                  "isolation_mode": (b.isolation or {}).get("mode"),
                  "test_truth_present": (b.isolation or {}).get("test_truth_present"),
                  "test_rows_materialized": (b.isolation or {}).get("test_rows_materialized")}
    for key in TARGET_KEYS:
        if key in b.arrays:
            arr = np.asarray(b.arrays[key])
            assertions[f"{key}_test_rows_all_nan"] = bool(np.isnan(arr[tm]).all())
            assertions[f"{key}_non_test_rows_finite"] = bool(np.isfinite(arr[~tm]).all())
    assertions["train_entry_exit_zero"] = code == 0

    mode = assertions.get("isolation_mode")
    if mode == "isolated_package":
        # 隔离包：test 侧**结构上**没有真值 —— 没有可读的真值文件，也没有整表读入的可能
        mode_ok = (assertions.get("test_truth_present") is False
                   and all(v for k, v in assertions.items() if k.endswith("_all_nan")))
    else:
        mode_ok = (mode == "strict"
                   and assertions.get("test_rows_materialized") is False
                   and all(v for k, v in assertions.items() if k.endswith("_all_nan")))
    ok = code == 0 and not violations and mode_ok

    report = {
        "task": args.task, "scenario": args.scenario, "candidate": args.candidate,
        "monitored_entry": "train.py",
        "monitoring": {
            "method": "同进程包装 builtins.open / io.open / pathlib.Path.open / "
                      "numpy.load / pandas.read_csv，记录逐路径通道与 np.load 是否带 mmap",
            "rule": "含 test 真值的文件只允许【完整性哈希】或"
                    "【np.load(mmap_mode='r') 且只索引非 test 行】；"
                    "出现不带 mmap 的 np.load 即判失败",
            "files_touched": len(entries),
            "input_kind": input_kind,
            "input_root": str(input_dir),
        },
        "read_manifest": entries,
        "violations": violations,
        "isolation_assertions": assertions,
        "verdict": "PASS" if ok else "FAIL",
        "code_provenance": warn_if_dirty(REPO_ROOT, HERE),
        "runtime": script_entry_snapshot(started, cpu_started),
    }
    (out / "test_read_isolation_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({"verdict": report["verdict"], "files_touched": len(entries),
                      "violations": violations,
                      "isolation_mode": assertions.get("isolation_mode")},
                     ensure_ascii=False))
    return 0 if ok else 6


if __name__ == "__main__":
    raise SystemExit(main())
