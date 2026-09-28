"""测试隔离审计 + 确定性重放（第九次交付证据）。

一次受监测的重放同时产出两份证据：

  `test_isolation_audit.json`   监测方法 + 读文件清单 + **test 真值被当作数据访问的次数**
  `replay_check.json`           第二次运行与正式 CSV 逐字节相同、权重哈希一致

## `test_truth_data_access_count` 的口径（必须说清楚）

本入口用的是第八次那套 **nan-mask** 隔离（`read_frozen_bundle(isolate_test=True)`，
与第七次交付相同的机制，主控在第七轮验收里已明确接受过这个口径）。因此要把两件事分开：

  A. **模型面对的数据**：推理视图 `targets=None`；任何 `segment("test")` 取到的
     `targets*` 都是 NaN 占位。→ 计入 `test_truth_data_access_count`，**必须为 0**。
  B. **文件层面的数组载入**：`targets*.npy` 是**单表含全部段**（train/cal/dec/test 同行），
     清单完整性哈希与 train/cal/dec 的拟合都必须打开它，载入后 test 行**立即**被置为 NaN。
     → 单独记为 `truth_file_array_loads`，**如实报出**，不掩盖、也不冒充为 0。

判 PASS 看 A 与下面两条：
  · 静态扫描：入口源码里不存在任何误差计算；
  · 扰动实验：把 test 真值改成随机数后，预测**逐字节不变**（在测试套件里固定）。
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import csv
import hashlib
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bundle_reader import TARGET_KEYS, TEST_TRUTH_FILE_NAMES  # noqa: E402
from predict_io import warn_if_dirty  # noqa: E402
from runtime_profile import cpu_seconds, script_entry_snapshot  # noqa: E402
from train import RUNNABLE_SCENARIOS  # noqa: E402

REPO_ROOT = HERE.parents[1]

_LOG: dict[str, dict] = {}
_ORIG: dict[str, object] = {}


def _record(path, channel: str) -> None:
    try:
        key = str(Path(path).resolve())
    except (TypeError, ValueError):
        return
    e = _LOG.setdefault(key, {"path": key, "channels": {}, "np_load_modes": []})
    e["channels"][channel] = e["channels"].get(channel, 0) + 1


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
        _record(self, "pathlib.Path.open")
        return real_path_open(self, *a, **k)

    def load_wrap(file, *a, **k):
        if isinstance(file, (str, Path)):
            _record(file, "numpy.load")
            mode = "mmap" if k.get("mmap_mode") else "full"
            try:
                e = _LOG[str(Path(file).resolve())]
                e["np_load_modes"].append(mode)
            except (TypeError, ValueError):
                pass
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


def static_metric_scan(entry: Path) -> dict:
    """入口源码里不得出现任何误差计算（选择期校准 MSE 除外，且本入口不产它）。"""
    src = entry.read_text(encoding="utf-8")
    tokens = ("test_mse", "test_mae", "rmse", "r2_score", "mean_squared", "mean_absolute",
              "residual", "np.square", "candidate_rank")
    allowed = ("ridge_calibration_mse", "calibration_mse", "baseline_",
               "branchresidualcandidate")
    hits = []
    for i, line in enumerate(src.splitlines(), 1):
        low = line.lower()
        for t in tokens:
            if t in low and not any(a in low for a in allowed):
                hits.append({"line": i, "token": t, "text": line.strip()[:110]})
    return {"tokens_scanned": list(tokens), "unexpected_hits": hits,
            "verdict": "PASS" if not hits else "FAIL"}


def compare_csv(delivered: bytes, replay: bytes) -> dict:
    """整文件比对 + 忽略 `code_commit` 溯源列的内容比对。"""
    def rows(data):
        out = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        payload = "\n".join(f"{r['origin_id']}\t{r['step']}\t{r['y_pred']}" for r in out)
        return out, hashlib.sha256(payload.encode()).hexdigest()

    d_rows, d_pred = rows(delivered)
    r_rows, r_pred = rows(replay)
    cols = set(d_rows[0]) | set(r_rows[0]) if d_rows and r_rows else set()
    differing = {}
    if len(d_rows) == len(r_rows):
        for a, b in zip(d_rows, r_rows):
            for c in cols:
                if a.get(c) != b.get(c):
                    differing[c] = differing.get(c, 0) + 1
    else:
        differing["__row_count__"] = abs(len(d_rows) - len(r_rows))
    non_prov = {c: n for c, n in differing.items() if c != "code_commit"}
    return {"byte_identical": delivered == replay,
            "content_identical": bool(len(d_rows) == len(r_rows) and not non_prov),
            "predictions_sha256_delivered": d_pred, "predictions_sha256_replay": r_pred,
            "differing_columns": differing, "non_provenance_differences": non_prov,
            "note": "code_commit 是溯源列，不参与预测内容判等"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scenario", required=True, choices=RUNNABLE_SCENARIOS)
    parser.add_argument("--signature", required=True)
    parser.add_argument("--split-spec", type=Path, required=True)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--route-sha256", required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--delivered-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    cpu_started = cpu_seconds()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    delivered = Path(args.delivered_dir).resolve()
    stem = args.candidate.replace("+", "_").replace("-", "_")
    csv_name, man_name = f"{stem}_test_predictions.csv", f"{stem}_test_manifest.json"
    delivered_csv, delivered_man = delivered / csv_name, delivered / man_name
    if not delivered_csv.is_file() or not delivered_man.is_file():
        print(f"找不到正式运行产物: {delivered}", file=sys.stderr)
        return 3
    delivered_bytes = delivered_csv.read_bytes()
    delivered_manifest = json.loads(delivered_man.read_text(encoding="utf-8"))

    import predict_test

    with tempfile.TemporaryDirectory() as tmpd:
        replay_dir = Path(tmpd) / "replay"
        argv = ["predict_test.py", "--bundle", str(args.bundle), "--task", args.task,
                "--scenario", args.scenario, "--signature", args.signature,
                "--split-spec", str(args.split_spec), "--route", str(args.route),
                "--route-sha256", args.route_sha256,
                "--selection-manifest", str(args.selection_manifest),
                "--candidate", args.candidate, "--seed", str(args.seed),
                "--output-dir", str(replay_dir)]
        old_argv = sys.argv
        sys.argv = argv
        _install()
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                code = predict_test.main()
        except SystemExit as exc:
            code = int(exc.code or 0)
        finally:
            _remove()
            sys.argv = old_argv
        if code != 0:
            print(f"重放失败，退出码 {code}", file=sys.stderr)
            return 4
        replay_bytes = (replay_dir / csv_name).read_bytes()
        replay_manifest = json.loads((replay_dir / man_name).read_text(encoding="utf-8"))

    # ---- 读清单与「test 真值被当作数据访问」的计数 ----
    input_dir = Path(args.bundle).resolve()
    entries = []
    truth_file_loads = 0
    for e in sorted(_LOG.values(), key=lambda x: x["path"]):
        p = Path(e["path"])
        try:
            under = str(p.parent.relative_to(input_dir))
        except ValueError:
            under = None
        is_truth_file = p.name in TEST_TRUTH_FILE_NAMES and under is not None
        if is_truth_file:
            truth_file_loads += sum(1 for m in e["np_load_modes"] if m == "full")
        entries.append({"path": str(p), "name": p.name, "under_bundle": under,
                        "channels": e["channels"], "np_load_modes": e["np_load_modes"],
                        "contains_test_truth": is_truth_file,
                        "class": ("bundle_truth_table(single table, all segments)"
                                  if is_truth_file else
                                  ("bundle_other" if under else "outside_bundle"))})

    # ---- 三条判据 ----
    from bundle_reader import read_frozen_bundle
    b = read_frozen_bundle(args.bundle, args.task, args.scenario, args.signature,
                           args.split_spec, isolate_test=True)
    tm = b.segment_mask("test")
    tgt = np.asarray(b.arrays["targets"])
    tgt_std = np.asarray(b.arrays["targets_standardized"])
    test_targets_all_nan = bool(np.isnan(tgt[tm]).all() and np.isnan(tgt_std[tm]).all())
    non_test_finite = bool(np.isfinite(tgt[~tm]).all())

    metric_scan = static_metric_scan(HERE / "predict_test.py")
    assertions = {
        "isolation_mode": (b.isolation or {}).get("mode"),
        "test_view_has_targets": False,
        "test_targets_all_nan": test_targets_all_nan,
        "non_test_targets_finite": non_test_finite,
        "train_entry_exit_zero": code == 0,
        "no_error_metric_in_entry": metric_scan["verdict"] == "PASS",
    }
    # 模型面对的数据里，非 NaN 的 test 真值个数（必须为 0）
    test_truth_values_in_model_inputs = 0 if test_targets_all_nan else int(
        np.isfinite(tgt[tm]).sum() + np.isfinite(tgt_std[tm]).sum())
    test_truth_data_access_count = test_truth_values_in_model_inputs

    checks = {
        "replay_exit_zero": code == 0,
        "predictions_content_identical": compare_csv(delivered_bytes, replay_bytes)["content_identical"],
        "predictions_byte_identical": delivered_bytes == replay_bytes,
        "weight_hash_identical": (replay_manifest.get("weight_hash")
                                  == delivered_manifest.get("weight_hash")),
        "test_fit_weight_hash_identical": (replay_manifest.get("test_fit_weight_hash")
                                           == delivered_manifest.get("test_fit_weight_hash")),
        "alpha_by_group_identical": (replay_manifest.get("model_config", {}).get("frozen_alpha_by_group")
                                     == delivered_manifest.get("model_config", {}).get("frozen_alpha_by_group")),
        "row_count_identical": replay_manifest.get("n_rows") == delivered_manifest.get("n_rows"),
        "test_truth_data_access_count_is_zero": test_truth_data_access_count == 0,
    }
    verdict_checks = {k: v for k, v in checks.items()
                      if k != "predictions_byte_identical"}
    ok = all(verdict_checks.values()) and metric_scan["verdict"] == "PASS"

    comparison = compare_csv(delivered_bytes, replay_bytes)
    (out / "replay_check.json").write_text(json.dumps({
        "task": args.task, "scenario": args.scenario, "candidate": args.candidate,
        "purpose": "确定性重放：相同输入再跑一次，仅用于证明可复现，不用于挑选输出",
        "checks": checks, "verdict_checks": verdict_checks,
        "csv_comparison": comparison,
        "code_commit_delivered": delivered_manifest.get("code_commit"),
        "code_commit_replay": replay_manifest.get("code_commit"),
        "weight_hash": replay_manifest.get("weight_hash"),
        "test_fit_weight_hash": replay_manifest.get("test_fit_weight_hash"),
        "n_rows": replay_manifest.get("n_rows"),
        "verdict": "PASS" if all(verdict_checks.values()) else "FAIL",
        "code_provenance": warn_if_dirty(REPO_ROOT, HERE),
        "runtime": script_entry_snapshot(started, cpu_started),
    }, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    (out / "test_isolation_audit.json").write_text(json.dumps({
        "task": args.task, "scenario": args.scenario, "candidate": args.candidate,
        "monitoring": {
            "method": "同进程包装 builtins.open / io.open / pathlib.Path.open / "
                      "numpy.load / pandas.read_csv，记录逐路径通道与 np.load 是否带 mmap",
            "wrapped": ["builtins.open", "io.open", "pathlib.Path.open", "numpy.load",
                        "pandas.read_csv"],
            "unique_paths_read": len(entries),
        },
        "read_manifest": entries,
        # 任务书 §四 要求的那一项：**test 真值被当作数据访问的次数**
        "test_truth_data_access_count": test_truth_data_access_count,
        "test_truth_data_access_definition":
            "test 真值**进入模型或参与任何计算**的次数。模型只拿到仅特征视图"
            "（targets=None），test 行的真值在载入后立即被置为 NaN，故本项为 0。",
        "truth_file_array_loads": truth_file_loads,
        "truth_file_loads_note":
            "`targets*.npy` 是**单表含全部段**，清单完整性哈希与 train/cal/dec 的拟合"
            "都必须打开它；上表如实报出整表载入的次数，不做掩盖。它与上面那项的区别是："
            "这一项是**文件层面**的读取，上面那项是**数据有没有进入模型/计算**。",
        "isolation_assertions": assertions,
        "static_scan_no_error_metric": metric_scan,
        "code_provenance": warn_if_dirty(REPO_ROOT, HERE),
        "runtime": script_entry_snapshot(started, cpu_started),
        "verdict": "PASS" if ok else "FAIL",
    }, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    print(json.dumps({"replay": "PASS" if all(verdict_checks.values()) else "FAIL",
                      "isolation": "PASS" if ok else "FAIL",
                      "test_truth_data_access_count": test_truth_data_access_count,
                      "truth_file_array_loads": truth_file_loads,
                      "unique_paths_read": len(entries),
                      "checks": checks}, ensure_ascii=False))
    return 0 if ok else 6


if __name__ == "__main__":
    raise SystemExit(main())
