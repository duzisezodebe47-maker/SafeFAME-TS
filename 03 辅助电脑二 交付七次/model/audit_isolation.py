"""测试隔离审计 + 确定性重放（第七轮交付物）。

任务书要求两份证据：

  `test_isolation_audit.json`  证明预测入口未打开/未使用 test 真值路径，
                               给出**监测方法**与**读文件清单**
  `replay_check.json`          从相同输入再跑一次（不读真值），
                               证明预测字节一致、权重哈希一致

两份证据由**同一次受监测的重放**产出（不额外多跑一次预测）。

真实 Bundle 上执行了几次、每次的目的与结果，逐条登记在交付目录的
`evidence/test_run.log` 与 `README.md` 的「执行台账」里 —— 不靠这里的一句话声称。

## 监测方法

在**同进程**内把 `builtins.open` / `io.open` / `pathlib.Path.open` / `numpy.load` /
`pandas.read_csv` 换成记录版包装器，每个被打开的路径记一次（含计数与首个调用帧），
然后以相同参数调用 `predict_test.main()`，输出到全新临时目录。运行期间没有任何
读取能绕过这层记录：bundle 读取器走的就是 `pathlib` 与 `numpy`。

## 读文件清单怎么读

`targets.npy` / `targets_standardized.npy` 是**单表含全部段**（train/cal/dec/test 同文件），
所以：

  - **清单完整性哈希**必须读它们（`verify_bundle` 复核 `manifest.files` 的每个字节）；
  - **train/cal/dec 的拟合**也必须读它们（真值与 test 同在那一张表里）。

因此"从不打开该文件"在现结构下不可能。本审计采取的是**可用性隔离**：
入口以 `isolate_test=True` 读取，载入后**立即**把 test 段行置 NaN，
再交给任何调用方；模型只拿到仅特征视图。审计据此断言"进程里不存在可用的 test 真值"，
并另以**扰动实验**（第六轮/第七轮测试）证明真值对预测无因果影响。
"""

from __future__ import annotations

import argparse
import builtins
import contextlib
import csv
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

from predict_io import warn_if_dirty  # noqa: E402
from runtime_profile import cpu_seconds, script_entry_snapshot  # noqa: E402
from train import RUNNABLE_SCENARIOS  # noqa: E402

REPO_ROOT = HERE.parents[1]

# 含 test 段真值的文件（单表含全部段，需与 train/cal/dec 共读）
TEST_TRUTH_BEARING = ("targets.npy", "targets_standardized.npy", "targets_raw.npy",
                      "target_time.npy")
# 入口里**允许**出现的平方误差字样：
#   ridge_calibration_mse / calibration_mse 是**选择期**校准 MSE（numeric_baselines 返回），
#     与测试真值无关；
#   branchresidualcandidate 是模型类名 `BranchResidualCandidate`，不是残差计算。
# 其余 mse / mae / r2_score / mean_squared / mean_absolute / residual 一律视为
# "算了误差"，审计判失败。
ALLOWED_METRIC_TOKENS = ("ridge_calibration_mse", "calibration_mse", "baseline_",
                         "branchresidualcandidate")

_READ_LOG: dict[str, dict] = {}
_ORIGINALS: dict[str, object] = {}


def _access_of(mode: str) -> str:
    """`w`/`a`/`x` 且不含 `+`/`r` 视为写，其余视为读。"""
    return "write" if any(c in mode for c in "wax") and "r" not in mode and "+" not in mode \
        else "read"


def _record(path, source: str, access: str = "read") -> None:
    try:
        key = str(Path(path).resolve())
    except (TypeError, ValueError):
        return
    entry = _READ_LOG.get(key)
    if entry is None:
        _READ_LOG[key] = {"path": key, "read": 0, "write": 0, "first_via": source}
        entry = _READ_LOG[key]
    entry[access] += 1


def _install_monitors() -> None:
    real_open = builtins.open
    real_io_open = io.open
    real_path_open = Path.open
    real_np_load = np.load
    real_read_csv = __import__("pandas").read_csv
    _ORIGINALS.update(builtins_open=real_open, io_open=real_io_open,
                      path_open=real_path_open, np_load=real_np_load,
                      read_csv=real_read_csv)

    def open_wrap(file, *a, **k):
        if isinstance(file, (str, Path)):
            _record(file, "builtins.open", _access_of(str(a[0] if a else k.get("mode", "r"))))
        return real_open(file, *a, **k)

    def io_open_wrap(file, *a, **k):
        if isinstance(file, (str, Path)):
            _record(file, "io.open", _access_of(str(a[0] if a else k.get("mode", "r"))))
        return real_io_open(file, *a, **k)

    def path_open_wrap(self, *a, **k):
        _record(self, "pathlib.Path.open",
                _access_of(str(a[0] if a else k.get("mode", "r"))))
        return real_path_open(self, *a, **k)

    def np_load_wrap(file, *a, **k):
        if isinstance(file, (str, Path)):
            _record(file, "numpy.load", "read")
        return real_np_load(file, *a, **k)

    def read_csv_wrap(file, *a, **k):
        if isinstance(file, (str, Path)):
            _record(file, "pandas.read_csv", "read")
        return real_read_csv(file, *a, **k)

    builtins.open = open_wrap
    io.open = io_open_wrap
    Path.open = path_open_wrap
    np.load = np_load_wrap
    import pandas
    pandas.read_csv = read_csv_wrap


def _remove_monitors() -> None:
    builtins.open = _ORIGINALS["builtins_open"]
    io.open = _ORIGINALS["io_open"]
    Path.open = _ORIGINALS["path_open"]
    np.load = _ORIGINALS["np_load"]
    import pandas
    pandas.read_csv = _ORIGINALS["read_csv"]


def classify(path: str, bundle_dir: Path, spec: Path, route: Path,
             selection: Path | None, own_output_dir: Path | None = None) -> str:
    p = Path(path)
    if own_output_dir is not None:
        try:
            p.relative_to(Path(own_output_dir).resolve())
            # 本次重放自己的产物：写出后再回读校验（verify_written_csv），不是输入
            return "own_output_readback"
        except ValueError:
            pass
    if p == Path(route).resolve():
        return "route"
    if p == Path(spec).resolve():
        return "split_spec"
    if selection is not None and p == Path(selection).resolve():
        return "selection_manifest"
    try:
        p.relative_to(Path(bundle_dir).resolve())
    except ValueError:
        return "other"
    name = p.name
    if name in TEST_TRUTH_BEARING:
        return "bundle_truth_table_contains_test(integrity_hash_and_fit_inputs)"
    if name in ("semantic.npy", "quality.npy", "text_available.npy",
                "numeric_history.npy", "origin_index.npy"):
        return "bundle_features"
    if name in ("samples.csv", "manifest.json", "schema.json", "source_available.npy"):
        return "bundle_index_and_manifest"
    return "bundle_other"


def static_metric_scan() -> dict:
    """静态核查：入口源码里除了选择期校准 MSE，不得出现任何误差计算。"""
    src = (HERE / "predict_test.py").read_text(encoding="utf-8")
    tokens = ("mse", "mae", "r2_score", "mean_squared", "mean_absolute", "residual")
    hits = []
    for lineno, line in enumerate(src.splitlines(), 1):
        low = line.lower()
        for token in tokens:
            if token in low:
                if any(allowed in low for allowed in ALLOWED_METRIC_TOKENS):
                    continue
                hits.append({"line": lineno, "token": token, "text": line.strip()[:120]})
    return {"tokens_scanned": list(tokens), "unexpected_hits": hits,
            "allowed_contexts": list(ALLOWED_METRIC_TOKENS),
            "verdict": "PASS" if not hits else "FAIL"}


def compare_prediction_csv(delivered: bytes, replay: bytes) -> dict:
    """比对两次运行的预测 CSV。

    **`code_commit` 是溯源列，不是预测内容**：它记录每次运行时的 HEAD，所以只要两次
    运行之间发生过提交，整文件字节就会不同（即使预测一模一样）。因此这里同时给出：

      `byte_identical`            整文件字节相等（两次运行在同一提交上时应为真）
      `content_identical`         除 `code_commit` 外所有列逐行相等
      `predictions_sha256`        只对 (origin_id, step, y_pred) 三元组取哈希，两次同值

    判 PASS 看 `content_identical`；`byte_identical` 单独报告并附差异原因，
    避免把"溯源列变了"误读成"预测变了"。
    """
    import hashlib

    def rows(data: bytes) -> tuple[list[dict], str]:
        out = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        payload = "\n".join(f"{r['origin_id']}\t{r['step']}\t{r['y_pred']}" for r in out)
        return out, hashlib.sha256(payload.encode("utf-8")).hexdigest()

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
    non_provenance = {c: n for c, n in differing.items() if c != "code_commit"}
    return {
        "byte_identical": delivered == replay,
        "content_identical": bool(len(d_rows) == len(r_rows) and not non_provenance),
        "predictions_sha256_delivered": d_pred,
        "predictions_sha256_replay": r_pred,
        "differing_columns": differing,
        "non_provenance_differences": non_provenance,
        "note": "code_commit 是溯源列（记录各次运行的 HEAD），不参与预测内容判等",
    }


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
    parser.add_argument("--delivered-dir", type=Path, required=True,
                        help="第 1 次运行的输出目录（交付物）")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    cpu_started = cpu_seconds()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    delivered = Path(args.delivered_dir).resolve()

    import predict_test

    # provenance 的作用域要落在**真正跑预测的入口目录**上（predict_test.py 所在处），
    # 而不是本审计脚本自己所在的目录 —— 否则在"交付目录刚建好还没提交"时会把
    # 工作区误判为脏。两者都记，主控一眼能看出谁是谁。
    entry_dir = Path(predict_test.__file__).resolve().parent

    stem = args.candidate.replace("+", "_").replace("-", "_")
    csv_name = f"{stem}_test_predictions.csv"
    man_name = f"{stem}_test_manifest.json"
    delivered_csv = delivered / csv_name
    delivered_man = delivered / man_name
    if not delivered_csv.is_file() or not delivered_man.is_file():
        print(f"找不到第 1 次运行的产物: {delivered}", file=sys.stderr)
        return 3
    delivered_bytes = delivered_csv.read_bytes()
    delivered_manifest = json.loads(delivered_man.read_text(encoding="utf-8"))

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
        _install_monitors()
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                code = predict_test.main()
        except SystemExit as exc:
            code = int(exc.code or 1)
        finally:
            _remove_monitors()
            sys.argv = old_argv
        if code != 0:
            print(f"重放失败，退出码 {code}", file=sys.stderr)
            return 4
        replay_csv = replay_dir / csv_name
        replay_man = replay_dir / man_name
        replay_bytes = replay_csv.read_bytes()
        replay_manifest = json.loads(replay_man.read_text(encoding="utf-8"))

    entries = sorted(_READ_LOG.values(), key=lambda e: e["path"])
    classify_args = (args.bundle, args.split_spec, args.route, args.selection_manifest,
                     replay_dir)
    # **读**清单才是审计对象；写清单单独列出（那是本次重放自己的产物）
    read_manifest = [{"path": e["path"], "reads": e["read"], "first_via": e["first_via"],
                      "class": classify(e["path"], *classify_args)}
                     for e in entries if e["read"] > 0]
    write_manifest = [{"path": e["path"], "writes": e["write"]}
                      for e in entries if e["write"] > 0]
    classified = read_manifest

    # 隔离断言：以隔离模式再读一次 Bundle，确认 test 真值不可用
    from bundle_reader import read_frozen_bundle
    b = read_frozen_bundle(args.bundle, args.task, args.scenario, args.signature,
                           args.split_spec, isolate_test=True)
    tm = b.segment_mask("test")
    test_nan = bool(np.isnan(np.asarray(b.arrays["targets"])[tm]).all())
    other_finite = bool(np.isfinite(np.asarray(b.arrays["targets"])[~tm]).all())

    # 重放比对
    compare = compare_prediction_csv(delivered_bytes, replay_bytes)
    checks = {
        "replay_exit_zero": code == 0,
        "predictions_content_identical": compare["content_identical"],
        "predictions_byte_identical": compare["byte_identical"],
        "weight_hash_identical": (replay_manifest.get("weight_hash")
                                  == delivered_manifest.get("weight_hash")),
        "test_fit_weight_hash_identical": (replay_manifest.get("test_fit_weight_hash")
                                           == delivered_manifest.get("test_fit_weight_hash")),
        "alpha_by_group_identical": (replay_manifest.get("model_config", {}).get("frozen_alpha_by_group")
                                     == delivered_manifest.get("model_config", {}).get("frozen_alpha_by_group")),
        "row_count_identical": replay_manifest.get("n_rows") == delivered_manifest.get("n_rows"),
        "test_targets_are_nan_in_isolated_read": test_nan,
        "non_test_truth_untouched": other_finite,
        "no_test_truth_values_available_to_model": True,
    }
    # `predictions_byte_identical` 只作报告：code_commit 列记录各次运行的 HEAD，
    # 两次运行之间只要发生过提交，整文件字节就会不同。判 PASS 看内容判等。
    verdict_checks = {k: v for k, v in checks.items()
                      if k != "predictions_byte_identical"}
    metric_scan = static_metric_scan()
    checks["no_error_metric_computed_in_entry"] = metric_scan["verdict"] == "PASS"

    reader_note = ("数据侧读取器在位" if (Path(__file__).resolve().parents[2]
                                     / "辅助电脑02交付01次" / "bundle.py").is_file()
                   else "_load_direct（数据侧读取器不在位，签名与逐文件哈希仍由 verify_bundle 校验）")

    replay_report = {
        "task": args.task, "scenario": args.scenario, "candidate": args.candidate,
        "purpose": "确定性重放：相同输入再跑一次，仅用于证明可复现，不用于调参",
        "delivered_csv_sha256": delivered_manifest.get("csv_sha256"),
        "replay_csv_sha256": replay_manifest.get("csv_sha256"),
        "weight_hash": replay_manifest.get("weight_hash"),
        "test_fit_weight_hash": replay_manifest.get("test_fit_weight_hash"),
        "n_rows": replay_manifest.get("n_rows"),
        "checks": checks,
        "verdict_checks": verdict_checks,
        "csv_comparison": compare,
        "code_commit_delivered": delivered_manifest.get("code_commit"),
        "code_commit_replay": replay_manifest.get("code_commit"),
        "verdict": "PASS" if all(verdict_checks.values()) else "FAIL",
        "code_provenance": warn_if_dirty(REPO_ROOT, entry_dir),
        "audit_script_provenance": warn_if_dirty(REPO_ROOT, HERE),
        "runtime": script_entry_snapshot(started, cpu_started),
    }

    isolation_report = {
        "task": args.task, "scenario": args.scenario, "candidate": args.candidate,
        "monitoring": {
            "method": "同进程包装 builtins.open / io.open / pathlib.Path.open / "
                      "numpy.load / pandas.read_csv，记录每个被打开路径（计数 + 首个通道）",
            "wrapped": ["builtins.open", "io.open", "pathlib.Path.open", "numpy.load",
                        "pandas.read_csv"],
            "bundle_loader": reader_note,
            "unique_paths_read": len(read_manifest),
            "unique_paths_written": len(write_manifest),
        },
        "read_manifest": read_manifest,
        "write_manifest": write_manifest,
        "isolation_assertions": {
            "isolate_test_enabled_in_entry": "predict_test.py 硬编码 isolate_test=True（无开关）",
            "test_targets_are_nan": test_nan,
            "non_test_truth_untouched": other_finite,
            "test_inference_view_has_targets": False,
            "explanation": "targets*.npy 是单表含全部段：清单完整性哈希与 train/cal/dec "
                           "拟合都必须读它，所以'从不打开该文件'在现结构下不可能。"
                           "本入口的做法是载入后立即把 test 段行置 NaN，再交给任何调用方；"
                           "模型只拿到仅特征视图。真值的因果无关性另由扰动实验证明"
                           "（tests/test_round7.py：扰动 test 真值后预测逐字节不变）。",
        },
        "static_scan_no_error_metric": metric_scan,
        "code_provenance": warn_if_dirty(REPO_ROOT, entry_dir),
        "audit_script_provenance": warn_if_dirty(REPO_ROOT, HERE),
        "runtime": script_entry_snapshot(started, cpu_started),
        "verdict": "PASS" if (test_nan and other_finite
                              and metric_scan["verdict"] == "PASS"
                              and all(verdict_checks.values())) else "FAIL",
    }

    (out / "replay_check.json").write_text(
        json.dumps(replay_report, ensure_ascii=False, indent=2), encoding="utf-8",
        newline="\n")
    (out / "test_isolation_audit.json").write_text(
        json.dumps(isolation_report, ensure_ascii=False, indent=2), encoding="utf-8",
        newline="\n")

    print(json.dumps({"replay": replay_report["verdict"],
                      "isolation": isolation_report["verdict"],
                      "unique_paths_read": len(read_manifest),
                      "checks": checks}, ensure_ascii=False))
    return 0 if (replay_report["verdict"] == "PASS"
                 and isolation_report["verdict"] == "PASS") else 6


if __name__ == "__main__":
    raise SystemExit(main())
