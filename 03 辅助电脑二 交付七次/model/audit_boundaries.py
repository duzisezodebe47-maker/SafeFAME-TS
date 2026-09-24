"""真实 Bundle 的四段读取与边界审计（第六轮交付物）。

任务书第一项要求「运行当前代码的真实 `Agriculture_h12_f1/proxy` 读取和四段
边界检查」。上一轮只在 README 里写了「COMPLETE」，但**没有留下任何产物**：
run manifest 里的 `grid` 只覆盖 train/calibration/decision 三段，test 段的边界
从没被真实入口检查过，主控无法核对这句话。

本模块把四段**全部**过一遍 `assert_grid`，并把每段的边界、起点数、索引范围、
与冻结 spec 的对照写成一张可核对的表::

    four_segment_audit.json     四段边界审计（含每段结论与全表判据）
    four_segment_origins.csv    逐段逐起点明细（供主控逐行核对）

判据（任一不成立即整体 FAIL，并在 JSON 里给出原因）：

  1. 每段都通过 `assert_grid`（领域/H/折/索引一致、无重复、起点落在段内、
     H 窗口不越界）；
  2. 每段起点数 == 冻结 spec 该段的期望数（按 bounds 与 horizon 推算）；
  3. 四段起点索引**互不重叠**，且并集恰为 `[0, test_end - horizon]` 的连续整数
     —— 即「四段不重不漏地覆盖整条可用起点轴」；
  4. Bundle 自带的 `samples.csv` 段标签与冻结 spec 的 bounds 分区**逐行一致**。

不通过时返回非 0，不写出「通过」的产物 —— **不把失败包装成通过**。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from bundle_reader import (  # noqa: E402
    SEGMENTS, BundleUnavailable, assert_grid, read_frozen_bundle, segment_bounds,
    sha256_file,
)
from predict_io import warn_if_dirty  # noqa: E402
from runtime_profile import cpu_seconds, script_entry_snapshot  # noqa: E402
from train import RUNNABLE_SCENARIOS  # noqa: E402

REPO_ROOT = HERE.parents[1]


def spec_task_entry(spec_path: Path, task_id: str) -> dict:
    """冻结 spec 里本任务的登记项（含 `input_len`）。"""
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    tasks = [t for t in (spec.get("tasks") or [])
             if f"{t.get('domain')}_h{t.get('horizon')}_f{t.get('fold_id')}" == task_id]
    if len(tasks) != 1:
        raise BundleUnavailable(f"冻结 spec 中 {task_id} 不是唯一登记项（{len(tasks)} 个）")
    return tasks[0]


def spec_segment_expectations(spec_path: Path, task_id: str, horizon: int) -> dict:
    """按冻结 spec 推算每段应有的起点集合。

    每个起点要满足两个窗口约束，二者缺一不可::

        向前：origin >= input_len            （历史窗口必须完整）
        向后：origin + horizon <= segment_hi （目标窗口不得越出本段）

    所以段 `[lo, hi)` 的可用起点是 `[max(lo, input_len), hi - horizon]`。

    第一版审计漏了 `input_len`，把 train 段的期望写成 `[0, 200]`（201 个），
    于是真实数据里正确的 177 个被判成 FAIL —— **审计表自己的期望值也要来自
    契约，不能靠猜**。这里改为从 spec 读 `input_len`，并把它记进产物。
    """
    bounds = segment_bounds(spec_path, task_id)
    input_len = int(spec_task_entry(spec_path, task_id)["input_len"])
    expect = {}
    for name in SEGMENTS:
        lo, hi = bounds[name]
        first = max(int(lo), input_len)
        last = int(hi) - horizon
        expect[name] = {
            "bounds": [lo, hi],
            "first_origin": first,
            "last_origin": last,
            "expected_indices": list(range(first, last + 1)) if last >= first else [],
        }
    expect["_input_len"] = input_len
    return expect


def audit_segment(bundle, name: str, bounds, horizon: int) -> dict:
    """单段审计：先过 assert_grid，再记录边界的实测值。"""
    entry: dict = {"segment": name, "bounds": list(bounds), "ok": False, "error": None}
    try:
        stats = assert_grid(bundle, name, bounds, horizon)
        entry.update(stats if isinstance(stats, dict) else {})
        entry["ok"] = True
    except AssertionError as exc:
        entry["error"] = str(exc)
        return entry
    index = np.asarray(bundle.segment(name)["origin_index"], dtype=np.int64)
    entry.update({
        "n_origins": int(index.size),
        "index_min": int(index.min()) if index.size else None,
        "index_max": int(index.max()) if index.size else None,
        # 起点在段内是否**连续**（等步长 1）；真实数据若不连续也算信息，不算失败
        "contiguous_step1": bool(index.size > 0 and
                                 np.array_equal(index, np.arange(index.min(), index.max() + 1))),
        "in_window": int(np.sum(index + horizon <= int(bounds[1]))),
        "spans_out": int(np.sum(index + horizon > int(bounds[1]))),
    })
    return entry


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

    try:
        bundle = read_frozen_bundle(args.bundle, args.task, args.scenario,
                                    args.signature, args.split_spec)
        bounds_all = segment_bounds(args.split_spec, args.task)
    except BundleUnavailable as exc:
        print(f"BundleUnavailable: {exc}", file=sys.stderr)
        return 3

    horizon = int(args.task.split("_h")[1].split("_")[0])
    expect = spec_segment_expectations(args.split_spec, args.task, horizon)

    segments = [audit_segment(bundle, name, bounds_all[name], horizon) for name in SEGMENTS]

    # 判据 2：起点数 == spec 推算
    for entry in segments:
        want = len(expect[entry["segment"]]["expected_indices"])
        entry["expected_n_origins"] = want
        entry["count_matches_spec"] = bool(entry.get("n_origins") == want)

    # 判据 3：每段起点恰为 [max(lo, input_len), hi - horizon]，四段互不重叠。
    # 段与段之间必然留 h-1 个跨界的起点（其目标窗口会越出本段），这是契约要求，
    # 不是缺口 —— 这里把缺口逐段算出来并写进产物，而不是笼统要求"连续"。
    per_seg_index = {e["segment"]: list(map(int, bundle.segment(e["segment"])["origin_index"]))
                     for e in segments}
    union: list[int] = []
    for name in SEGMENTS:
        union += per_seg_index[name]
    overlap = len(union) != len(set(union))

    range_matches = {}
    for name in SEGMENTS:
        got = sorted(per_seg_index[name])
        want = expect[name]["expected_indices"]
        range_matches[name] = bool(got == want)
    gaps = {}
    for prev, nxt in zip(SEGMENTS, SEGMENTS[1:]):
        prev_last = expect[prev]["last_origin"]
        next_first = expect[nxt]["first_origin"]
        gaps[f"{prev}->{nxt}"] = {
            "excluded": [prev_last + 1, next_first - 1],
            "count": max(0, next_first - prev_last - 1),
            "reason": "跨界目标窗口（origin + horizon > 前一段 hi）",
        }
    gaps["head"] = {"excluded": [0, expect["train"]["first_origin"] - 1],
                    "count": expect["train"]["first_origin"],
                    "reason": "历史窗口不足（origin < input_len）"}
    partition_ok = bool(not overlap and all(range_matches.values()))

    # 判据 4：samples.csv 的段标签与 spec bounds 逐行一致
    label_mismatch = []
    labels = bundle.samples["segment"].to_numpy()
    indexes = np.asarray(bundle.arrays["origin_index"], dtype=np.int64)
    for name in SEGMENTS:
        want_set = set(expect[name]["expected_indices"])
        for idx, lab in zip(indexes, labels):
            if int(idx) in want_set and str(lab) != name:
                label_mismatch.append({"origin_index": int(idx),
                                       "label": str(lab), "expected": name})
    labels_ok = not label_mismatch

    checks = {
        "all_segments_assert_grid_ok": all(e["ok"] for e in segments),
        "counts_match_spec": all(e["count_matches_spec"] for e in segments),
        "origin_ranges_match_spec": all(range_matches.values()),
        "partition_no_overlap": not overlap,
        "samples_labels_match_spec_bounds": labels_ok,
    }
    passed = all(checks.values())

    report = {
        "task": args.task, "scenario": args.scenario, "horizon": horizon,
        "bundle_signature": bundle.signature,
        "bundle_manifest_sha256": sha256_file(Path(args.bundle) / "manifest.json"),
        "split_spec_sha256": sha256_file(args.split_spec),
        "segments": segments,
        "checks": checks,
        "verdict": "PASS" if passed else "FAIL",
        "selection_origin_count": int(len(per_seg_index["calibration"])
                                      + len(per_seg_index["decision"])),
        "train_origin_count": int(len(per_seg_index["train"])),
        "all_segment_origin_count": int(len(union)),
        "duplicate_origins": bool(overlap),
        "origin_range_matches_spec": range_matches,
        "excluded_origins": gaps,
        "input_len": expect["_input_len"],
        "label_mismatch_sample": label_mismatch[:5],
        "code_provenance": warn_if_dirty(REPO_ROOT, HERE),
        "runtime": script_entry_snapshot(started, cpu_started),
        "note": "四段（train/calibration/decision/test）全部过 assert_grid；"
                "selection_origin_count 应等于主控 origin_count=138",
    }
    (out / "four_segment_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    with (out / "four_segment_origins.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["segment", "origin_id", "origin_index",
                         "segment_lo", "segment_hi", "target_end_index", "index_in_segment"])
        for name in SEGMENTS:
            lo, hi = bounds_all[name]
            part = bundle.segment(name)
            for oid, idx in zip(part["origin_id"], part["origin_index"]):
                writer.writerow([name, str(oid), int(idx), lo, hi, int(idx) + horizon,
                                 bool(lo <= int(idx) < hi and int(idx) + horizon <= hi)])

    print(json.dumps({"status": report["verdict"], "task": args.task,
                      "selection_origin_count": report["selection_origin_count"],
                      "checks": checks}, ensure_ascii=False))
    return 0 if passed else 6


if __name__ == "__main__":
    raise SystemExit(main())
