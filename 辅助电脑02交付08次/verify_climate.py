"""Standalone verification and feature-only Climate audit for the strict Release."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


sys.dont_write_bytecode = True
TASK = "Climate_h4_f2"
BUNDLE = "a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38"
SPEC_SHA = "a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031"
FORMAL_MANIFEST_SHA = "6c889af1a687592453a5d1d4d83dc2b4eaf9118653f1ac1085b1997c91800143"
BOUNDS = {"train": (0, 636), "calibration": (636, 763),
          "decision": (763, 1017), "test": (1017, 1144)}
SEGMENTS = tuple(BOUNDS)
SCENARIOS = ("proxy", "conservative_lag")
SAFE_SHARED = ("origin_index.npy", "horizon_time.npy", "numeric_missing.npy")
SAFE_SCENARIO = ("semantic.npy", "quality.npy", "source_available.npy", "text_available.npy")
FIT_ONLY = ("numeric_history.npy", "numeric_history_raw.npy", "frequency.npy",
            "targets.npy", "targets_raw.npy", "targets_standardized.npy")


class VerificationError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_array(path: Path) -> np.ndarray:
    require(path.is_file(), f"missing feature file: {path.name}")
    return np.load(path, allow_pickle=False)


def byte_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype \
        and np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes()


def check_package(root: Path) -> tuple[pd.DataFrame, dict]:
    require((root / "MANIFEST.json").is_file(), "MANIFEST.json missing")
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    entries = manifest["files"]
    expected = set()
    for entry in entries:
        relative = entry["relative_path"]
        parts = PurePosixPath(relative)
        require(not parts.is_absolute() and ".." not in parts.parts, f"manifest path escapes Release: {relative}")
        file = root / relative
        require(file.is_file(), f"missing Release input: {relative}")
        require(file.stat().st_size == entry["bytes"] and digest(file) == entry["sha256"],
                f"manifest SHA256/bytes mismatch: {relative}")
        expected.add(relative)
    actual = {item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_file()
              and "__pycache__" not in item.parts and item.suffix != ".pyc"}
    require(actual == expected | {"MANIFEST.json"},
            f"unexpected Release file(s): {sorted(actual - expected - {'MANIFEST.json'})}; missing: {sorted(expected - actual)}")
    test_files = {item.relative_to(root / "test_features").as_posix()
                  for item in (root / "test_features").rglob("*") if item.is_file()}
    allowed_test = set(SAFE_SHARED) | {"metadata.csv"} | {
        f"{scenario}/{name}" for scenario in SCENARIOS for name in SAFE_SCENARIO}
    require(test_files == allowed_test,
            f"forbidden or missing test feature file: {sorted(test_files ^ allowed_test)}")
    for file in test_files:
        require(not re.search(r"target|truth|label|numeric_history|frequency", file, re.I),
                f"forbidden test truth/derived field: {file}")

    anchors = json.loads((root / "source_anchors.json").read_text(encoding="utf-8"))
    require(anchors["bundle_signature"] == BUNDLE and anchors["split_spec_sha256"] == SPEC_SHA,
            "frozen Bundle or spec anchor mismatch")
    formal_path = root / "anchors/formal_bundle_manifest.json"
    require(digest(formal_path) == FORMAL_MANIFEST_SHA
            and anchors["formal_bundle_manifest_sha256"] == FORMAL_MANIFEST_SHA,
            "formal Bundle manifest SHA256 mismatch")
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    require(formal["bundle_signature"] == BUNDLE, "formal Bundle manifest signature mismatch")
    for relative, expected_sha in anchors["source_safe_full_file_sha256"].items():
        original = "target_time.npy" if relative == "horizon_time.npy" else relative
        require(expected_sha == formal["files"][f"{TASK}/{original}"],
                f"feature anchor differs from formal Bundle manifest: {relative}")
        require(digest(root / "source_safe" / relative) == expected_sha,
                f"formal Bundle feature SHA256 mismatch: {relative}")
    for scenario, expected_sha in anchors["text_trace_full_file_sha256"].items():
        require(expected_sha == formal["files"][f"{TASK}/{scenario}/text_trace.jsonl"],
                f"trace anchor differs from formal Bundle manifest: {scenario}")
        require(digest(root / "text_trace" / f"{scenario}.jsonl") == expected_sha,
                f"formal Bundle trace SHA256 mismatch: {scenario}")
    commitment = json.loads((root / "truth_commitment.json").read_text(encoding="utf-8"))
    require(commitment["task"] == TASK and commitment["bundle_signature"] == BUNDLE
            and commitment["split_spec_sha256"] == SPEC_SHA and commitment["test_shape"] == [124, 4]
            and commitment["test_dtype"] == "float32", "test truth commitment fields mismatch")
    require(set(commitment) == {"task", "bundle_signature", "split_spec_sha256", "source_file",
                                "full_bundle_file_sha256", "test_slice_npy_sha256", "test_shape",
                                "test_dtype", "test_origin_count"}, "truth commitment contains unexpected fields")
    require(commitment["full_bundle_file_sha256"] == formal["files"][commitment["source_file"]],
            "truth commitment differs from formal Bundle manifest")

    mapping = pd.read_csv(root / "mapping.csv", keep_default_na=False)
    require(len(mapping) == 1080 and mapping.origin_id.is_unique and mapping.origin_index.is_unique,
            "duplicate origin or wrong mapping count")
    require(np.array_equal(mapping.source_bundle_row.to_numpy(), np.arange(len(mapping))),
            "source Bundle row mapping mismatch")
    require(mapping.segment.value_counts().to_dict() == {"train": 581, "calibration": 124,
            "decision": 251, "test": 124}, "mapping segment counts mismatch")
    axis = pd.read_csv(root / "weekly_axis.csv")
    require(len(axis) == 1272 and np.array_equal(axis.numerical_row.to_numpy(), np.arange(1272)),
            "weekly axis row grid mismatch")
    source_origin = load_array(root / "source_safe/origin_index.npy")
    source_time = load_array(root / "source_safe/horizon_time.npy")
    require(source_time.shape == (1080, 4), "horizon time shape mismatch")
    for row in mapping.itertuples():
        lo, hi = BOUNDS[row.segment]
        require(row.origin_index >= max(lo, 52) and row.origin_index + 4 <= hi,
                f"origin+horizon crosses segment boundary: {row.origin_id}")
        require(row.segment_lo == lo and row.segment_hi == hi,
                f"segment boundary metadata mismatch: {row.origin_id}")
        require(row.cutoff_time[:10] == axis.start_date.iloc[row.origin_index]
                and row.history_start[:10] == axis.start_date.iloc[row.origin_index - 52],
                f"origin cutoff/history time grid mismatch: {row.origin_id}")
        actual_time = list(source_time[row.source_bundle_row].astype(str))
        expected_time = axis.start_date.iloc[row.origin_index:row.origin_index + 4].tolist()
        require([value[:10] for value in actual_time] == expected_time,
                f"horizon time grid mismatch: {row.origin_id}")
    require(np.array_equal(mapping.origin_index.to_numpy(), source_origin), "origin differs from formal Bundle")
    for partition, mask in (("selection_fit", mapping.segment.ne("test")),
                            ("test_features", mapping.segment.eq("test"))):
        rows = np.flatnonzero(mask.to_numpy())
        subset = mapping.loc[mask].reset_index(drop=True)
        require(np.array_equal(subset.package_row.to_numpy(), np.arange(len(rows)))
                and (subset.package_partition == partition).all(),
                f"{partition} row mapping mismatch")
        metadata = pd.read_csv(root / partition / "metadata.csv", keep_default_na=False)
        if partition == "selection_fit":
            require(not (metadata.segment == "test").any(), "selection package contains test row")
        require(metadata.origin_id.tolist() == subset.origin_id.tolist()
                and metadata.segment.tolist() == subset.segment.tolist(),
                f"{partition} metadata/mapping mismatch")
        if partition == "test_features":
            require((metadata.segment == "test").all(), "test package contains non-test row")
        if partition == "test_features":
            require(not any(re.search(r"target|truth|label", column, re.I) for column in metadata.columns),
                    "test metadata contains target/truth column")
        for name in SAFE_SHARED:
            source = load_array(root / "source_safe" / name)
            sub = load_array(root / partition / name)
            require(sub.shape[0] == len(rows) and byte_equal(sub, source[rows]),
                    f"feature mismatch with formal Bundle: {partition}/{name}")
        for scenario in SCENARIOS:
            for name in SAFE_SCENARIO:
                source = load_array(root / "source_safe" / scenario / name)
                sub = load_array(root / partition / scenario / name)
                require(sub.shape[0] == len(rows) and byte_equal(sub, source[rows]),
                        f"feature mismatch with formal Bundle: {partition}/{scenario}/{name}")
    commitments = json.loads((root / "selection_fit_commitments.json").read_text(encoding="utf-8"))
    for name in FIT_ONLY:
        file = root / "selection_fit" / name
        array = load_array(file)
        require(commitments[name]["full_bundle_file_sha256"] == formal["files"][f"{TASK}/{name}"],
                f"selection parent anchor differs from formal Bundle: {name}")
        require(array.shape[0] == 956 and digest(file) == commitments[name]["selection_slice_npy_sha256"],
                f"selection fit slice commitment mismatch: {name}")
    return mapping, anchors


def describe(values: np.ndarray) -> dict:
    flat = np.asarray(values, dtype=float).reshape(-1)
    finite = flat[np.isfinite(flat)]
    require(len(finite) > 0, "empty finite feature distribution")
    return {"n": int(len(flat)), "finite_n": int(len(finite)),
            "mean": float(finite.mean()), "std": float(finite.std()),
            "q25": float(np.quantile(finite, .25)), "median": float(np.median(finite)),
            "q75": float(np.quantile(finite, .75)),
            "missing_rate": float(1 - len(finite) / len(flat))}


def audit(root: Path, out: Path, mapping: pd.DataFrame, anchors: dict) -> list[str]:
    out.mkdir(parents=True, exist_ok=True)
    audit_rows = pd.read_csv(root / "sample_audit.csv", keep_default_na=False)
    boundary = []
    for segment in SEGMENTS:
        group = mapping.loc[mapping.segment.eq(segment)]
        excluded = audit_rows.loc[audit_rows.segment.eq(segment) & audit_rows.reason.ne("retained")]
        partition = "test_features" if segment == "test" else "selection_fit"
        mask = load_array(root / partition / "numeric_missing.npy")[group.package_row.to_numpy()]
        boundary.append({"segment": segment, "candidate_origins": int(len(audit_rows.loc[audit_rows.segment.eq(segment)])),
                         "retained_origins": int(len(group)), "origin_min": int(group.origin_index.min()),
                         "origin_max": int(group.origin_index.max()),
                         "excluded_by_reason": {k: int(v) for k, v in excluded.reason.value_counts().items()},
                         "historical_cells": int(mask.size), "historical_missing_cells": int(mask.sum()),
                         "historical_missing_rate": float(mask.mean())})
    write_json(out / "climate_boundary_audit.json", {
        "task": TASK, "bundle_signature": BUNDLE, "split_spec_sha256": SPEC_SHA,
        "input_len": 52, "horizon": 4, "period": 52, "bounds": [636, 763, 1017, 1144],
        "segments": boundary, "test_truth_used_for_audit": False,
        "test_numeric_history_distributed": False,
    })

    drift = []
    for segment in SEGMENTS:
        group = mapping.loc[mapping.segment.eq(segment)]
        partition = "test_features" if segment == "test" else "selection_fit"
        rows = group.package_row.to_numpy()
        if segment == "test":
            aggregate = json.loads((root / "test_history_aggregate.json").read_text(encoding="utf-8"))
            numeric_stats = {"n": aggregate["history_cells"], "finite_n": aggregate["finite_cells"],
                             "mean": aggregate["mean"], "std": aggregate["std"],
                             "q25": aggregate["q25"], "median": aggregate["median"],
                             "q75": aggregate["q75"],
                             "missing_rate": aggregate["missing_cells"] / aggregate["history_cells"]}
            numeric_basis = "pre-release aggregate of historical features; no per-origin test history in Release"
        else:
            numeric_stats = describe(load_array(root / partition / "numeric_history_raw.npy")[rows])
            numeric_basis = "recomputed from selection_fit historical input"
        drift.append({"segment": segment, "feature": "numeric_history_raw/OT", "unit": "source OT unit",
                      "basis": numeric_basis, **numeric_stats})
        for scenario in SCENARIOS:
            available = load_array(root / partition / scenario / "text_available.npy")[rows]
            drift.append({"segment": segment, "feature": f"{scenario}/text_available", "unit": "0/1",
                          "basis": "recomputed from safe released feature", **describe(available)})
            quality = load_array(root / partition / scenario / "quality.npy")[rows]
            for index in range(quality.shape[1]):
                drift.append({"segment": segment, "feature": f"{scenario}/quality_{index}", "unit": "feature value",
                              "basis": "recomputed from safe released feature", **describe(quality[:, index])})
    drift_df = pd.DataFrame(drift)
    train = drift_df.loc[drift_df.segment.eq("train")].set_index("feature")
    drift_df["delta_mean_from_train"] = [row["mean"] - train.loc[row["feature"], "mean"]
                                          for _, row in drift_df.iterrows()]
    drift_df["delta_mean_in_train_sd"] = [row["delta_mean_from_train"] / train.loc[row["feature"], "std"]
                                         if train.loc[row["feature"], "std"] > 0 else None
                                         for _, row in drift_df.iterrows()]
    drift_df.to_csv(out / "climate_feature_drift.csv", index=False)

    coverage = []
    for scenario in SCENARIOS:
        traces = [json.loads(line) for line in (root / "text_trace" / f"{scenario}.jsonl").read_text(encoding="utf-8").splitlines()]
        require(len(traces) == len(mapping), f"trace row count mismatch: {scenario}")
        require([row["origin_id"] for row in traces] == mapping.origin_id.tolist(),
                f"trace origin order mismatch: {scenario}")
        for segment in SEGMENTS:
            indices = np.flatnonzero(mapping.segment.eq(segment).to_numpy())
            subset = [traces[i] for i in indices]
            counts = np.asarray([len(row["report_ids"]) + len(row["search_ids"]) for row in subset])
            report = np.asarray([len(row["report_ids"]) for row in subset])
            search = np.asarray([len(row["search_ids"]) for row in subset])
            partition = "test_features" if segment == "test" else "selection_fit"
            rows = mapping.iloc[indices].package_row.to_numpy()
            availability = load_array(root / partition / scenario / "text_available.npy")[rows]
            require(np.array_equal(counts > 0, availability), f"text availability/trace mismatch: {scenario}/{segment}")
            require(int(max(report)) <= 32 and int(max(search)) <= 32,
                    f"source selection cap exceeded: {scenario}/{segment}")
            coverage.append({"segment": segment, "scenario": scenario, "origins_denominator": len(indices),
                             "text_available_origins": int(availability.sum()),
                             "text_available_rate": float(availability.mean()),
                             "selected_facts_total": int(counts.sum()),
                             "selected_facts_q25": float(np.quantile(counts, .25)),
                             "selected_facts_median": float(np.median(counts)),
                             "selected_facts_q75": float(np.quantile(counts, .75)),
                             "report_cap_32_origins": int((report == 32).sum()),
                             "search_cap_32_origins": int((search == 32).sum()),
                             "either_cap_32_rate": float(((report == 32) | (search == 32)).mean()),
                             "cap_exceeded_reason_origins": sum(row["reason_counts"].get("source_cap_exceeded", 0) > 0
                                                                for row in subset)})
    pd.DataFrame(coverage).to_csv(out / "climate_source_coverage.csv", index=False)

    lineage = pd.read_csv(root / "source_lineage_slim.csv", keep_default_na=False)
    source_rows = []
    for source, group in lineage.groupby("source", sort=True):
        retained = group.loc[group.reason.eq("retained")]
        nonempty = group.loc[group.reason.ne("missing_fact")]
        duplicates = int(group.reason.eq("duplicate_fact_later").sum())
        unverified = ~retained.publication_verified.astype(str).str.lower().eq("true")
        source_rows.append({"source": source, "raw_rows_denominator": len(group),
                            "nonempty_rows_denominator": len(nonempty), "canonical_facts": len(retained),
                            "start_min": retained.start_date.min(), "end_max": retained.end_date.max(),
                            "duplicate_fact_rows": duplicates,
                            "duplicate_rate_nonempty": duplicates / len(nonempty),
                            "original_url_missing_count": int(retained.source_url.eq("").sum()),
                            "original_url_missing_rate": float(retained.source_url.eq("").mean()),
                            "publication_time_proxy_count": int(unverified.sum()),
                            "publication_time_proxy_rate": float(unverified.mean()),
                            "source_file": retained.source_file.iloc[0],
                            "upstream_revision": "00281e2d86058286d5548b15a7670e8eda57ef62"})
    pd.DataFrame(source_rows).to_csv(out / "climate_source_evidence.csv", index=False)

    axis = pd.read_csv(root / "weekly_axis.csv")
    starts = pd.to_datetime(axis.start_date)
    ends = pd.to_datetime(axis.end_date)
    delta = starts.diff().dt.days
    weekly = axis.copy()
    weekly["delta_days_from_previous"] = delta
    weekly["missing_weeks_before"] = delta.map(lambda x: 0 if pd.isna(x) else max(int(x // 7) - 1, 0))
    weekly["duplicate_start"] = starts.duplicated().astype(int)
    weekly["ordered_after_previous"] = np.where(delta.isna(), True, delta > 0).astype(int)
    weekly["interval_days_inclusive"] = (ends - starts).dt.days + 1
    weekly.to_csv(out / "climate_weekly_axis_audit.csv", index=False)
    weekly_summary = {"rows": len(axis), "min_date": axis.start_date.min(), "max_date": axis.start_date.max(),
                      "seven_day_steps": int(delta.eq(7).sum()), "missing_weeks": int(weekly.missing_weeks_before.sum()),
                      "duplicate_start_weeks": int(weekly.duplicate_start.sum()),
                      "out_of_order_steps": int((delta.dropna() <= 0).sum()),
                      "seven_day_intervals": int(weekly.interval_days_inclusive.eq(7).sum()),
                      "history_length": 52, "horizon_length": 4}
    write_json(out / "climate_weekly_axis_summary.json", weekly_summary)
    report = ("# Climate 数据与边界说明\n\n"
              f"固定任务 `{TASK}`，清洗后周频数值轴 {len(axis)} 行；四段分别保留 "
              + "、".join(f"{row['segment']} {row['retained_origins']} 个起点" for row in boundary) + "。"
              f"相邻 7 天间隔 {weekly_summary['seven_day_steps']} 次，缺周 {weekly_summary['missing_weeks']}，"
              f"重复周 {weekly_summary['duplicate_start_weeks']}。历史窗口 52 周、预测范围 4 周；"
              "所有起点满足冻结半开区间边界。\n\n"
              "选择包仅包含 train、calibration、decision 的特征和对应真值。测试包只交付来源特征、质量特征、缺失掩码、"
              "origin 与时间网格；不交付逐起点数值历史、频率派生特征或任何测试真值。"
              "这是严格防止通过重叠历史窗口反推测试真值的取舍，因此测试包尚不能单独运行完整数值预测。"
              "测试数值历史的分布行是封包前计算的聚合描述，Release 无法逐起点重新计算该行；其他安全特征与审计表可在包内重放。\n\n"
              "report/search 的来源标识指向 Time-MMD 文件，不是经核验的原始网页。当前原始 URL 未登记；"
              "`end_date` 仅作为可用时间代理，不能声称真实发布时间已核验，也不能由本审计证明绝无泄漏。"
              "分布漂移为描述性比较，不读取或比较测试真值；不代表因果影响或模型收益。\n")
    paper = out / "paper_ready/Climate数据与边界说明.md"
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text(report, encoding="utf-8")
    return ["climate_boundary_audit.json", "climate_feature_drift.csv", "climate_source_coverage.csv",
            "climate_source_evidence.csv", "climate_weekly_axis_audit.csv",
            "climate_weekly_axis_summary.json", "paper_ready/Climate数据与边界说明.md"]


def run(root: Path, out: Path, reference: bool) -> dict:
    root = root.resolve(strict=True)
    out = out.resolve()
    require(not out.is_relative_to(root), "output directory must be outside the Release")
    mapping, anchors = check_package(root)
    outputs = audit(root, out, mapping, anchors)
    actual = {name: digest(out / name) for name in outputs}
    expected_path = root / "EXPECTED_OUTPUTS.json"
    if not reference:
        require(expected_path.is_file(), "EXPECTED_OUTPUTS.json missing")
        expected = json.loads(expected_path.read_text(encoding="utf-8"))["files"]
        require(actual == expected, f"audit output hash mismatch: {sorted(set(actual) ^ set(expected))}")
    result = {"status": "PASS", "bundle_signature": BUNDLE, "task": TASK,
              "origin_count": len(mapping), "selection_fit_origins": int(mapping.segment.ne("test").sum()),
              "test_origins": int(mapping.segment.eq("test").sum()),
              "test_truth_read_for_audit": False, "test_numeric_history_distributed": False,
              "output_sha256": actual, "expected_output_hashes_checked": not reference}
    write_json(out / "replay_report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.package, args.out, args.reference), ensure_ascii=False, allow_nan=False))
    except (VerificationError, ValueError, KeyError, FileNotFoundError) as error:
        print(f"VERIFY FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)
