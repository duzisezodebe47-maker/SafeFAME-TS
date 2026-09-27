"""Fail-closed verification and replay of the SocialGood safe input Release."""

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
TASK = "SocialGood_h3_f1"
BUNDLE_SHA = "a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38"
SPEC_SHA = "a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031"
FORMAL_MANIFEST_SHA = "6c889af1a687592453a5d1d4d83dc2b4eaf9118653f1ac1085b1997c91800143"
SEGMENTS = ("train", "calibration", "decision", "test")
SCENARIOS = ("proxy", "conservative_lag")
SAFE_SHARED = ("origin_index.npy", "horizon_time.npy", "numeric_missing.npy")
SAFE_TEXT = ("semantic.npy", "quality.npy", "source_available.npy", "text_available.npy")
FIT_ONLY = ("numeric_history.npy", "numeric_history_raw.npy", "frequency.npy",
            "targets.npy", "targets_raw.npy", "targets_standardized.npy")
OUTPUTS = ("socialgood_segment_coverage.csv", "socialgood_scenario_flip_detail.csv",
           "socialgood_degradation_audit.json", "socialgood_source_evidence.csv",
           "paper_ready/SocialGood数据边界与退化说明.md", "socialgood_origin_coverage.csv")


class VerifyError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> np.ndarray:
    require(path.is_file(), f"missing array: {path}")
    return np.load(path, allow_pickle=False)


def put_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def same_array(left: np.ndarray, right: np.ndarray) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype and \
        np.ascontiguousarray(left).tobytes() == np.ascontiguousarray(right).tobytes()


def check(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    require(root.drive.upper() == "D:", "Release must be on D drive")
    test_dir = root / "test_features"
    allowed = set(SAFE_SHARED) | {"metadata.csv"} | {
        f"{scenario}/{name}" for scenario in SCENARIOS for name in SAFE_TEXT}
    actual_test = {f.relative_to(test_dir).as_posix() for f in test_dir.rglob("*") if f.is_file()}
    require(actual_test == allowed, f"test directory whitelist violation: {sorted(actual_test ^ allowed)}")
    require(not any(re.search(r"target|truth|label|numeric_history|frequency", name, re.I)
                    for name in actual_test), "test truth/numeric/frequency file forbidden")
    manifest_file = root / "MANIFEST.json"
    require(manifest_file.is_file(), "MANIFEST.json missing")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    require(manifest["bundle_signature"] == BUNDLE_SHA and manifest["split_spec_sha256"] == SPEC_SHA,
            "Release manifest anchors mismatch")
    listed = set()
    for entry in manifest["files"]:
        relative = entry["relative_path"]
        parts = PurePosixPath(relative)
        require(not parts.is_absolute() and ".." not in parts.parts and relative not in listed,
                f"bad manifest path: {relative}")
        file = root / relative
        require(file.is_file() and file.stat().st_size == entry["bytes"] and sha(file) == entry["sha256"],
                f"manifest bytes/SHA256 mismatch: {relative}")
        listed.add(relative)
    actual = {f.relative_to(root).as_posix() for f in root.rglob("*") if f.is_file()
              and "__pycache__" not in f.parts and f.suffix != ".pyc"}
    require(actual == listed | {"MANIFEST.json"},
            f"Release inventory mismatch: extra={sorted(actual-listed-{'MANIFEST.json'})}; missing={sorted(listed-actual)}")

    anchors = json.loads((root / "source_anchors.json").read_text(encoding="utf-8"))
    require(anchors["task"] == TASK and anchors["bundle_signature"] == BUNDLE_SHA
            and anchors["split_spec_sha256"] == SPEC_SHA, "frozen Bundle signature or spec anchor mismatch")
    formal_path = root / "anchors/formal_bundle_manifest.json"
    require(sha(formal_path) == FORMAL_MANIFEST_SHA
            and anchors["formal_bundle_manifest_sha256"] == FORMAL_MANIFEST_SHA,
            "formal Bundle manifest anchor mismatch")
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    require(formal["bundle_signature"] == BUNDLE_SHA, "formal Bundle signature mismatch")
    spec_path = root / "anchors/split_spec_v2.json"
    require(sha(spec_path) == SPEC_SHA, "frozen spec SHA256 mismatch")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    task_spec = next(t for t in spec["tasks"] if t["domain"] == "SocialGood")
    require((task_spec["input_len"], task_spec["horizon"], task_spec["bounds"],
             spec["seasonal_periods"]["SocialGood"]) == (24, 3, [366, 458, 641, 732], 12),
            "frozen task parameters mismatch")
    require(anchors["numeric_snapshot_sha256"] == task_spec["numerical_sha256"],
            "numeric source snapshot anchor mismatch")
    for relative, expected in anchors["source_safe_sha256"].items():
        original = "target_time.npy" if relative == "horizon_time.npy" else relative
        require(expected == formal["files"][f"{TASK}/{original}"]
                and sha(root / "source_safe" / relative) == expected,
                f"safe feature differs from frozen Bundle: {relative}")
    for scenario, expected in anchors["text_trace_sha256"].items():
        require(expected == formal["files"][f"{TASK}/{scenario}/text_trace.jsonl"]
                and sha(root / "text_trace" / f"{scenario}.jsonl") == expected,
                f"text trace differs from frozen Bundle: {scenario}")
    truth = json.loads((root / "truth_commitment.json").read_text(encoding="utf-8"))
    require(set(truth) == {"task", "bundle_signature", "split_spec_sha256", "source_file",
                            "formal_full_sha256", "test_slice_sha256", "test_shape",
                            "test_dtype", "test_origins"}, "truth commitment has unexpected data")
    require(truth["task"] == TASK and truth["bundle_signature"] == BUNDLE_SHA
            and truth["split_spec_sha256"] == SPEC_SHA and truth["source_file"] == f"{TASK}/targets_raw.npy"
            and truth["formal_full_sha256"] == formal["files"][truth["source_file"]]
            and truth["test_shape"] == [89, 3] and truth["test_dtype"] == "float32"
            and truth["test_origins"] == 89 and re.fullmatch(r"[0-9a-f]{64}", truth["test_slice_sha256"]),
            "truth commitment anchor/shape mismatch")

    mapping = pd.read_csv(root / "mapping.csv", keep_default_na=False)
    sample_audit = pd.read_csv(root / "sample_audit.csv", keep_default_na=False)
    axis = pd.read_csv(root / "monthly_axis.csv", keep_default_na=False)
    require(len(axis) == task_spec["n_rows_expected"] and np.array_equal(axis.numerical_row, np.arange(len(axis))),
            "monthly time axis mismatch")
    require(len(mapping) == 700 and mapping.origin_id.is_unique and mapping.origin_index.is_unique
            and np.array_equal(mapping.source_bundle_row, np.arange(len(mapping))),
            "mapping Bundle row/origin mismatch")
    bounds = [0, *task_spec["bounds"]]
    calculated = {segment: len(range(max(lo, 24), hi - 3 + 1))
                  for segment, lo, hi in zip(SEGMENTS, bounds[:-1], bounds[1:])}
    require(mapping.segment.value_counts().to_dict() == calculated
            and anchors["computed_retained_by_segment"] == calculated,
            "four-segment retained count mismatch")
    candidates = {segment: len(range(max(lo, 24), hi))
                  for segment, lo, hi in zip(SEGMENTS, bounds[:-1], bounds[1:])}
    require(sample_audit.segment.value_counts().to_dict() == candidates
            and len(sample_audit) == sum(candidates.values()), "sample audit candidate count mismatch")
    require(sample_audit.reason.value_counts().to_dict() == {"retained": 700, "target_crosses_boundary": 8},
            "sample audit exclusion reasons mismatch")
    source_origin = load(root / "source_safe/origin_index.npy")
    source_time = load(root / "source_safe/horizon_time.npy")
    require(source_time.shape == (700, 3) and np.array_equal(source_origin, mapping.origin_index),
            "formal Bundle origin/time shape mismatch")
    for row in mapping.itertuples():
        segment_index = SEGMENTS.index(row.segment)
        lo, hi = bounds[segment_index:segment_index + 2]
        require(max(lo, 24) <= row.origin_index and row.origin_index + 3 <= hi
                and row.target_end_index == row.origin_index + 3
                and (row.segment_lo, row.segment_hi) == (lo, hi),
                f"origin/target crosses segment boundary: {row.origin_id}")
        require(row.cutoff_time[:10] == axis.start_date.iloc[row.origin_index]
                and row.history_start[:10] == axis.start_date.iloc[row.origin_index - 24],
                f"cutoff/history grid mismatch: {row.origin_id}")
        expected = axis.start_date.iloc[row.origin_index:row.origin_index + 3].tolist()
        require([str(v)[:10] for v in source_time[row.source_bundle_row]] == expected,
                f"horizon grid mismatch: {row.origin_id}")
    for partition, mask in (("selection_fit", mapping.segment.ne("test")),
                            ("test_features", mapping.segment.eq("test"))):
        rows = np.flatnonzero(mask.to_numpy())
        subset = mapping.loc[mask].reset_index(drop=True)
        require(np.array_equal(subset.package_row, np.arange(len(rows)))
                and subset.package_partition.eq(partition).all(), f"{partition} row mapping mismatch")
        metadata = pd.read_csv(root / partition / "metadata.csv", keep_default_na=False)
        require(metadata.origin_id.tolist() == subset.origin_id.tolist()
                and metadata.segment.tolist() == subset.segment.tolist(),
                f"{partition} metadata/mapping mismatch")
        if partition == "test_features":
            expected_columns = [column for column in mapping.columns if column != "source_bundle_row"]
            require(metadata.segment.eq("test").all() and metadata.columns.tolist() == expected_columns,
                    "test metadata contains forbidden field or non-test row")
        for name in SAFE_SHARED:
            require(same_array(load(root / partition / name), load(root / "source_safe" / name)[rows]),
                    f"slice mismatch: {partition}/{name}")
        for scenario in SCENARIOS:
            for name in SAFE_TEXT:
                relative = f"{scenario}/{name}"
                require(same_array(load(root / partition / relative), load(root / "source_safe" / relative)[rows]),
                        f"slice mismatch: {partition}/{relative}")
    commitments = json.loads((root / "selection_fit_commitments.json").read_text(encoding="utf-8"))
    require(set(commitments) == set(FIT_ONLY), "selection commitment inventory mismatch")
    for name in FIT_ONLY:
        file = root / "selection_fit" / name
        array = load(file)
        record = commitments[name]
        require(record["formal_full_sha256"] == formal["files"][f"{TASK}/{name}"]
                and sha(file) == record["selection_slice_sha256"]
                and list(array.shape) == record["shape"] and str(array.dtype) == record["dtype"]
                and len(array) == 611, f"selection slice commitment mismatch: {name}")
    return mapping, sample_audit, anchors


def audit(root: Path, out: Path, mapping: pd.DataFrame, sample_audit: pd.DataFrame,
          anchors: dict) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    traces = {scenario: [json.loads(line) for line in (root / "text_trace" / f"{scenario}.jsonl")
                         .read_text(encoding="utf-8").splitlines()] for scenario in SCENARIOS}
    for scenario in SCENARIOS:
        require(len(traces[scenario]) == len(mapping)
                and [t["origin_id"] for t in traces[scenario]] == mapping.origin_id.tolist(),
                f"trace origin order mismatch: {scenario}")
    coverage = []
    origin_coverage = []
    for scenario in SCENARIOS:
        for segment in SEGMENTS:
            rows = np.flatnonzero(mapping.segment.eq(segment).to_numpy())
            group = mapping.iloc[rows]
            partition = "test_features" if segment == "test" else "selection_fit"
            available = load(root / partition / scenario / "text_available.npy")[group.package_row.to_numpy()]
            source = load(root / partition / scenario / "source_available.npy")[group.package_row.to_numpy()]
            selected_report = sum(len(traces[scenario][i]["report_ids"]) for i in rows)
            selected_search = sum(len(traces[scenario][i]["search_ids"]) for i in rows)
            require(all(bool(available[j]) == bool(traces[scenario][i]["report_ids"]
                            or traces[scenario][i]["search_ids"]) for j, i in enumerate(rows)),
                    f"text mask/trace mismatch: {scenario}/{segment}")
            require(np.array_equal(source.any(axis=1), available),
                    f"source mask/text mask mismatch: {scenario}/{segment}")
            for position, source_row in enumerate(rows):
                trace = traces[scenario][source_row]
                origin_coverage.append({"scenario": scenario, "segment": segment,
                                        "origin_id": group.iloc[position].origin_id,
                                        "origin_index": int(group.iloc[position].origin_index),
                                        "text_available": int(available[position]),
                                        "report_selected": len(trace["report_ids"]),
                                        "search_selected": len(trace["search_ids"]),
                                        "selected_source_uses": len(trace["report_ids"]) + len(trace["search_ids"]),
                                        "report_source_ids": ";".join(trace["report_ids"]),
                                        "search_source_ids": ";".join(trace["search_ids"])})
            coverage.append({"scenario": scenario, "segment": segment,
                             "candidate_origins": int(sample_audit.segment.eq(segment).sum()),
                             "retained_origins": len(rows), "text_available_origins": int(available.sum()),
                             "text_available_rate": float(available.mean()),
                             "report_selected_uses": selected_report, "search_selected_uses": selected_search,
                             "selected_source_uses": selected_report + selected_search,
                             "distinct_source_ids": len({item for i in rows for key in ("report_ids", "search_ids")
                                                          for item in traces[scenario][i][key]})})
    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(out / OUTPUTS[0], index=False)
    pd.DataFrame(origin_coverage).to_csv(out / OUTPUTS[5], index=False)

    lineage = pd.read_csv(root / "source_lineage_slim.csv", keep_default_na=False)
    canonical = lineage.loc[lineage.reason.eq("retained")].copy()
    require(len(canonical) == 4724 and canonical.canonical_text_id.is_unique,
            "canonical SocialGood source identity mismatch")
    evidence = canonical[["canonical_text_id", "source", "source_file", "source_url",
                          "start_date", "end_date", "publication_verified"]].rename(
                              columns={"canonical_text_id": "source_id", "source_url": "original_url"})
    evidence["availability_time_proxy"] = evidence.end_date
    evidence["proxy_is_verified_publication"] = False
    evidence.to_csv(out / OUTPUTS[3], index=False)
    source_by_id = evidence.set_index("source_id")

    flips = []
    changed_sets = 0
    for row in mapping.itertuples():
        proxy = traces["proxy"][row.source_bundle_row]
        lag = traces["conservative_lag"][row.source_bundle_row]
        p_ids = set(proxy["report_ids"] + proxy["search_ids"])
        l_ids = set(lag["report_ids"] + lag["search_ids"])
        changed_sets += p_ids != l_ids
        if bool(p_ids) != bool(l_ids):
            require(bool(p_ids) and not l_ids, f"unexpected coverage flip: {row.origin_id}")
            removed = sorted(p_ids - l_ids)
            require(set(removed).issubset(source_by_id.index), "flip source ID absent from evidence")
            cutoff = pd.Timestamp(row.cutoff_time)
            require(all(pd.Timestamp(source_by_id.loc[i, "end_date"]) <= cutoff
                        < pd.Timestamp(source_by_id.loc[i, "end_date"]) + pd.Timedelta(days=31)
                        for i in removed), "coverage flip does not satisfy lag trigger")
            part = "test_features" if row.segment == "test" else "selection_fit"
            qi = row.package_row
            flips.append({"origin_id": row.origin_id, "origin_index": row.origin_index,
                          "segment": row.segment, "cutoff_time": row.cutoff_time,
                          "proxy_selected": len(p_ids), "lag_selected": len(l_ids),
                          "removed_source_ids": ";".join(removed),
                          "removed_end_dates": ";".join(str(source_by_id.loc[i, "end_date"]) for i in removed),
                          "trigger_rule": "end_date <= cutoff but end_date + 31 days > cutoff",
                          "publication_time_status": proxy["publication_time_status"],
                          "proxy_quality": json.dumps(load(root / part / "proxy/quality.npy")[qi].tolist()),
                          "lag_quality": json.dumps(load(root / part / "conservative_lag/quality.npy")[qi].tolist())})
    flip_df = pd.DataFrame(flips)
    require(len(flips) == 1 and flips[0]["origin_id"] == "SocialGood:h3:f1:o385"
            and changed_sets == 341, "scenario flip/set-change count mismatch")
    flip_df.to_csv(out / OUTPUTS[1], index=False)
    zero_vectors = {}
    for scenario in SCENARIOS:
        mask = load(root / "source_safe" / scenario / "text_available.npy")
        semantic = load(root / "source_safe" / scenario / "semantic.npy")
        quality = load(root / "source_safe" / scenario / "quality.npy")
        source = load(root / "source_safe" / scenario / "source_available.npy")
        no_text = ~mask
        require(np.all(semantic[no_text] == 0) and np.all(source[no_text] == 0)
                and np.all(quality[no_text] == quality[no_text][0]),
                f"no-text feature sentinel inconsistent: {scenario}")
        zero_vectors[scenario] = {"no_text_origins": int(no_text.sum()),
                                  "semantic_all_zero": True, "source_available": [False, False],
                                  "text_available": False, "quality_vector": quality[no_text][0].tolist()}
    require(all(coverage_df.loc[(coverage_df.scenario == scenario) &
                                  (coverage_df.segment == "train"), "text_available_origins"].iloc[0] == 0
                for scenario in SCENARIOS), "train text is not zero under origin-level definition")
    degradation = {"task": TASK, "bundle_signature": BUNDLE_SHA, "split_spec_sha256": SPEC_SHA,
                   "unit": "retained forecast origin", "bounds_half_open": [366, 458, 641, 732],
                   "input_len": 24, "horizon": 3, "seasonal_period": 12,
                   "candidate_origins_by_segment": sample_audit.segment.value_counts().reindex(SEGMENTS).to_dict(),
                   "retained_origins_by_segment": mapping.segment.value_counts().reindex(SEGMENTS).to_dict(),
                   "train_zero_text_definition": "among all 340 retained train forecast origins, no report/search fact selected by either scenario",
                   "scenario_changed_selected_sets": int(changed_sets), "coverage_flips": len(flips),
                   "no_text_features": zero_vectors,
                   "safe_degradation_interpretation": "Semantic vector is zero and availability masks false; quality is a nonzero missingness sentinel. A text contribution is absent only if downstream model correctly masks it. This audit does not run a model or assert exact numeric-only predictions.",
                   "source_original_url_missing": int(evidence.original_url.eq("").sum()),
                   "source_total": len(evidence), "verified_publication_times": int(
                       evidence.publication_verified.astype(str).str.lower().eq("true").sum()),
                   "time_boundary": "end_date is an availability proxy, not verified publication time",
                   "test_truth_read_for_audit": False, "test_numeric_history_distributed": False}
    put_json(out / OUTPUTS[2], degradation)
    report = ("# SocialGood 数据边界与退化说明\n\n"
              "固定任务 `SocialGood_h3_f1`，数值轴 916 个按月记录；历史 24、预测 3、季节周期 12。"
              "四段半开边界 `[0,366)、[366,458)、[458,641)、[641,732)`，候选起点 "
              "342/92/183/91，完整目标窗口筛选后保留 340/90/181/89，总计 700。"
              "选择侧 611、测试侧 89；统计单位为预测起点，非独立文本或月记录。\n\n"
              "两种情景在 train 段均为 0/340 起点有选中文本。语义向量全零、来源与文本可用掩码均为假，"
              "但质量向量并非全零，而是缺失哨兵值 `[0,0,0,0,1,1,1,1,0,0]`。"
              "因此只有下游模型确实遵守掩码时，文本贡献才安全关闭；本交付没有运行模型，不能声称已证实预测退化等同纯数值。\n\n"
              "`proxy` 与 `conservative_lag` 的唯一覆盖翻转为校准段 `SocialGood:h3:f1:o385`："
              "前者选中 7 条 search 事实，后者为 0。保守情景在 `end_date` 代理后追加 31 天，"
              "这些事实在该起点尚未达到滞后门槛。两情景的选中集合还在 341/700 起点不同，"
              "其中 340 个虽改变但仍有文本。覆盖表逐段给出分子、分母及重复来源使用次数。\n\n"
              "所有 4724 条规范事实的原始 URL 当前均为空，真实发布时间均未独立核验。"
              "`end_date` 只是可得时间代理，不可解释为真实发布日期。"
              "测试包不含目标、逐起点数值历史或频率，防止重叠窗口反推出测试真值；"
              "因此它是安全输入子集，不是可单独运行完整预测的全特征包。"
              "正式 Bundle 在其他位置可能含测试真值，本隔离仅约束此 Release。"
              "未接触测试评分、未训练或运行 SocialGood 模型，也不由本审计推断文本增益或因果效果。\n")
    paper = out / OUTPUTS[4]
    paper.parent.mkdir(parents=True, exist_ok=True)
    paper.write_text(report, encoding="utf-8")
    return {"coverage": coverage_df.to_dict(orient="records"), "flips": len(flips)}


def run(root: Path, out: Path, reference: bool) -> dict:
    root = root.resolve(strict=True)
    out = out.resolve()
    require(not out.is_relative_to(root), "output must be outside Release")
    mapping, sample_audit, anchors = check(root)
    data = audit(root, out, mapping, sample_audit, anchors)
    hashes = {relative: sha(out / relative) for relative in OUTPUTS}
    if not reference:
        expected = json.loads((root / "EXPECTED_OUTPUTS.json").read_text(encoding="utf-8"))["files"]
        require(hashes == expected, "replayed audit output SHA256 mismatch")
        for relative in OUTPUTS:
            require(sha(root / relative) == expected[relative], f"published audit file mismatch: {relative}")
    result = {"status": "PASS", "task": TASK, "bundle_signature": BUNDLE_SHA,
              "retained_origins": len(mapping), "selection_origins": 611, "test_origins": 89,
              "coverage": data["coverage"], "coverage_flips": data["flips"],
              "negative_test_scope": "run separately", "test_truth_read_for_audit": False,
              "output_hashes_checked": not reference, "output_sha256": hashes}
    put_json(out / "replay_report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.package, args.out, args.reference), ensure_ascii=False, allow_nan=False))
    except (VerifyError, ValueError, KeyError, FileNotFoundError, TypeError, IndexError) as error:
        print(f"VERIFY FAIL: {error}", file=sys.stderr)
        raise SystemExit(2)
