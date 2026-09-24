"""Derive source-boundary and paper tables from the packaged replay evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REVISION = "00281e2d86058286d5548b15a7670e8eda57ef62"
SCENARIOS = ("proxy", "conservative_lag")


def save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def build(root: Path, output: Path) -> None:
    facts = pd.read_csv(output / "derived_input/fact_corpus.csv", low_memory=False)
    facts["source_file"] = facts.source_file.fillna("").astype(str)
    facts["source_url"] = facts.source_url.fillna("").astype(str)
    facts["start_date"] = pd.to_datetime(facts.start_date, errors="coerce")
    facts["end_date"] = pd.to_datetime(facts.end_date, errors="coerce")
    spec = json.loads((root / "inputs/split_spec_v2.json").read_text(encoding="utf-8"))
    task_map = {t["domain"]: f"{t['domain']}_h{t['horizon']}_f{t['fold_id']}" for t in spec["tasks"]}
    facts = facts.loc[facts.domain.isin(task_map)].copy()

    inventories = []
    missing = []
    for (domain, source), group in facts.groupby(["domain", "source"], sort=True):
        n = len(group)
        files = sorted(set(x for x in group.source_file if x))
        if len(files) > 1:
            raise ValueError(f"Multiple source files for {domain}/{source}: {files}")
        file_empty = int(group.source_file.eq("").sum())
        url_empty = int(group.source_url.eq("").sum())
        date_proxy = n  # No independently verified per-record publication timestamp exists.
        inventories.append({
            "domain": domain, "source": source, "source_file": files[0] if files else "",
            "fact_count": n, "start_min": group.start_date.min().date().isoformat(),
            "end_max": group.end_date.max().date().isoformat(),
            "original_url_count": n - url_empty,
            "publication_time_status": "end_date_proxy_unverified",
            "upstream_revision": REVISION,
        })
        missing.append({
            "task_id": task_map[domain], "domain": domain, "source": source,
            "fact_count_denominator": n,
            "source_file_missing_count": file_empty,
            "source_file_missing_rate": file_empty / n,
            "original_url_missing_count": url_empty,
            "original_url_missing_rate": url_empty / n,
            "publication_time_proxy_count": date_proxy,
            "publication_time_proxy_rate": date_proxy / n,
        })
    inventory = pd.DataFrame(inventories)
    missing_df = pd.DataFrame(missing)
    save(inventory, output / "source_tables/source_evidence_inventory.csv")
    save(missing_df, output / "source_tables/missing_source_evidence.csv")

    sample_segments = {}
    sample_sizes = {}
    for task_id in sorted(task_map.values()):
        samples = pd.read_csv(root / "inputs/formal_bundle" / task_id / "samples.csv")
        retained = samples.loc[samples.reason.eq("retained")]
        sample_segments.update(dict(zip(retained.origin_id, retained.segment)))
        sample_sizes[task_id] = len(retained)

    origin_rows = []
    social = {}
    with (output / "origin_time_audit.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            oid = row["origin_id"]
            if oid not in sample_segments:
                raise ValueError(f"Origin missing from retained samples: {oid}")
            selected = row["selected_ids"]
            count = len(selected["report"]) + len(selected["search"])
            origin_rows.append({
                "task_id": row["task_id"], "segment": sample_segments[oid],
                "scenario": row["scenario"], "origin_id": oid,
                "text_available": bool(row["text_available_bundle"]),
                "selected_fact_count": count,
            })
            if row["task_id"].startswith("SocialGood"):
                social.setdefault(oid, {})[row["scenario"]] = {
                    "selected": set(selected["report"] + selected["search"]),
                    "available": bool(row["text_available_bundle"]),
                    "segment": sample_segments[oid],
                }

    origins = pd.DataFrame(origin_rows)
    coverage = []
    for (task_id, segment, scenario), group in origins.groupby(["task_id", "segment", "scenario"], sort=True):
        counts = group.selected_fact_count.to_numpy(dtype=float)
        coverage.append({
            "task_id": task_id, "segment": segment, "scenario": scenario,
            "origins_denominator": len(group), "text_available_origins": int(group.text_available.sum()),
            "text_available_rate": float(group.text_available.mean()),
            "selected_facts_total": int(counts.sum()),
            "selected_facts_q1": float(np.percentile(counts, 25)),
            "selected_facts_median": float(np.percentile(counts, 50)),
            "selected_facts_q3": float(np.percentile(counts, 75)),
        })
    coverage_df = pd.DataFrame(coverage)
    save(coverage_df, output / "source_tables/coverage_by_segment_scenario.csv")

    if len(social) != sample_sizes[task_map["SocialGood"]] or any(set(v) != set(SCENARIOS) for v in social.values()):
        raise ValueError("SocialGood scenario pairing is incomplete")
    social_rows = []
    for segment in ("train", "calibration", "decision", "test", "all"):
        pairs = [v for v in social.values() if segment == "all" or v["proxy"]["segment"] == segment]
        changed = [v for v in pairs if v["proxy"]["selected"] != v["conservative_lag"]["selected"]]
        flips = [v for v in pairs if v["proxy"]["available"] and not v["conservative_lag"]["available"]]
        social_rows.append({
            "task_id": task_map["SocialGood"], "segment": segment,
            "origins_denominator": len(pairs), "selected_set_changed_origins": len(changed),
            "selected_set_changed_rate": len(changed) / len(pairs),
            "coverage_true_to_false_origins": len(flips),
            "coverage_true_to_false_rate": len(flips) / len(pairs),
            "changed_but_still_available_origins": sum(v["conservative_lag"]["available"] for v in changed),
        })
    social_df = pd.DataFrame(social_rows)
    save(social_df, output / "source_tables/socialgood_scenario_difference_summary.csv")

    boundaries = []
    audit_coverage = pd.read_csv(output / "coverage_summary.csv")
    for task in spec["tasks"]:
        task_id = task_map[task["domain"]]
        sources = inventory.loc[inventory.domain.eq(task["domain"])]
        boundaries.append({
            "task_id": task_id, "domain": task["domain"], "numerical_rows": task["n_rows_expected"],
            "retained_origins": sample_sizes[task_id], "canonical_facts": int(sources.fact_count.sum()),
            "report_facts": int(sources.loc[sources.source.eq("report"), "fact_count"].sum()),
            "search_facts": int(sources.loc[sources.source.eq("search"), "fact_count"].sum()),
            "registered_original_url_count": int(sources.original_url_count.sum()),
            "upstream_revision": REVISION,
        })
    save(pd.DataFrame(boundaries), output / "paper_ready/表_数据来源与样本边界.csv")
    time_rows = []
    for _, row in audit_coverage.iterrows():
        time_rows.append({
            "task_id": row.task_id, "scenario": row.scenario,
            "origins_denominator": int(row.origins), "text_available_origins": int(row.text_available),
            "text_available_rate": float(row.text_available_rate),
            "selected_facts_total": int(row.selected_report_total + row.selected_search_total),
            "mask_mismatches": int(row.mask_mismatches), "trace_mismatches": int(row.trace_mismatches),
            "quality_max_abs_error": float(row.quality_max_abs_error),
        })
    save(pd.DataFrame(time_rows), output / "paper_ready/表_时间边界与覆盖审计.csv")
    save(missing_df, output / "paper_ready/表_来源证据缺口.csv")

    all_row = social_df.loc[social_df.segment.eq("all")].iloc[0]
    note = (
        "# 数据方法与限制说明\n\n"
        "本表以 Time-MMD 固定 revision `" + REVISION + "` 和正式 Bundle 的四项任务为范围。"
        "规范事实以 `text_id` 与保留的血缘记录一对一合并；`fact_count` 的统计单位是一条规范事实。"
        "预测起点的统计单位是一条保留的 `origin_id`。\n\n"
        "按冻结规则，文本事实 `end_date < cutoff` 才可进入 proxy 情景；保守滞后情景还要求 "
        "`end_date + lag_days < cutoff`。每来源按结束日期倒序和 `text_id` 顺序最多保留 32 条。"
        "`end_date` 是可用时间代理，未被核验为真实发布时间；当前规范事实没有已登记的原始 URL。\n\n"
        "四任务两情景的保留起点、`text_trace`、来源掩码和质量值按冻结规则重算一致。"
        f"SocialGood 有 {int(all_row.selected_set_changed_origins)}/{int(all_row.origins_denominator)} 个起点的选中事实集合变化，"
        f"其中 {int(all_row.coverage_true_to_false_origins)} 个起点从文本可用变为不可用。"
        "集合变化与覆盖翻转是不同统计量。\n\n"
        "本审计不能核验全部事实的真实发布时间，也不能证明不存在任何时间泄漏。"
        "来源标签只指向 Time-MMD 文件，不代表已追溯到原始发布网页或机构。"
        "`selected_facts_total` 对起点求和，同一规范事实可在多个起点重复计入；不得解释为不同事实总数。\n"
    )
    (output / "paper_ready/数据方法与限制说明.md").write_text(note, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit("Run through replay.py so input hashes and corpus linkage are checked first")
