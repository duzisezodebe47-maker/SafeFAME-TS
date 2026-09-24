"""Replay the frozen source/time audit using only this release directory.

Requires Python 3.12, NumPy and pandas. All data and project code are packaged
beside this file; no repository checkout or cached audit directory is read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True


FROZEN_BUNDLE = "a69821be115445265cdec0f005686f978de7afb46700edb075dd47aaeee86d38"
FROZEN_SPEC = "a0947a5b64d2a609e624c432ef6c6ce9faa6ae8ab86570d4c90594f70c6f0031"
RAW_CORPUS_SHA = "c07126f36c05278486b1379a7c4c9be2e1c2a890214802bfccca7b517e7d7ffb"
LINEAGE_SHA = "90603c6e7fd62dc0f7b8d0ad80c93edea17a4a479afd536ef3668724ff01e9d1"
REVISION = "00281e2d86058286d5548b15a7670e8eda57ef62"
CORPUS_COLUMNS = ("domain", "source", "start_date", "end_date", "fact", "text_id", "characters", "future_language_flag")
LINEAGE_COLUMNS = ("domain", "source", "canonical_text_id", "reason", "source_file", "source_url", "publication_verified")


class ReplayError(ValueError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inside(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ReplayError(f"Input path escapes Release: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ReplayError(f"Input path escapes Release: {relative}")
    if not path.is_file():
        raise ReplayError(f"Required Release input missing: {relative}")
    return path


def required_columns(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ReplayError(f"{label} missing required columns: {missing}")


def validate_corpus(corpus: pd.DataFrame) -> None:
    required_columns(corpus, CORPUS_COLUMNS, "fact_corpus.csv")
    if corpus.text_id.isna().any() or corpus.text_id.duplicated().any():
        raise ReplayError("Canonical facts contain missing or duplicate text_id")
    if set(corpus.source) - {"report", "search"}:
        raise ReplayError("Canonical facts contain an unregistered source class")


def verify_inputs(root: Path) -> dict:
    manifest_path = inside(root, "MANIFEST.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = inside(root, item["relative_path"])
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ReplayError(f"Release input hash/size mismatch: {item['relative_path']}")
    corpus_path = inside(root, "inputs/fact_corpus.csv")
    lineage_path = inside(root, "inputs/text_lineage.csv")
    spec_path = inside(root, "inputs/split_spec_v2.json")
    bundle_manifest = inside(root, "inputs/formal_bundle/manifest.json")
    for path, expected in ((corpus_path, RAW_CORPUS_SHA), (lineage_path, LINEAGE_SHA), (spec_path, FROZEN_SPEC)):
        if digest(path) != expected:
            raise ReplayError(f"Frozen SHA256 mismatch: {path.relative_to(root)}")
    bm = json.loads(bundle_manifest.read_text(encoding="utf-8"))
    if bm.get("bundle_signature") != FROZEN_BUNDLE or bm["inputs"]["split_spec_sha256"] != FROZEN_SPEC:
        raise ReplayError("Formal Bundle signature or split spec differs from frozen input")
    corpus = pd.read_csv(corpus_path, low_memory=False)
    lineage = pd.read_csv(lineage_path, low_memory=False)
    validate_corpus(corpus)
    required_columns(lineage, LINEAGE_COLUMNS, "text_lineage.csv")
    return {"manifest": manifest, "corpus": corpus, "lineage": lineage}


def enrich_corpus(corpus: pd.DataFrame, lineage: pd.DataFrame, out_dir: Path) -> Path:
    retained = lineage.loc[lineage.reason.eq("retained"), ["canonical_text_id", "source_file", "source_url", "publication_verified"]].copy()
    if retained.canonical_text_id.isna().any() or retained.canonical_text_id.duplicated().any():
        raise ReplayError("Retained lineage has missing or duplicate canonical_text_id")
    merged = corpus.merge(retained, left_on="text_id", right_on="canonical_text_id", how="left", validate="one_to_one", indicator=True)
    if not merged._merge.eq("both").all():
        raise ReplayError("Every canonical fact must match one retained lineage row")
    merged = merged.drop(columns=["canonical_text_id", "_merge"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fact_corpus.csv"
    merged.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ReplayError(f"Cannot import packaged code: {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(root: Path, output_dir: Path, *, check_expected: bool = True) -> dict:
    started = time.monotonic()
    inputs = verify_inputs(root)
    # The output may be anywhere; every *input* was resolved under root above.
    input_dir = output_dir / "derived_input"
    enrich_corpus(inputs["corpus"], inputs["lineage"], input_dir)
    audit = load_module(inside(root, "code/audit_source_time_boundaries.py"), "source_time_audit_v6")
    result = audit.run_audit(root / "inputs/formal_bundle", input_dir, root / "inputs/split_spec_v2.json", output_dir)
    old_negative = audit.run_negative_tests(audit.load_facts(input_dir), output_dir / "negative_cases_v6.json")
    result["negative_cases_v6"] = old_negative
    result["status"] = "PASS" if result["status"] == old_negative["status"] == "PASS" else "FAIL"
    result["audit_folder"] = "derived_input"
    result["origin_detail"] = "origin_time_audit.jsonl"
    result["upstream_revision"] = REVISION
    audit.json_dump(output_dir / "audit_summary.json", result)
    tables = load_module(inside(root, "code/build_tables.py"), "build_tables_v7")
    tables.build(root, output_dir)
    output_names = [
        "audit_summary.json", "origin_time_audit.jsonl", "fact_catalog.csv", "coverage_summary.csv",
        "socialgood_differences.csv", "negative_cases_v6.json", "derived_input/fact_corpus.csv",
        "source_tables/source_evidence_inventory.csv", "source_tables/missing_source_evidence.csv",
        "source_tables/coverage_by_segment_scenario.csv", "source_tables/socialgood_scenario_difference_summary.csv",
        "paper_ready/表_数据来源与样本边界.csv", "paper_ready/表_时间边界与覆盖审计.csv",
        "paper_ready/表_来源证据缺口.csv", "paper_ready/数据方法与限制说明.md",
    ]
    actual = {name: digest(output_dir / name) for name in output_names}
    expected_path = root / "EXPECTED_OUTPUTS.json"
    if check_expected and expected_path.exists():
        expected = json.loads(expected_path.read_text(encoding="utf-8"))["files"]
        if actual != expected:
            differing = sorted(name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name))
            raise ReplayError(f"Clean replay output hashes differ: {differing}")
    report = {
        "status": result["status"],
        "bundle_signature": FROZEN_BUNDLE,
        "spec_sha256": FROZEN_SPEC,
        "origin_records": sum(1 for _ in (output_dir / "origin_time_audit.jsonl").open(encoding="utf-8")),
        "output_sha256": actual,
        "expected_hashes_checked": bool(check_expected and expected_path.exists()),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (output_dir / "replay_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise ReplayError("Start-level audit failed; inspect audit_summary.json")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("replay_output"))
    parser.add_argument("--build-reference", action="store_true", help="first package build only")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        report = run(root, args.output_dir.resolve(), check_expected=not args.build_reference)
    except (ReplayError, OSError, KeyError, ValueError) as exc:
        print(f"REPLAY FAIL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
