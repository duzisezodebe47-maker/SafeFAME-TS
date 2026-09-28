"""Recompute the SocialGood raw-to-clean audit from frozen Time-MMD inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


RAW_SHA = {
    "numerical": "cfa785d875d9cd826115c24eedb2747e6f762a61c1af8b258f00769a09470387",
    "report": "6d6a9efb2a885511d08fb45a7adce4b1b0c53fba81a31d3f4d62183320512fa0",
    "search": "6a3700b4b499f9f55dfab1aaca87ff0d944b98baade97a3d6e99f2cf41edbf18",
}
CLEAN_SHA = "f25a9a64ec0e21dc8e8f176565a12c366296e556d9ababe808c46273d073869d"
LINEAGE_SHA = "90603c6e7fd62dc0f7b8d0ad80c93edea17a4a479afd536ef3668724ff01e9d1"
SOURCE_REVISION = "00281e2d86058286d5548b15a7670e8eda57ef62"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(raw_root: Path, clean_path: Path, lineage_path: Path) -> dict:
    numerical_path = raw_root / "numerical/SocialGood/SocialGood.csv"
    sources = {name: raw_root / f"textual/SocialGood/SocialGood_{name}.csv"
               for name in ("report", "search")}
    for name, path in {"numerical": numerical_path, **sources}.items():
        if sha(path) != RAW_SHA[name]:
            raise ValueError(f"raw {name} SHA256 mismatch")
    if sha(clean_path) != CLEAN_SHA or sha(lineage_path) != LINEAGE_SHA:
        raise ValueError("clean numeric or text lineage SHA256 mismatch")

    raw = pd.read_csv(numerical_path)
    if raw.columns.tolist() != ["date", "start_date", "end_date", "OT"]:
        raise ValueError("unexpected raw numerical columns")
    cleaned = raw.loc[raw.OT.notna(), ["start_date", "end_date", "OT"]]
    if cleaned.to_csv(index=False).encode("utf-8") != clean_path.read_bytes():
        raise ValueError("raw-to-clean numeric replay differs byte-for-byte")
    clean = pd.read_csv(clean_path)
    lineage = pd.read_csv(lineage_path, keep_default_na=False)
    lineage = lineage.loc[lineage.domain.eq("SocialGood")]
    numeric_audit = {
        "raw_rows": len(raw), "clean_rows": len(clean),
        "removed_missing_OT_rows": int(raw.OT.isna().sum()),
        "raw_missing_by_field": {name: int(raw[name].isna().sum()) for name in raw.columns},
        "clean_missing_by_field": {name: int(clean[name].isna().sum()) for name in clean.columns},
        "raw_duplicate_start_dates": int(raw.start_date.duplicated().sum()),
        "clean_duplicate_start_dates": int(clean.start_date.duplicated().sum()),
        "raw_invalid_start_dates": int(pd.to_datetime(raw.start_date, errors="coerce").isna().sum()),
        "clean_sha256": CLEAN_SHA,
        "clean_rule": "drop rows with missing OT; retain start_date,end_date,OT in original order",
        "byte_exact_replay": True,
    }
    if (numeric_audit["raw_rows"] - numeric_audit["removed_missing_OT_rows"]
            != numeric_audit["clean_rows"]):
        raise ValueError("raw/clean numeric row conservation failed")

    text_audit = {}
    for name, path in sources.items():
        frame = pd.read_csv(path)
        rows = lineage.loc[lineage.source.eq(name)]
        fact = frame.fact.fillna("").astype(str).str.strip()
        missing_fact = fact.eq("") | fact.str.match(r"^NA\b.*", case=False)
        reasons = {key: int(value) for key, value in rows.reason.value_counts().items()}
        if (len(frame) != len(rows)
                or sorted(rows.raw_row.astype(int).tolist()) != list(range(len(frame)))
                or reasons.get("missing_fact", 0) != int(missing_fact.sum())):
            raise ValueError(f"raw-to-lineage text row audit failed: {name}")
        text_audit[name] = {
            "raw_rows": len(frame), "lineage_rows": len(rows),
            "raw_missing_fact_rows": int(missing_fact.sum()),
            "raw_duplicate_intervals": int(frame.duplicated(["start_date", "end_date"]).sum()),
            "raw_missing_by_field": {column: int(frame[column].isna().sum())
                                     for column in ("start_date", "end_date", "fact")},
            "lineage_reasons": reasons,
            "canonical_facts": reasons.get("retained", 0),
            "raw_sha256": RAW_SHA[name],
        }
    return {
        "schema_version": 1, "task": "SocialGood_h3_f1",
        "source_repository_revision": SOURCE_REVISION,
        "raw_sha256": RAW_SHA,
        "numeric": numeric_audit,
        "text": text_audit,
        "text_lineage_sha256": LINEAGE_SHA,
        "text_raw_rows": sum(record["raw_rows"] for record in text_audit.values()),
        "text_canonical_facts": sum(record["canonical_facts"] for record in text_audit.values()),
        "limits": "Raw inputs and full numeric snapshot are not distributed in the safe Release because later OT values disclose test truth; reproduce from the anchored Time-MMD revision and frozen local inputs.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--clean-numeric", type=Path, required=True)
    parser.add_argument("--lineage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.raw_root, args.clean_numeric, args.lineage)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "raw_rows": result["numeric"]["raw_rows"],
                      "clean_rows": result["numeric"]["clean_rows"]}, ensure_ascii=False))
