"""Verify the final SafeFAME-TS v2 evidence chain without retraining."""

from __future__ import annotations

import csv
import json
import math
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "outputs" / "tables" / "v2"
FINAL = ROOT / "outputs" / "tables" / "final"
SENSITIVITY = ROOT / "outputs" / "reviewer_sensitivity"


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f"{label}: {actual} != {expected}")


def check_docx(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
        xml = archive.read("word/document.xml").decode("utf-8")
        for marker in ("TODO", "待补充", "XXXXX"):
            assert marker not in xml, f"placeholder remains in {path.name}: {marker}"
        assert "Test-Time Training for Multimodal Time Series Forecasting" not in xml


def main() -> int:
    claims = json.loads((V2 / "verified_claims.json").read_text(encoding="utf-8"))
    metrics = rows(ROOT / "outputs" / "safefame_v2" / "safefame_v2_metrics.csv")
    audit = rows(ROOT / "outputs" / "safefame_v2" / "safefame_v2_selection_audit.csv")
    candidates = [row for row in metrics if row["model"] in {"semantic_residual", "frequency_residual"}]
    semantic = [row for row in candidates if row["model"] == "semantic_residual"]
    frequency = [row for row in candidates if row["model"] == "frequency_residual"]

    assert len(audit) == claims["tasks"] == 18
    assert len(candidates) == claims["total_candidate_pathways"] == 36
    assert sum(row["selected_path"] != "numeric_fallback" for row in audit) == claims["selected_text_tasks"]
    assert sum(float(row["improvement_vs_fallback_pct"]) > 0 for row in semantic) == claims["semantic_point_improvement_tasks"]
    assert sum(float(row["block_ci_low"]) > 0 for row in semantic) == claims["semantic_significant_improvement_tasks"]
    assert sum(float(row["block_ci_high"]) < 0 for row in semantic) == claims["semantic_significant_harm_tasks"]
    assert sum(float(row["improvement_vs_fallback_pct"]) > 0 for row in frequency) == claims["frequency_point_improvement_tasks"]
    assert sum(float(row["block_ci_low"]) > 0 for row in frequency) == claims["frequency_significant_improvement_tasks"]
    assert sum(float(row["block_ci_high"]) < 0 for row in frequency) == claims["frequency_significant_harm_tasks"]
    assert sum(int(row["test_windows"]) for row in audit) == claims["test_windows"]

    sensitivity_summary = json.loads((SENSITIVITY / "reviewer_sensitivity_summary.json").read_text(encoding="utf-8"))
    sensitivity = rows(SENSITIVITY / "reviewer_sensitivity_results.csv")
    assert len(sensitivity) == sensitivity_summary["pathways"] == 36
    assert sum(float(row["circular_shift_p_value"]) <= 0.025 for row in sensitivity) == sensitivity_summary["circular_shift_passes_at_0_025"] == 0
    assert sum(float(row["test_gain_ci_low_pct"]) > 0 for row in sensitivity) == sensitivity_summary["matched_control_test_ci_positive"] == 2
    assert sum(float(row["test_gain_ci_high_pct"]) < 0 for row in sensitivity) == sensitivity_summary["matched_control_test_ci_negative"] == 9
    assert sum(row["approx_mde_loss"] == "" for row in sensitivity) == sensitivity_summary["power_audit_unavailable_lt3_blocks"] == 32

    old_claims = json.loads((FINAL / "verified_claims.json").read_text(encoding="utf-8"))
    ett = rows(FINAL / "table_ett_external.csv")
    assert len(ett) == old_claims["ett_tasks"]
    close(sum(float(row["patchtst_gain_vs_best_non_patch_pct"]) for row in ett) / len(ett), old_claims["ett_mean_patchtst_gain_pct"], "ETT mean gain")

    docx_paths = [
        ROOT / "paper" / "final" / "第4周_课程设计选题与初步技术方案.docx",
        ROOT / "paper" / "final" / "第10周_课程设计书面中期进展报告.docx",
        ROOT / "paper" / "final" / "SafeFAME-TS_课程设计报告_修订版.docx",
    ]
    pdf_paths = [path.with_suffix(".pdf") for path in docx_paths]
    expected = [*docx_paths, *pdf_paths]
    expected += list((ROOT / "outputs" / "figures" / "v2").glob("*.png"))
    expected += list(SENSITIVITY.glob("*.png"))
    expected += [
        ROOT / "outputs" / "safefame_v2" / "safefame_v2_protocol.json",
        ROOT / "outputs" / "safefame_v2" / "safefame_v2_metrics.csv",
        ROOT / "outputs" / "safefame_v2" / "safefame_v2_selection_audit.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in expected if not path.is_file() or path.stat().st_size == 0]
    assert not missing, f"missing artifacts: {missing}"
    for path in docx_paths:
        check_docx(path)
    for path in pdf_paths:
        assert path.read_bytes()[:5] == b"%PDF-", f"invalid PDF: {path.name}"

    protocol = json.loads((ROOT / "outputs" / "safefame_v2" / "safefame_v2_protocol.json").read_text(encoding="utf-8"))
    assert protocol["permutations"] == 99
    close(float(protocol["permutation_p_threshold"]), 0.025, "permutation threshold")
    assert protocol["seeds"] == [2026, 2027, 2028]

    print("PASS: v2 claims, reviewer sensitivity, ETT claims, protocols, figures, DOCX and PDF deliverables are consistent.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
