"""Contract checks for the round-three independent intake tools; D: temp only."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from team_eval.core import EvidenceError, sha256
from team_eval.null_bridge import convert_nulls, write_conversion
from team_eval.snapshot_audit import audit_one


class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stage = self.root / "selection"
        self.stage.mkdir()
        self.candidates = ["N+S+Q", "N+S+Q+SF"]
        task = dict(task_id="Demo_h1_f1", fold_id=1, horizon=1, input_len=2, n_rows=20,
                    train_end=8, cal_end=10, dec_end=14, test_end=18,
                    snapshot_sha256="a" * 64, feature_manifest_sha256="b" * 64,
                    seasonal_period=2)
        spec = dict(status="FROZEN", original_split_spec_sha256="f" * 64,
                    candidate_variants=self.candidates, selection=dict(row_null_draws=999))
        samples = []
        predictions = []
        for segment, origins in (("cal", (8, 9)), ("dec", (10, 11))):
            for origin in origins:
                samples.append(dict(task_id=task["task_id"], fold_id=1, origin_id=origin,
                                    segment=segment, horizon=1, target=[1.0], cutoff_index=origin - 1,
                                    target_start_index=origin, snapshot_sha256=task["snapshot_sha256"],
                                    feature_manifest_sha256=task["feature_manifest_sha256"],
                                    preprocessing_fit_end=8))
                for candidate in self.candidates:
                    predictions.append(dict(task_id=task["task_id"], fold_id=1, origin_id=origin,
                                            segment=segment, candidate_id=candidate, seed=2026,
                                            prediction=[0.0], config_sha256="c" * 64,
                                            feature_manifest_sha256=task["feature_manifest_sha256"]))
        for name, value in (("task.json", task), ("spec.json", spec)):
            (self.stage / name).write_text(json.dumps(value), encoding="utf-8")
        for name, value in (("samples.jsonl", samples), ("predictions.jsonl", predictions)):
            (self.stage / name).write_text("".join(json.dumps(v) + "\n" for v in value), encoding="utf-8")
        manifest = dict(status="READY_FOR_FREEZE", segments=["calibration", "decision"],
                        scenario="proxy", bundle_signature="b" * 64,
                        split_spec_sha256="f" * 64, sample_rows=len(samples),
                        prediction_rows=len(predictions),
                        files_sha256={p.name: sha256(p) for p in self.stage.iterdir()})
        (self.stage / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        self.null_dirs = []
        for candidate in self.candidates:
            folder = self.root / ("first" if candidate == self.candidates[0] else "second")
            folder.mkdir()
            self.null_dirs.append(folder)
            with (folder / "null_scores.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["iteration", "seed", "status", "loss", "error"])
                for i in range(999):
                    writer.writerow([i, 2026000 + i, "ok", 2.0, ""])
            summary = dict(candidate=candidate, task="Demo_h1_f1", scenario="proxy",
                           bundle_signature="b" * 64, requested=999, successful=999, failed=0,
                           contract_minimum=999, code_commit="d" * 40,
                           config_sha256="e" * 64, observed_decision_loss=1.0, p_value=0.001)
            (folder / "permutation_summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (folder / "null_scores_summary.json").write_text(json.dumps(dict(candidate=candidate,
                segment="decision", requested=999, successful=999, failed=0,
                contract_minimum=999,
                observed_loss=1.0, p_value=0.001)), encoding="utf-8")

    def test_conversion_preserves_every_loss_and_refuses_overwrite(self):
        rows, report = convert_nulls(self.stage, self.null_dirs)
        self.assertEqual([r["candidate_id"] for r in rows], self.candidates)
        self.assertEqual([len(r["decision_null_mse"]) for r in rows], [999, 999])
        self.assertTrue(all(set(r["decision_null_mse"]) == {2.0} for r in rows))
        out = self.root / "converted.jsonl"
        write_conversion(out, rows, report)
        self.assertEqual(len(out.read_text(encoding="utf-8").splitlines()), 2)
        with self.assertRaises(EvidenceError):
            write_conversion(out, rows, report)

    def test_rejects_wrong_scenario(self):
        path = self.null_dirs[0] / "permutation_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["scenario"] = "conservative_lag"
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "scenario"):
            convert_nulls(self.stage, self.null_dirs)

    def test_rejects_selection_without_frozen_split_anchor(self):
        spec_path = self.stage / "spec.json"
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        spec["original_split_spec_sha256"] = "0" * 64
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        manifest_path = self.stage / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files_sha256"]["spec.json"] = sha256(spec_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "frozen split"):
            convert_nulls(self.stage, self.null_dirs)

    def test_rejects_missing_or_shifted_null_rows(self):
        path = self.null_dirs[0] / "null_scores.csv"
        rows = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
        with self.assertRaises(EvidenceError):
            convert_nulls(self.stage, self.null_dirs)
        rows[1] = rows[1].replace("0,2026000,", "0,2026001,")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        with self.assertRaises(EvidenceError):
            convert_nulls(self.stage, self.null_dirs)

    def test_rejects_failed_iteration_even_when_summary_claims_success(self):
        path = self.null_dirs[0] / "null_scores.csv"
        rows = path.read_text(encoding="utf-8").splitlines()
        rows[2] = "1,2026001,failed,,injected failure"
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        with self.assertRaises(EvidenceError):
            convert_nulls(self.stage, self.null_dirs)

    def test_rejects_self_reported_mse_mismatch(self):
        path = self.null_dirs[0] / "permutation_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        summary["observed_decision_loss"] = 0.9
        path.write_text(json.dumps(summary), encoding="utf-8")
        with self.assertRaisesRegex(EvidenceError, "observed MSE"):
            convert_nulls(self.stage, self.null_dirs)

    def test_snapshot_checks_bytes_rows_and_dates(self):
        path = self.root / "Demo_numerical.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["start_date", "end_date", "OT"])
            for i in range(20):
                writer.writerow([f"2026-01-{i+1:02d}", f"2026-01-{i+1:02d}", i])
        task = dict(domain="Demo", fold_id=1, input_len=2, horizon=1, lag_days=1,
                    n_rows_expected=20, bounds=[8, 10, 14, 16])
        declared = dict(task_id="Demo_h1_f1", domain="Demo", fold_id="1", input_len="2",
                        horizon="1", lag_days="1", n_rows="20", train_end="8",
                        calibration_end="10", decision_end="14", test_end="16",
                        unknown_target_rows="0", numerical_sha256=sha256(path))
        self.assertEqual(audit_one(task, declared, path)["status"], "BYTE_AND_CONTENT_CHECK_PASSED")
        path.write_text(path.read_text(encoding="utf-8").replace("2026-01-02", "2026-01-01"), encoding="utf-8")
        declared["numerical_sha256"] = sha256(path)
        with self.assertRaises(EvidenceError):
            audit_one(task, declared, path)


if __name__ == "__main__":
    unittest.main()
