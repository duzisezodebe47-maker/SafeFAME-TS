"""Synthetic contract tests; temporary artifacts are created on D: only."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from team_eval.core import EvidenceError, evaluate_test, freeze_route, jsonl, validate
from team_eval.v2 import (SCALE, digest, load_bundle, load_prediction_grid,
                          numeric_baselines, signature, write_baseline_predictions,
                          write_stage_inputs)


class V2BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[3])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "bundle"
        self.task_dir = self.folder / "Demo_h2_f1"
        (self.task_dir / "proxy").mkdir(parents=True)
        self.spec_path = self.root / "spec.json"
        self.spec = {"status": "frozen", "approved_by": "main:test", "tasks": [
            {"domain": "Demo", "fold_id": 1, "input_len": 3, "horizon": 2, "bounds": [8, 12, 18, 24],
             "numerical_sha256": "a" * 64, "n_rows_expected": 24}]}
        self.spec_path.write_text(json.dumps(self.spec), encoding="utf-8")
        self.rows = [
            {"segment": segment, "origin_index": str(origin),
             "origin_id": f"Demo:h2:f1:o{origin}"}
            for segment, origins in (("train", [3, 4]), ("calibration", [8, 9]),
                                     ("decision", [12, 13]), ("test", [18, 19])) for origin in origins]
        with (self.task_dir / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        origins = np.array([int(r["origin_index"]) for r in self.rows])
        raw = np.stack([origins + 10., origins + 11.], axis=1)
        np.save(self.task_dir / "origin_index.npy", origins)
        np.save(self.task_dir / "numeric_history.npy", np.tile(np.array([1., 2., 3.]), (8, 1)))
        np.save(self.task_dir / "targets.npy", raw)
        np.save(self.task_dir / "targets_standardized.npy", (raw - 2.) / 4.)
        np.save(self.task_dir / "proxy" / "text_available.npy", np.ones(8, dtype=bool))
        (self.task_dir / "numeric_fit.json").write_text(json.dumps({"mean": 2., "std": 4., "train_end": 8}), encoding="utf-8")
        (self.folder / "schema.json").write_text(json.dumps({"status": "frozen", "numeric_features": ["OT"]}), encoding="utf-8")
        self.inputs = {"mode": "frozen", "split_spec": self.spec, "split_spec_sha256": digest(self.spec_path)}
        self.sig = signature(self.inputs)
        self.reseal()
        self.pred_path = self.root / "predictions.csv"
        self.write_predictions()

    def reseal(self):
        files = {p.relative_to(self.folder).as_posix(): digest(p)
                 for p in self.folder.rglob("*") if p.is_file() and p.name != "manifest.json"}
        (self.folder / "manifest.json").write_text(
            json.dumps({"signature": self.sig, "inputs": self.inputs, "files": files}), encoding="utf-8")

    def write_predictions(self, rows=None):
        if rows is None:
            rows = []
            for sample in self.rows:
                if sample["segment"] not in ("calibration", "decision"):
                    continue
                for step in (1, 2):
                    rows.append({"task_id": "Demo_h2_f1", "fold_id": 1,
                                 "origin_id": sample["origin_id"], "origin_index": sample["origin_index"],
                                 "segment": sample["segment"], "scenario": "proxy", "candidate_id": "N+S+Q",
                                 "seed": 2026, "step": step, "y_pred": 1.5,
                                 "target_scale": SCALE, "bundle_signature": self.sig,
                                 "config_sha256": "b" * 64, "code_commit": "c" * 40})
        with self.pred_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return rows

    def get_bundle(self):
        return load_bundle(self.folder, self.sig, self.spec_path, "Demo_h2_f1", "proxy")

    def test_aligned_bundle_and_prediction_grid(self):
        bundle = self.get_bundle()
        grid = load_prediction_grid([self.pred_path], bundle, ("calibration", "decision"))
        self.assertEqual((grid["origins"], grid["horizon"]), (4, 2))
        self.assertEqual(len(grid["values"][("N+S+Q", 2026)]), 8)

    def test_preview_spec_rejected(self):
        self.spec["status"] = "draft_not_for_training"
        self.spec_path.write_text(json.dumps(self.spec), encoding="utf-8")
        with self.assertRaises(EvidenceError):
            self.get_bundle()

    def test_bundle_byte_tamper_rejected(self):
        with (self.task_dir / "samples.csv").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        with self.assertRaises(EvidenceError):
            self.get_bundle()

    def test_origin_array_shift_rejected(self):
        np.save(self.task_dir / "origin_index.npy", np.array([4, 3, 8, 9, 12, 13, 18, 19]))
        self.reseal()
        with self.assertRaises(EvidenceError):
            self.get_bundle()

    def test_missing_step_rejected(self):
        rows = self.write_predictions()
        self.write_predictions(rows[:-1])
        with self.assertRaises(EvidenceError):
            load_prediction_grid([self.pred_path], self.get_bundle(), ("calibration", "decision"))

    def test_equal_count_but_shifted_origin_rejected(self):
        rows = self.write_predictions()
        rows[0]["origin_id"] = "Demo:h2:f1:o7"
        rows[0]["origin_index"] = "7"
        self.write_predictions(rows)
        with self.assertRaises(EvidenceError):
            load_prediction_grid([self.pred_path], self.get_bundle(), ("calibration", "decision"))

    def test_same_rows_reordered_rejected(self):
        rows = self.write_predictions()
        rows[0], rows[1] = rows[1], rows[0]
        self.write_predictions(rows)
        with self.assertRaises(EvidenceError):
            load_prediction_grid([self.pred_path], self.get_bundle(), ("calibration", "decision"))

    def test_duplicate_step_rejected(self):
        rows = self.write_predictions()
        self.write_predictions(rows + [rows[0]])
        with self.assertRaises(EvidenceError):
            load_prediction_grid([self.pred_path], self.get_bundle(), ("calibration", "decision"))

    def test_mixed_scale_rejected(self):
        rows = self.write_predictions()
        rows[0]["target_scale"] = "raw_OT"
        self.write_predictions(rows)
        with self.assertRaises(EvidenceError):
            load_prediction_grid([self.pred_path], self.get_bundle(), ("calibration", "decision"))

    def test_old_smoke_schema_rejected(self):
        self.pred_path.write_text("task_id,fold_id,origin_id,candidate_id,seed,horizon,y_pred\n", encoding="utf-8")
        with self.assertRaises(EvidenceError):
            load_prediction_grid([self.pred_path], self.get_bundle(), ("calibration", "decision"))

    def test_numeric_baselines_never_fit_test_truth(self):
        bundle = self.get_bundle()
        original = numeric_baselines(bundle, [0.1, 1., 10.], seasonal_period=2)
        bundle["arrays"]["targets_standardized"][6:] += 1000.
        changed = numeric_baselines(bundle, [0.1, 1., 10.], seasonal_period=2)
        self.assertEqual(original["ridge_alpha"], changed["ridge_alpha"])
        for candidate in original["predictions"]:
            np.testing.assert_array_equal(original["predictions"][candidate],
                                          changed["predictions"][candidate])

    def test_numeric_baseline_export_aligns_with_contract(self):
        bundle = self.get_bundle()
        baselines = numeric_baselines(bundle, [0.1, 1.], seasonal_period=2)
        out = self.root / "numeric.csv"
        count = write_baseline_predictions(out, bundle, baselines, "proxy", "c" * 40)
        self.assertEqual(count, 3 * 4 * 2)
        # The same strict loader must accept the independent numeric predictions.
        with out.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        selection = rows
        # CSV groups candidate first; write one candidate per file as required by the loader.
        for candidate in baselines["predictions"]:
            selected = [r for r in selection if r["candidate_id"] == candidate]
            self.write_predictions(selected)
            grid = load_prediction_grid([self.pred_path], bundle, ("calibration", "decision"))
            self.assertEqual(grid["origins"], 4)

    def test_selection_package_is_test_free_and_freezes_route(self):
        bundle = self.get_bundle()
        baselines = numeric_baselines(bundle, [0.1, 1.], seasonal_period=2)
        numeric_path = self.root / "numeric.csv"
        write_baseline_predictions(numeric_path, bundle, baselines, "proxy", "c" * 40)
        grid = load_prediction_grid([numeric_path, self.pred_path], bundle, ("calibration", "decision"))
        policy = {"numeric_fallback_candidates": ["Last", "SeasonalNaive", "AR-Ridge"],
                  "gate_candidates": ["N+S+Q"], "seasonal_periods": {"Demo": 2},
                  "row_permutations": 999, "p_threshold": 0.025, "seed": 2026}
        folder = self.root / "selection"
        manifest = write_stage_inputs(folder, bundle, grid, policy, ("calibration", "decision"))
        self.assertEqual(manifest["status"], "READY_FOR_FREEZE")
        self.assertEqual(manifest["scenario"], "proxy")
        self.assertNotIn('"segment": "test"', (folder / "samples.jsonl").read_text(encoding="utf-8"))
        task = json.loads((folder / "task.json").read_text(encoding="utf-8"))
        spec = json.loads((folder / "spec.json").read_text(encoding="utf-8"))
        samples, predictions, _ = validate(task, jsonl(folder / "samples.jsonl"),
                                           jsonl(folder / "predictions.jsonl"), spec=spec)
        null = [dict(task_id="Demo_h2_f1", fold_id=1, candidate_id="N+S+Q",
                     decision_null_mse=[100.] * 999, method="row-permutation-refit",
                     seed=2026, source_sha256="d" * 64)]
        route = freeze_route(task, samples, predictions, null, spec)
        self.assertEqual(route["selection_data_segments"], ["cal", "dec"])
        # The held-out stage is materialized separately and uses byte-identical
        # task and evaluator spec files, so the frozen route can be checked.
        numeric_test = self.root / "numeric_test.csv"
        write_baseline_predictions(numeric_test, bundle, baselines, "proxy", "c" * 40,
                                   segments=("test",))
        with self.pred_path.open(encoding="utf-8", newline="") as stream:
            example = next(csv.DictReader(stream))
        test_rows = []
        for sample in self.rows:
            if sample["segment"] == "test":
                for step in (1, 2):
                    row = dict(example)
                    row.update(origin_id=sample["origin_id"], origin_index=sample["origin_index"],
                               segment="test", step=step)
                    test_rows.append(row)
        self.write_predictions(test_rows)
        test_grid = load_prediction_grid([numeric_test, self.pred_path], bundle, ("test",))
        test_folder = self.root / "test_stage"
        write_stage_inputs(test_folder, bundle, test_grid, policy, ("test",))
        self.assertEqual(digest(folder / "task.json"), digest(test_folder / "task.json"))
        self.assertEqual(digest(folder / "spec.json"), digest(test_folder / "spec.json"))
        test_samples, test_predictions, _ = validate(task, jsonl(test_folder / "samples.jsonl"),
                                                      jsonl(test_folder / "predictions.jsonl"),
                                                      require_test=True, spec=spec)
        result = evaluate_test(task, test_samples, test_predictions, route, spec)
        self.assertEqual(result["test_windows"], 2)

    def test_short_test_interval_is_reported_unestimable(self):
        from team_eval.core import evaluate_test
        task = dict(task_id="small", fold_id=1, horizon=2, seasonal_period=12)
        samples = {("test", 10): {"target": [1., 1.]}}
        predictions = {("test", 10, "Last", 2026): [0., 0.]}
        route = {"fallback": "Last", "selected": "numeric_fallback"}
        report = evaluate_test(task, samples, predictions, route,
                               {"uncertainty": {"draws": 100, "seed": 2026}})
        self.assertIsNone(report["delta_ci95"])
        self.assertEqual(report["interval_status"], "NOT_ESTIMABLE_TOO_FEW_ORIGINS")


if __name__ == "__main__":
    unittest.main()
