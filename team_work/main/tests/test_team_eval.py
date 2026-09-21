import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from team_eval.core import EvidenceError, freeze_route, moving_block_ci, validate


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.task = dict(task_id="demo-H1", fold_id=1, horizon=1, n_rows=50, train_end=20,
                         cal_end=28, dec_end=40, test_end=50, seasonal_period=2,
                         snapshot_sha256="a" * 64, feature_manifest_sha256="b" * 64)
        self.spec = {"numeric_candidates": ["Last"], "candidate_variants": ["semantic_residual"],
                     "selection": {"row_null_draws": 99, "p_threshold": .025}}
        self.samples = []
        self.predictions = []
        for segment, origins in (("cal", [20, 21]), ("dec", [28, 29, 35, 36])):
            for origin in origins:
                self.samples.append(dict(task_id="demo-H1", fold_id=1, origin_id=origin, segment=segment,
                                         horizon=1, target=[1.0], cutoff_index=origin-1,
                                         target_start_index=origin, snapshot_sha256="a"*64,
                                         feature_manifest_sha256="b"*64, preprocessing_fit_end=20))
                for candidate, value in (("Last", 0.0), ("semantic_residual", 1.0)):
                    self.predictions.append(dict(task_id="demo-H1", fold_id=1, origin_id=origin,
                                                 segment=segment, candidate_id=candidate, seed=2026,
                                                 prediction=[value], config_sha256=candidate,
                                                 feature_manifest_sha256="b"*64))
        self.nulls = [dict(task_id="demo-H1", fold_id=1, candidate_id="semantic_residual",
                           decision_null_mse=[.5]*99, method="row-permutation-refit", seed=2026,
                           source_sha256="c"*64)]

    def test_frozen_route_uses_only_cal_and_dec(self):
        samples, predictions, _ = validate(self.task, self.samples, self.predictions)
        route = freeze_route(self.task, samples, predictions, self.nulls, self.spec)
        self.assertEqual(route["selected"], "semantic_residual")
        self.assertEqual(route["selection_data_segments"], ["cal", "dec"])

    def test_crossing_target_rejected(self):
        samples = copy.deepcopy(self.samples)
        samples[0]["origin_id"] = 28
        samples[0]["target_start_index"] = 28
        with self.assertRaises(EvidenceError):
            validate(self.task, samples, self.predictions)

    def test_missing_prediction_rejected(self):
        with self.assertRaises(EvidenceError):
            validate(self.task, self.samples, self.predictions[:-1])

    def test_duplicate_origin_rejected(self):
        with self.assertRaises(EvidenceError):
            validate(self.task, self.samples + [self.samples[0]], self.predictions)

    def test_future_preprocessing_rejected(self):
        samples = copy.deepcopy(self.samples)
        samples[0]["preprocessing_fit_end"] = 30
        with self.assertRaises(EvidenceError):
            validate(self.task, samples, self.predictions)

    def test_feature_hash_mismatch_rejected(self):
        predictions = copy.deepcopy(self.predictions)
        predictions[0]["feature_manifest_sha256"] = "f"*64
        with self.assertRaises(EvidenceError):
            validate(self.task, self.samples, predictions)

    def test_null_evidence_required(self):
        samples, predictions, _ = validate(self.task, self.samples, self.predictions)
        with self.assertRaises(EvidenceError):
            freeze_route(self.task, samples, predictions, [], self.spec)

    def test_fixed_seed_interval_replays(self):
        data = [1.0, -1.0, 2.0, 0.0, 1.5, -.5]
        self.assertEqual(moving_block_ci(data, 2, 200, 2026), moving_block_ci(data, 2, 200, 2026))

    def test_wrong_prediction_horizon_rejected(self):
        predictions = copy.deepcopy(self.predictions)
        predictions[0]["prediction"] = [0.0, 0.0]
        with self.assertRaises(EvidenceError):
            validate(self.task, self.samples, predictions)


if __name__ == "__main__":
    unittest.main()
