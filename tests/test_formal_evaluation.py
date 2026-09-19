"""Lightweight tests for frozen formal-evaluation helpers."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO_ROOT / "experiments" / "feasibility" / "prm_router"
sys.path.insert(0, str(MODULE_DIR))

from evaluate_formal_development import make_cost_predictor, top_budget_indices  # noqa: E402
from evaluate_formal_test import (  # noqa: E402
    bootstrap_group_metric,
    final_predictions,
    selected_runtime,
)
from select_reasoneval_threshold import candidate_thresholds  # noqa: E402


class FormalEvaluationHelperTests(unittest.TestCase):
    def test_top_budget_indices_uses_stable_descending_order(self) -> None:
        scores = np.asarray([0.2, 0.9, 0.9, 0.1], dtype=np.float64)
        selected = top_budget_indices(scores, 0.5)
        np.testing.assert_array_equal(selected, np.asarray([1, 2]))

    def test_final_predictions_replace_only_selected_rows(self) -> None:
        base = np.asarray([0, 0, 1, 1], dtype=np.int64)
        second = np.asarray([1, 1, 0, 0], dtype=np.int64)
        selected = np.asarray([1, 3], dtype=np.int64)
        result = final_predictions(base, second, selected)
        np.testing.assert_array_equal(result, np.asarray([0, 1, 1, 0]))

    def test_selected_runtime_is_sequential_cascade_cost(self) -> None:
        base = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)
        second = np.asarray([4.0, 5.0, 6.0], dtype=np.float64)
        selected = np.asarray([0, 2], dtype=np.int64)
        self.assertEqual(selected_runtime(base, second, selected), 13.0)

    def test_group_bootstrap_is_reproducible(self) -> None:
        values = np.asarray([1.0, 0.0, 1.0, 1.0], dtype=np.float64)
        groups = np.asarray(["a", "a", "b", "c"], dtype=object)
        first = bootstrap_group_metric(values, groups, 50, 0.95, 123)
        second = bootstrap_group_metric(values, groups, 50, 0.95, 123)
        self.assertEqual(first, second)

    def test_threshold_candidates_include_endpoints_and_midpoints(self) -> None:
        values = candidate_thresholds(np.asarray([0.2, 0.8], dtype=np.float64))
        np.testing.assert_allclose(values, np.asarray([0.0, 0.2, 0.5, 0.8, 1.0]))

    def test_formal_config_freezes_cost_alpha_and_budget_curve(self) -> None:
        config_path = REPO_ROOT / "configs" / "experiments" / "prm_router_formal.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["routing"]["cost_aware"]["ridge_alpha"], 1.0)
        self.assertEqual(config["routing"]["primary_call_budget"], 0.2)
        self.assertEqual(
            config["routing"]["secondary_call_budgets"],
            [0.1, 0.3, 0.4, 0.5, 0.75, 1],
        )
        predictor = make_cost_predictor(
            alpha=float(config["routing"]["cost_aware"]["ridge_alpha"])
        )
        self.assertEqual(predictor.named_steps["ridge"].alpha, 1.0)


if __name__ == "__main__":
    unittest.main()
