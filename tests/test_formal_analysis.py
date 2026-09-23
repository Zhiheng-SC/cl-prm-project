"""Tests for frozen held-out descriptive-analysis helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
ANALYSIS_DIR = REPO_ROOT / "experiments" / "formal" / "prm_router" / "analysis"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ANALYSIS_DIR))

from analyze_test_statistics import (  # noqa: E402
    binary_router_diagnostics,
    overlap_partitions,
    reliability_bins,
    selection_outcome,
)


class FormalAnalysisTests(unittest.TestCase):
    def test_reliability_bins_cover_every_example(self) -> None:
        labels = np.asarray([0, 0, 1, 0, 1, 1, 0, 1, 0, 1])
        probabilities = np.linspace(0.05, 0.95, len(labels))
        rows = reliability_bins(labels, probabilities, bins=5)
        self.assertEqual(sum(int(row["examples"]) for row in rows), len(labels))
        self.assertEqual(len(rows), 5)

    def test_binary_metrics_distinguish_ap_and_trapezoidal_pr_auc(self) -> None:
        labels = np.asarray([0, 1, 0, 1, 0, 1])
        probabilities = np.asarray([0.1, 0.8, 0.3, 0.7, 0.6, 0.4])
        result = binary_router_diagnostics(labels, probabilities, bins=3)
        self.assertIn("average_precision", result)
        self.assertIn("pr_auc_trapezoidal", result)
        self.assertIn("roc_auc", result)
        self.assertTrue(result["calibration"]["converged"])

    def test_selection_outcome_accounts_for_all_selected_examples(self) -> None:
        gains = np.asarray([1, 0, -1, 1, 0])
        selected = np.asarray([0, 1, 2, 3])
        result = selection_outcome("example", selected, gains)
        self.assertEqual(result["beneficial"], 2)
        self.assertEqual(result["neutral"], 1)
        self.assertEqual(result["harmful"], 1)
        self.assertEqual(result["net_gain"], 1)

    def test_overlap_partitions_are_disjoint_and_exhaustive(self) -> None:
        gains = np.asarray([1, 0, -1, 1, 0, -1])
        result = overlap_partitions(
            np.asarray([0, 1, 2]),
            np.asarray([1, 2, 3]),
            gains,
        )
        counts = {row["partition"]: row["examples"] for row in result["partitions"]}
        self.assertEqual(counts["both"], 2)
        self.assertEqual(counts["expected_gain_only"], 1)
        self.assertEqual(counts["failure_prediction_only"], 1)
        self.assertEqual(counts["neither"], 2)
        self.assertEqual(sum(counts.values()), len(gains))
        self.assertAlmostEqual(result["jaccard"], 0.5)


if __name__ == "__main__":
    unittest.main()
