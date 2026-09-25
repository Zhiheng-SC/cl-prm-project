"""Tests for the isolated post-hoc threshold-sensitivity extension."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "experiments"
    / "extensions"
    / "prm_router"
    / "analyze_threshold_sensitivity.py"
)
SPEC = importlib.util.spec_from_file_location("threshold_sensitivity", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ThresholdSensitivityTests(unittest.TestCase):
    def test_tie_break_prefers_closest_to_half_then_smaller(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.2, 0.4, 0.6, 0.8])
        curve, selected = MODULE.threshold_curve(labels, scores)
        self.assertTrue(curve)
        self.assertEqual(selected["balanced_accuracy"], 1.0)
        self.assertEqual(selected["threshold"], 0.5)

    def test_raw_score_rethresholding_ignores_stored_prediction(self) -> None:
        base = [{"label": 1, "disprm_score": 0.9}]
        second = [
            {
                "label": 1,
                "extension_score": 0.7,
                "extension_prediction": 0,
                "extension_runtime_seconds": 1.0,
            }
        ]
        arrays = MODULE.arrays_for_threshold(
            base, second, "math_prm", re_threshold=0.8, second_threshold=0.6
        )
        np.testing.assert_array_equal(arrays["second_predictions"], np.asarray([1]))

    def test_pathfinder_uses_final_gated_score(self) -> None:
        base = [{"label": 0, "disprm_score": 0.1}]
        second = [
            {
                "label": 0,
                "pathfinder_official_score": -1.0,
                "pathfinder_prediction": 1,
                "pathfinder_runtime_seconds": 1.0,
            }
        ]
        arrays = MODULE.arrays_for_threshold(
            base, second, "pathfinder", re_threshold=0.5, second_threshold=0.2
        )
        np.testing.assert_array_equal(arrays["second_predictions"], np.asarray([0]))

    def test_output_guard_allows_only_dedicated_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact_root = Path(temporary)
            allowed = artifact_root / "extensions" / "threshold_sensitivity"
            MODULE.validate_output_root(artifact_root, allowed)
            with self.assertRaisesRegex(ValueError, "must be written only"):
                MODULE.validate_output_root(
                    artifact_root, artifact_root / "extensions" / "full"
                )

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact_root = Path(temporary)
            output = artifact_root / "extensions" / "threshold_sensitivity"
            output.mkdir(parents=True)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                MODULE.validate_output_root(artifact_root, output)

    def test_metadata_hash_accepts_only_crlf_checkout_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "validation.jsonl"
            linux_bytes = b'{"value": 1}\n'
            path.write_bytes(linux_bytes.replace(b"\n", b"\r\n"))
            path.with_suffix(".metadata.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "input": {
                            "path": "data/validation.jsonl",
                            "selected_examples": 1,
                        },
                        "output": {"sha256": hashlib.sha256(linux_bytes).hexdigest()},
                    }
                ),
                encoding="utf-8",
            )
            metadata = MODULE.validate_metadata(path, "validation", 1)
            self.assertEqual(
                metadata["_local_hash_verification"]["method"],
                "normalized_crlf_to_lf",
            )


if __name__ == "__main__":
    unittest.main()
