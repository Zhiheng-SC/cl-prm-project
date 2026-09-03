"""Tests for safe verifier-inference resumption."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_DIR = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "feasibility"
    / "prm_router"
)
sys.path.insert(0, str(MODULE_DIR))

from inference_io import prepare_resume  # noqa: E402


def sample_record(
    example_id: str,
    current_step: int,
    label: int,
    **metadata: object,
) -> dict[str, object]:
    """Create the minimal record required by resume validation."""
    return {
        "example_id": example_id,
        "current_step": current_step,
        "label": label,
        **metadata,
    }


class PrepareResumeTests(unittest.TestCase):
    """Exercise success and corruption paths without loading a model."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_path = (
            Path(self.temporary_directory.name) / "predictions.jsonl"
        )
        self.metadata = {
            "verifier_model": "example/model",
            "verifier_revision": "abc123",
        }
        self.input_records = [
            sample_record("example-1", 1, 0),
            sample_record("example-2", 3, 1),
        ]

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def write_output(self, records: list[dict[str, object]]) -> None:
        """Write test records as UTF-8 JSONL."""
        with self.output_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(json.dumps(record) + "\n")

    def completed_record(
        self,
        example_id: str,
        current_step: int,
        label: int,
    ) -> dict[str, object]:
        """Create an output row containing the expected run metadata."""
        return sample_record(
            example_id,
            current_step,
            label,
            **self.metadata,
        )

    def test_missing_output_starts_new_file(self) -> None:
        state = prepare_resume(
            output_path=self.output_path,
            input_records=self.input_records,
            resume=True,
            expected_metadata=self.metadata,
        )

        self.assertEqual(state.completed, 0)
        self.assertEqual(state.file_mode, "w")
        self.assertEqual(state.existing_records, [])

    def test_valid_prefix_uses_append_mode(self) -> None:
        first_result = self.completed_record("example-1", 1, 0)
        self.write_output([first_result])

        state = prepare_resume(
            output_path=self.output_path,
            input_records=self.input_records,
            resume=True,
            expected_metadata=self.metadata,
        )

        self.assertEqual(state.completed, 1)
        self.assertEqual(state.file_mode, "a")
        self.assertEqual(state.existing_records, [first_result])

    def test_metadata_mismatch_is_rejected(self) -> None:
        result = self.completed_record("example-1", 1, 0)
        result["verifier_revision"] = "different"
        self.write_output([result])

        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            prepare_resume(
                output_path=self.output_path,
                input_records=self.input_records,
                resume=True,
                expected_metadata=self.metadata,
            )

    def test_wrong_prefix_order_is_rejected(self) -> None:
        self.write_output([
            self.completed_record("example-2", 3, 1),
        ])

        with self.assertRaisesRegex(ValueError, "not an exact prefix"):
            prepare_resume(
                output_path=self.output_path,
                input_records=self.input_records,
                resume=True,
                expected_metadata=self.metadata,
            )

    def test_duplicate_input_key_is_rejected(self) -> None:
        duplicate_input = [
            sample_record("example-1", 1, 0),
            sample_record("example-1", 1, 0),
        ]
        self.write_output([
            self.completed_record("example-1", 1, 0),
        ])

        with self.assertRaisesRegex(ValueError, "Duplicate record key"):
            prepare_resume(
                output_path=self.output_path,
                input_records=duplicate_input,
                resume=True,
                expected_metadata=self.metadata,
            )

    def test_truncated_json_is_rejected(self) -> None:
        self.output_path.write_text(
            '{"example_id": "unfinished"',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "partial record"):
            prepare_resume(
                output_path=self.output_path,
                input_records=self.input_records,
                resume=True,
                expected_metadata=self.metadata,
            )


if __name__ == "__main__":
    unittest.main()
