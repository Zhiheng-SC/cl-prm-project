"""Tests for the guarded formal verifier-inference plan."""

from __future__ import annotations

import hashlib
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

from run_formal_inference import build_command, validate_split  # noqa: E402


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FormalInferencePlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.split_path = self.root / "train.jsonl"
        self.split_path.write_text(
            json.dumps({"example_id": "one"}) + "\n"
            + json.dumps({"example_id": "two"}) + "\n",
            encoding="utf-8",
        )
        self.manifest = {
            "splits": {
                "train": {
                    "path": str(self.split_path),
                    "sha256": file_sha256(self.split_path),
                    "examples": 2,
                }
            }
        }
        self.config = {
            "models": {
                "base": {
                    "name": "base/model",
                    "revision": "base-revision",
                },
                "primary_second_stage": {
                    "name": "path/model",
                    "revision": "path-revision",
                },
                "alternative_second_stage": {
                    "name": "gen/model",
                    "revision": "gen-revision",
                },
            },
            "runtime": {"warmup_examples": 3},
            "inference": {
                "pathfinder": {
                    "threshold": 0.5,
                    "max_input_tokens": 4096,
                    "attention_implementation": "flash_attention_2",
                    "load_in_4bit": False,
                },
                "genprm": {
                    "seed": 42,
                    "max_analysis_tokens": 256,
                    "max_input_tokens": 4096,
                },
            },
        }

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_matching_split_manifest_is_accepted(self) -> None:
        self.assertEqual(
            validate_split("train", self.manifest),
            self.split_path,
        )

    def test_modified_split_is_rejected(self) -> None:
        self.split_path.write_text(
            self.split_path.read_text(encoding="utf-8") + "{}\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_split("train", self.manifest)

    def test_pathfinder_command_uses_frozen_settings(self) -> None:
        command = build_command(
            verifier="pathfinder",
            split_name="train",
            input_path=self.split_path,
            output_dir=self.root / "outputs",
            config=self.config,
            reason_eval_threshold=0.5,
        )

        self.assertIn("path/model", command)
        self.assertIn("path-revision", command)
        self.assertIn("flash_attention_2", command)
        self.assertIn("4096", command)
        self.assertIn("--resume", command)
        warmup_index = command.index("--warmup-examples")
        self.assertEqual(command[warmup_index + 1], "3")


if __name__ == "__main__":
    unittest.main()
