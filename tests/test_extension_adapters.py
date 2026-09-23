"""Input-boundary checks for the three alternative verifier conditions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.extensions.prm_router.run_alternative import (
    messages_for_genprm, read_records, skywork_input,
)


class ExtensionInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.row = {"example_id": "x", "current_step": 2,
                    "question": "Find x", "steps": ["First\nline", "Second"], "label": 1}

    def test_genprm_sees_only_prefix_and_current_step(self) -> None:
        messages = messages_for_genprm(self.row)
        self.assertIn("Paragraph 2: Second", messages[1]["content"])
        self.assertNotIn("Paragraph 3", messages[1]["content"])

    def test_skywork_adapter_has_one_reward_marker_per_step(self) -> None:
        def prepare(question, response, tokenizer, step_token):
            self.assertEqual(question, "Find x")
            self.assertEqual(response, "First line\nSecond")
            steps = response.split(step_token)
            return [1, 2, 3, 4], steps, [0, 1, 0, 1]

        ids, final_position = skywork_input(object(), self.row, prepare)
        self.assertEqual(len(ids), 4)
        self.assertEqual(final_position, 3)

    def test_rejects_full_solution_after_current_step(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.jsonl"
            row = dict(self.row, steps=["First", "Second", "Future"])
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact step prefix"):
                read_records(path)


if __name__ == "__main__":
    unittest.main()
