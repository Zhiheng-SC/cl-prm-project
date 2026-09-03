"""Shared validation for safely resuming JSONL verifier inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ResumeState:
    """Validated state for an inference output file."""

    existing_records: list[dict[str, Any]]
    completed: int
    file_mode: str


def record_key(record: dict[str, Any]) -> tuple[str, int]:
    """Return the stable example-and-step key used by inference outputs."""
    try:
        return str(record["example_id"]), int(record["current_step"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Every inference record must contain example_id and current_step."
        ) from error


def read_output_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read an existing JSONL output, allowing an empty file."""
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}. "
                    "The output may end with a partial record; inspect it "
                    "before resuming."
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {path}."
                )
            records.append(record)

    return records


def ensure_unique_keys(
    records: list[dict[str, Any]],
    description: str,
) -> list[tuple[str, int]]:
    """Return record keys after rejecting duplicates."""
    keys = [record_key(record) for record in records]
    seen: set[tuple[str, int]] = set()

    for key in keys:
        if key in seen:
            raise ValueError(
                f"Duplicate record key {key!r} in {description}."
            )
        seen.add(key)

    return keys


def prepare_resume(
    *,
    output_path: Path,
    input_records: list[dict[str, Any]],
    resume: bool,
    expected_metadata: dict[str, Any],
) -> ResumeState:
    """Validate an existing output as an exact prefix of the input."""
    if not resume or not output_path.exists():
        return ResumeState(
            existing_records=[],
            completed=0,
            file_mode="w",
        )

    existing_records = read_output_jsonl(output_path)

    if len(existing_records) > len(input_records):
        raise ValueError(
            f"Existing output has {len(existing_records)} records, but "
            f"the selected input has only {len(input_records)}."
        )

    input_keys = ensure_unique_keys(input_records, "selected input")
    output_keys = ensure_unique_keys(existing_records, "existing output")

    for index, output_record in enumerate(existing_records):
        expected_record = input_records[index]
        if output_keys[index] != input_keys[index]:
            raise ValueError(
                "Existing output is not an exact prefix of the selected "
                f"input at position {index + 1}: expected "
                f"{input_keys[index]!r}, found {output_keys[index]!r}."
            )

        if int(output_record["label"]) != int(expected_record["label"]):
            raise ValueError(
                f"Label mismatch at position {index + 1} for "
                f"{output_keys[index]!r}."
            )

        for field, expected_value in expected_metadata.items():
            if field not in output_record:
                raise ValueError(
                    f"Existing output is missing resume metadata {field!r} "
                    f"at position {index + 1}."
                )
            if output_record[field] != expected_value:
                raise ValueError(
                    f"Resume metadata mismatch for {field!r} at position "
                    f"{index + 1}: expected {expected_value!r}, found "
                    f"{output_record[field]!r}."
                )

    return ResumeState(
        existing_records=existing_records,
        completed=len(existing_records),
        file_mode="a",
    )
