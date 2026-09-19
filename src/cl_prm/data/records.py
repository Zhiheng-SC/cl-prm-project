"""Shared JSON/JSONL record loading and verifier alignment helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSON object/list or JSONL file as a list of records."""
    text = path.read_text(encoding="utf-8-sig").strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON structure in {path}")


def record_key(record: dict[str, Any]) -> tuple[str, int]:
    """Return the stable example-and-step key used across verifier outputs."""
    return str(record["example_id"]), int(record["current_step"])


def index_records(
    records: list[dict[str, Any]],
    source_name: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Index records by stable key while rejecting duplicates."""
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for record in records:
        key = record_key(record)
        if key in indexed:
            raise ValueError(f"Duplicate {source_name} record: {key}")
        indexed[key] = record
    return indexed


def align_verifier_records(
    reasoneval_path: Path,
    pathfinder_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Align ReasonEval and PathFinder outputs by example/step key."""
    base = index_records(read_records(reasoneval_path), "ReasonEval")
    second = index_records(read_records(pathfinder_path), "PathFinder")
    keys = sorted(set(base) & set(second))
    if len(keys) != len(base) or len(keys) != len(second):
        raise ValueError("ReasonEval and PathFinder records do not match exactly.")
    return [base[key] for key in keys], [second[key] for key in keys]


def correctness_arrays(
    base_records: list[dict[str, Any]],
    second_records: list[dict[str, Any]],
    threshold: float,
) -> dict[str, np.ndarray]:
    """Construct aligned labels, predictions, correctness, and gain labels."""
    labels = np.asarray([int(row["label"]) for row in base_records], dtype=np.int64)
    second_labels = np.asarray(
        [int(row["label"]) for row in second_records],
        dtype=np.int64,
    )
    if not np.array_equal(labels, second_labels):
        raise ValueError("ReasonEval and PathFinder ground-truth labels differ.")

    scores = np.asarray(
        [float(row["disprm_score"]) for row in base_records],
        dtype=np.float64,
    )
    base_predictions = (scores >= threshold).astype(np.int64)
    second_predictions = np.asarray(
        [int(row["pathfinder_prediction"]) for row in second_records],
        dtype=np.int64,
    )

    base_correct = base_predictions == labels
    second_correct = second_predictions == labels
    gains = second_correct.astype(np.int64) - base_correct.astype(np.int64)

    return {
        "labels": labels,
        "scores": scores,
        "base_predictions": base_predictions,
        "second_predictions": second_predictions,
        "base_correct": base_correct,
        "second_correct": second_correct,
        "gain_labels": gains,
    }
