"""Shared PRMBench parsing and step-prefix construction helpers."""

from __future__ import annotations

import ast
import json
from typing import Any

from datasets import DatasetDict


def parse_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    raise TypeError(f"Cannot convert value to list: {value!r}")


def get_question(row: dict) -> str:
    for key in ("modified_question", "question", "original_question"):
        value = row.get(key)
        if value:
            return str(value).strip()
    raise KeyError("No question field found in the dataset row.")


def get_steps(row: dict) -> list[str]:
    value = row.get("modified_process")
    if value is None:
        value = row.get("process")
    if value is None:
        raise KeyError("No modified_process or process field found.")

    steps = parse_list(value)
    steps = [str(step).strip() for step in steps if str(step).strip()]
    if not steps:
        raise ValueError("The reasoning process contains no steps.")
    return steps


def get_error_steps(row: dict, number_of_steps: int) -> list[int]:
    values = parse_list(row.get("error_steps"))
    return sorted(
        {
            int(step_number)
            for step_number in values
            if 1 <= int(step_number) <= number_of_steps
        }
    )


def make_record(
    row: dict,
    source_row_index: int,
    step_number: int,
    label: int,
) -> dict:
    question = get_question(row)
    all_steps = get_steps(row)
    prefix_steps = all_steps[:step_number]
    formatted_steps = "\n".join(
        f"Step {index}: {step}"
        for index, step in enumerate(prefix_steps, start=1)
    )
    model_input = f"Question: {question}\n\n{formatted_steps}"

    return {
        "example_id": str(row.get("idx", f"row_{source_row_index}")),
        "source_row_index": source_row_index,
        "classification": row.get("classification"),
        "question": question,
        "steps": prefix_steps,
        "current_step": step_number,
        "current_step_text": prefix_steps[-1],
        "total_steps": len(all_steps),
        "step_position": step_number / len(all_steps),
        "label": label,
        "label_text": "correct" if label == 1 else "error",
        "model_input": model_input,
    }


def choose_split(dataset: DatasetDict, requested_split: str | None):
    if requested_split is not None:
        if requested_split not in dataset:
            available = ", ".join(dataset.keys())
            raise KeyError(
                f"Split '{requested_split}' not found. "
                f"Available splits: {available}"
            )
        return requested_split, dataset[requested_split]

    if "train" in dataset:
        return "train", dataset["train"]

    split_name = next(iter(dataset.keys()))
    return split_name, dataset[split_name]
