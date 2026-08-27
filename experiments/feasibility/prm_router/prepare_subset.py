"""Prepare a balanced step-level subset from PRMBench.

The numbers of correct and erroneous step prefixes are configured through
command-line arguments.

For erroneous examples, we select the first annotated error step.
For correct examples, we select a step before the first error, so that
the prefix does not already contain an earlier annotated error.
"""

import argparse
import ast
import json
import random
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset


DATASET_NAME = "hitsmy/PRMBench_Preview"


def parse_list(value: Any) -> list:
    """Convert a dataset field into a Python list."""
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
    """Get the question used in the modified PRMBench example."""
    for key in ("modified_question", "question", "original_question"):
        value = row.get(key)

        if value:
            return str(value).strip()

    raise KeyError("No question field found in the dataset row.")


def get_steps(row: dict) -> list[str]:
    """Get and clean the modified reasoning process."""
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
    """Return valid 1-based error-step indices."""
    values = parse_list(row.get("error_steps"))

    error_steps = sorted(
        {
            int(step_number)
            for step_number in values
            if 1 <= int(step_number) <= number_of_steps
        }
    )

    return error_steps


def make_record(
    row: dict,
    source_row_index: int,
    step_number: int,
    label: int,
) -> dict:
    """Create one step-prefix evaluation example."""
    question = get_question(row)
    all_steps = get_steps(row)
    prefix_steps = all_steps[:step_number]

    formatted_steps = "\n".join(
        f"Step {index}: {step}"
        for index, step in enumerate(prefix_steps, start=1)
    )

    model_input = f"Question: {question}\n\n{formatted_steps}"

    return {
        "example_id": str(
            row.get("idx", f"row_{source_row_index}")
        ),
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
    """Choose a dataset split without assuming that it is called train."""
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


def prepare_subset(
    rows,
    n_correct: int,
    n_error: int,
    seed: int,
) -> list[dict]:
    """Construct a balanced subset using distinct problems."""
    random_generator = random.Random(seed)

    row_indices = list(range(len(rows)))
    random_generator.shuffle(row_indices)

    selected_records: list[dict] = []
    used_example_ids: set[str] = set()

    # Select erroneous examples.
    # We use the first annotated error so the prefix does not contain
    # another annotated error before the current step.
    for row_index in row_indices:
        if sum(record["label"] == 0 for record in selected_records) >= n_error:
            break

        row = rows[row_index]
        steps = get_steps(row)
        error_steps = get_error_steps(row, len(steps))

        if not error_steps:
            continue

        example_id = str(row.get("idx", f"row_{row_index}"))

        if example_id in used_example_ids:
            continue

        first_error_step = min(error_steps)

        selected_records.append(
            make_record(
                row=row,
                source_row_index=row_index,
                step_number=first_error_step,
                label=0,
            )
        )
        used_example_ids.add(example_id)

    # Select correct examples from different problems.
    # Only steps before the first error are eligible, because a later
    # step would receive a prefix that already contains an error.
    for row_index in row_indices:
        if sum(record["label"] == 1 for record in selected_records) >= n_correct:
            break

        row = rows[row_index]
        steps = get_steps(row)
        error_steps = get_error_steps(row, len(steps))
        example_id = str(row.get("idx", f"row_{row_index}"))

        if example_id in used_example_ids:
            continue

        if error_steps:
            first_error_step = min(error_steps)
            correct_candidates = list(range(1, first_error_step))
        else:
            correct_candidates = list(range(1, len(steps) + 1))

        if not correct_candidates:
            continue

        selected_step = random_generator.choice(correct_candidates)

        selected_records.append(
            make_record(
                row=row,
                source_row_index=row_index,
                step_number=selected_step,
                label=1,
            )
        )
        used_example_ids.add(example_id)

    actual_correct = sum(record["label"] == 1 for record in selected_records)
    actual_error = sum(record["label"] == 0 for record in selected_records)

    if actual_correct != n_correct or actual_error != n_error:
        raise RuntimeError(
            "Could not construct the requested subset. "
            f"Requested correct={n_correct}, error={n_error}; "
            f"obtained correct={actual_correct}, error={actual_error}."
        )

    random_generator.shuffle(selected_records)

    return selected_records


def save_jsonl(records: list[dict], output_path: Path) -> None:
    """Save records as UTF-8 JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--n-correct",
        type=int,
        default=10,
        help="Number of correct step-prefix examples.",
    )
    parser.add_argument(
        "--n-error",
        type=int,
        default=10,
        help="Number of erroneous step-prefix examples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split. By default, use train or the first split.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/prm_router/smoke_test.jsonl"),
        help="Output JSONL path.",
    )

    args = parser.parse_args()

    print(f"Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME)

    split_name, rows = choose_split(dataset, args.split)

    print(f"Using split: {split_name}")
    print(f"Number of source examples: {len(rows)}")

    records = prepare_subset(
        rows=rows,
        n_correct=args.n_correct,
        n_error=args.n_error,
        seed=args.seed,
    )

    save_jsonl(records, args.output)

    number_correct = sum(record["label"] == 1 for record in records)
    number_error = sum(record["label"] == 0 for record in records)

    print()
    print("Subset created successfully.")
    print(f"Correct steps: {number_correct}")
    print(f"Error steps:   {number_error}")
    print(f"Total:         {len(records)}")
    print(f"Saved to:      {args.output.resolve()}")

    print()
    print("First example:")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()