"""Prepare a balanced step-level subset from PRMBench.

The numbers of correct and erroneous step prefixes are configured through
command-line arguments.

For erroneous examples, we select the first annotated error step.
For correct examples, we select a step before the first error, so that
the prefix does not already contain an earlier annotated error.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_dataset

from cl_prm.data.prmbench import (
    choose_split,
    get_error_steps,
    get_question,
    get_steps,
    make_record,
    parse_list,
)


DATASET_NAME = "hitsmy/PRMBench_Preview"


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