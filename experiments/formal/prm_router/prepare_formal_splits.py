"""Create leakage-controlled formal PRMBench train/validation/test splits.

The script excludes every original-question group represented in the frozen
feasibility pilot, assigns all remaining groups to exactly one formal split,
and selects a balanced set of correct and erroneous step-prefix records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import DatasetDict, load_dataset

from cl_prm.data.prmbench import (
    choose_split,
    get_error_steps,
    get_question,
    get_steps,
    make_record,
)
from cl_prm.utils.formal_config import validate_formal_config


DEFAULT_CONFIG = Path("configs/experiments/prm_router_formal.json")
DEFAULT_OUTPUT_DIR = Path("data/prm_router/formal")


@dataclass(frozen=True)
class Candidate:
    source_row_index: int
    step_number: int
    label: int
    classification: str
    group_id: str
    original_question: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def normalize_question(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"\s+", " ", text).strip().casefold()
    if not text:
        raise ValueError("Original-question group text is empty.")
    return text


def original_question(row: dict[str, Any]) -> str:
    value = row.get("original_question")
    if value is None or not str(value).strip():
        value = row.get("question")
    if value is None or not str(value).strip():
        value = get_question(row)
    return str(value).strip()


def stable_seed(seed: int, *parts: Any) -> int:
    payload = "\0".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def choose_correct_step(
    row: dict[str, Any],
    source_row_index: int,
    seed: int,
) -> int | None:
    steps = get_steps(row)
    error_steps = get_error_steps(row, len(steps))
    upper_bound = min(error_steps) - 1 if error_steps else len(steps)
    if upper_bound < 1:
        return None
    generator = random.Random(
        stable_seed(seed, "correct-step", source_row_index)
    )
    return generator.randint(1, upper_bound)


def resolve_pilot_groups(
    rows,
    pilot_records: list[dict[str, Any]],
) -> set[str]:
    indices_by_id: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_id[str(row.get("idx", f"row_{index}"))].append(index)

    groups: set[str] = set()
    for record in pilot_records:
        example_id = str(record["example_id"])
        source_index = record.get("source_row_index")

        if isinstance(source_index, int) and 0 <= source_index < len(rows):
            source_id = str(
                rows[source_index].get("idx", f"row_{source_index}")
            )
            if source_id != example_id:
                source_index = None

        if source_index is None:
            matches = indices_by_id.get(example_id, [])
            if len(matches) != 1:
                raise ValueError(
                    f"Could not uniquely map pilot example {example_id!r}; "
                    f"found {len(matches)} source rows."
                )
            source_index = matches[0]

        question = original_question(rows[source_index])
        groups.add(normalize_question(question))

    return groups


def build_candidates(
    rows,
    excluded_groups: set[str],
    seed: int,
) -> list[Candidate]:
    candidates: list[Candidate] = []

    for source_index, row in enumerate(rows):
        question = original_question(row)
        group_id = normalize_question(question)
        if group_id in excluded_groups:
            continue

        classification = str(row.get("classification") or "unknown")
        steps = get_steps(row)
        error_steps = get_error_steps(row, len(steps))

        if error_steps:
            candidates.append(
                Candidate(
                    source_row_index=source_index,
                    step_number=min(error_steps),
                    label=0,
                    classification=classification,
                    group_id=group_id,
                    original_question=question,
                )
            )

        correct_step = choose_correct_step(row, source_index, seed)
        if correct_step is not None:
            candidates.append(
                Candidate(
                    source_row_index=source_index,
                    step_number=correct_step,
                    label=1,
                    classification=classification,
                    group_id=group_id,
                    original_question=question,
                )
            )

    return candidates


def assign_group(
    group_id: str,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> str:
    value = stable_seed(seed, "group-split", group_id) / 2**64
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def proportional_allocation(
    buckets: dict[str, list[Candidate]],
    sample_size: int,
) -> dict[str, int]:
    available = {name: len(values) for name, values in buckets.items()}
    total = sum(available.values())
    if total < sample_size:
        raise RuntimeError(
            f"Requested {sample_size} records but only {total} are available."
        )

    allocation = {name: 0 for name in available}
    nonempty = [name for name, count in available.items() if count > 0]

    if sample_size >= len(nonempty):
        for name in nonempty:
            allocation[name] = 1

    while sum(allocation.values()) < sample_size:
        eligible = [
            name
            for name in nonempty
            if allocation[name] < available[name]
        ]
        if not eligible:
            raise RuntimeError("Candidate allocation exhausted unexpectedly.")

        name = max(
            eligible,
            key=lambda item: (
                sample_size * available[item] / total - allocation[item],
                available[item] - allocation[item],
                item,
            ),
        )
        allocation[name] += 1

    return allocation


def stratified_sample(
    candidates: Iterable[Candidate],
    sample_size: int,
    seed: int,
    excluded_rows: set[int] | None = None,
) -> list[Candidate]:
    excluded_rows = excluded_rows or set()
    buckets: dict[str, list[Candidate]] = defaultdict(list)

    for candidate in candidates:
        if candidate.source_row_index not in excluded_rows:
            buckets[candidate.classification].append(candidate)

    for classification, bucket in buckets.items():
        random.Random(
            stable_seed(seed, "bucket", classification)
        ).shuffle(bucket)

    allocation = proportional_allocation(buckets, sample_size)
    selected = [
        candidate
        for classification, count in allocation.items()
        for candidate in buckets[classification][:count]
    ]
    random.Random(stable_seed(seed, "selected-order")).shuffle(selected)
    return selected


def materialize_record(
    rows,
    candidate: Candidate,
    formal_split: str,
) -> dict[str, Any]:
    record = make_record(
        row=rows[candidate.source_row_index],
        source_row_index=candidate.source_row_index,
        step_number=candidate.step_number,
        label=candidate.label,
    )
    record.update(
        {
            "original_question": candidate.original_question,
            "original_question_group": candidate.group_id,
            "formal_split": formal_split,
        }
    )
    return record


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_splits(
    split_records: dict[str, list[dict[str, Any]]],
    expected_sizes: dict[str, int],
    pilot_groups: set[str],
) -> None:
    group_sets: dict[str, set[str]] = {}
    record_keys: set[tuple[str, int]] = set()

    for split_name, records in split_records.items():
        if len(records) != expected_sizes[split_name]:
            raise RuntimeError(
                f"{split_name} has {len(records)} records; "
                f"expected {expected_sizes[split_name]}."
            )

        labels = Counter(int(record["label"]) for record in records)
        expected_per_label = expected_sizes[split_name] // 2
        if labels != Counter({0: expected_per_label, 1: expected_per_label}):
            raise RuntimeError(
                f"Unexpected label counts for {split_name}: {labels}."
            )

        groups = {
            str(record["original_question_group"])
            for record in records
        }
        if groups & pilot_groups:
            raise RuntimeError(f"Pilot-group leakage detected in {split_name}.")
        group_sets[split_name] = groups

        for record in records:
            key = (str(record["example_id"]), int(record["current_step"]))
            if key in record_keys:
                raise RuntimeError(f"Duplicate formal record key: {key!r}")
            record_keys.add(key)

    names = list(group_sets)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = group_sets[left_name] & group_sets[right_name]
            if overlap:
                raise RuntimeError(
                    f"Group leakage between {left_name} and {right_name}: "
                    f"{len(overlap)} groups."
                )


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    dataset_config = config["dataset"]
    split_config = config["formal_splits"]

    requested_sizes = {
        "train": int(split_config["train_examples"]),
        "validation": int(split_config["validation_examples"]),
        "test": int(split_config["test_examples"]),
    }
    if any(size <= 0 or size % 2 for size in requested_sizes.values()):
        raise ValueError("Every formal split size must be a positive even number.")
    if float(split_config["correct_fraction"]) != 0.5:
        raise ValueError("This implementation requires correct_fraction = 0.5.")

    seed = int(split_config["seed"])
    total_size = sum(requested_sizes.values())
    train_fraction = requested_sizes["train"] / total_size
    validation_fraction = requested_sizes["validation"] / total_size

    dataset_revision = str(dataset_config["revision"])
    print(
        f"Loading dataset: {dataset_config['name']} "
        f"at revision {dataset_revision}"
    )
    dataset = load_dataset(
        dataset_config["name"],
        revision=dataset_revision,
    )
    if not isinstance(dataset, DatasetDict):
        raise TypeError("Expected load_dataset to return a DatasetDict.")
    source_split, rows = choose_split(dataset, dataset_config.get("split"))

    pilot_path = Path(dataset_config["pilot_path"])
    pilot_records = read_jsonl(pilot_path)
    pilot_groups = resolve_pilot_groups(rows, pilot_records)
    candidates = build_candidates(rows, pilot_groups, seed)

    assigned: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        split_name = assign_group(
            candidate.group_id,
            seed,
            train_fraction,
            validation_fraction,
        )
        assigned[split_name].append(candidate)

    split_records: dict[str, list[dict[str, Any]]] = {}
    for split_name, split_size in requested_sizes.items():
        label_size = split_size // 2
        split_seed = stable_seed(seed, "sample", split_name)

        error_candidates = [
            candidate
            for candidate in assigned[split_name]
            if candidate.label == 0
        ]
        errors = stratified_sample(
            error_candidates,
            label_size,
            stable_seed(split_seed, "error"),
        )
        used_rows = {candidate.source_row_index for candidate in errors}

        correct_candidates = [
            candidate
            for candidate in assigned[split_name]
            if candidate.label == 1
        ]
        correct = stratified_sample(
            correct_candidates,
            label_size,
            stable_seed(split_seed, "correct"),
            excluded_rows=used_rows,
        )

        selected = [*errors, *correct]
        random.Random(stable_seed(split_seed, "final-order")).shuffle(selected)
        split_records[split_name] = [
            materialize_record(rows, candidate, split_name)
            for candidate in selected
        ]

    audit_splits(split_records, requested_sizes, pilot_groups)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    for split_name, records in split_records.items():
        path = args.output_dir / f"{split_name}.jsonl"
        write_jsonl(records, path)
        output_paths[split_name] = path

    manifest = {
        "schema_version": 1,
        "config_path": args.config.as_posix(),
        "dataset": dataset_config["name"],
        "dataset_revision": dataset_revision,
        "dataset_fingerprint": str(rows._fingerprint),
        "source_split": source_split,
        "seed": seed,
        "pilot_examples": len(pilot_records),
        "excluded_pilot_groups": len(pilot_groups),
        "splits": {},
    }

    for split_name, records in split_records.items():
        manifest["splits"][split_name] = {
            "path": output_paths[split_name].as_posix(),
            "sha256": sha256_file(output_paths[split_name]),
            "examples": len(records),
            "labels": dict(sorted(Counter(
                int(record["label"]) for record in records
            ).items())),
            "original_question_groups": len({
                record["original_question_group"] for record in records
            }),
            "classifications": dict(sorted(Counter(
                str(record["classification"]) for record in records
            ).items())),
        }

    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Source split: {source_split}")
    print(f"Source rows: {len(rows)}")
    print(f"Pilot examples: {len(pilot_records)}")
    print(f"Excluded pilot groups: {len(pilot_groups)}")
    print(f"Available candidates after exclusion: {len(candidates)}")
    print()
    for split_name, records in split_records.items():
        labels = Counter(int(record["label"]) for record in records)
        groups = len({record["original_question_group"] for record in records})
        print(
            f"{split_name:10s}: {len(records):4d} examples, "
            f"labels 0/1={labels[0]}/{labels[1]}, groups={groups}"
        )
    print()
    print("Leakage audit: passed")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
