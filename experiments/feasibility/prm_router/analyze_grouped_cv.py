"""Compare standard and original-question-grouped OOF router evaluation.

This script reuses saved DisPRM and GenPRM outputs. It does not run either
language model. Original PRMBench questions are used as group identifiers so
that variants of the same problem cannot appear in both training and test
folds.
"""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
)

from train_benefit_router import (
    build_features,
    make_router,
    read_records,
    record_key,
    routing_accuracy,
)


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DISPRM = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "reasoneval_feasibility_100.jsonl"
)

DEFAULT_GENPRM = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "genprm_feasibility_100.jsonl"
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "robustness"
    / "grouped_cv_comparison.csv"
)

DATASET_NAME = "hitsmy/PRMBench_Preview"
DEFAULT_BUDGETS = [10, 20, 30]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--disprm",
        type=Path,
        default=DEFAULT_DISPRM,
    )
    parser.add_argument(
        "--genprm",
        type=Path,
        default=DEFAULT_GENPRM,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NAME,
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.96,
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=DEFAULT_BUDGETS,
    )

    return parser.parse_args()


def normalize_question(text: Any) -> str:
    """Create a stable group key from an original question."""
    normalized = unicodedata.normalize("NFKC", str(text))
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().casefold()


def index_records(
    records: list[dict[str, Any]],
    model_name: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}

    for record in records:
        key = record_key(record)

        if key in indexed:
            raise ValueError(
                f"Duplicate {model_name} record: {key}"
            )

        indexed[key] = record

    return indexed


def load_question_groups(
    dataset_name: str,
    requested_split: str | None,
    example_ids: list[str],
) -> tuple[np.ndarray, str]:
    """Map PRMBench example IDs to normalized original questions."""
    dataset = load_dataset(dataset_name)

    if requested_split is not None:
        if requested_split not in dataset:
            raise KeyError(
                f"Split {requested_split!r} was not found."
            )
        split_name = requested_split
    elif "train" in dataset:
        split_name = "train"
    else:
        split_name = next(iter(dataset))

    rows = dataset[split_name]
    question_by_id: dict[str, str] = {}

    for row in rows:
        example_id = str(row["idx"])

        original_question = (
            row.get("original_question")
            or row.get("question")
            or row.get("modified_question")
        )

        if not original_question:
            raise ValueError(
                f"No question found for {example_id!r}."
            )

        question_by_id[example_id] = normalize_question(
            original_question
        )

    missing_ids = sorted(
        set(example_ids) - set(question_by_id)
    )

    if missing_ids:
        raise ValueError(
            "Dataset rows are missing for example IDs: "
            + ", ".join(missing_ids[:10])
        )

    groups = np.asarray(
        [
            question_by_id[example_id]
            for example_id in example_ids
        ],
        dtype=object,
    )

    return groups, split_name


def evaluate_splitter(
    method_name: str,
    splitter,
    grouped: bool,
    features: np.ndarray,
    benefit_labels: np.ndarray,
    groups: np.ndarray,
    labels: np.ndarray,
    dis_predictions: np.ndarray,
    gen_predictions: np.ndarray,
    budgets: list[int],
    seed: int,
) -> dict[str, float | int | str]:
    """Generate OOF probabilities and routing results for one splitter."""
    total = len(labels)
    oof_probabilities = np.zeros(total, dtype=np.float64)
    fold_ids = np.zeros(total, dtype=np.int64)

    if grouped:
        split_iterator = splitter.split(
            features,
            benefit_labels,
            groups,
        )
    else:
        split_iterator = splitter.split(
            features,
            benefit_labels,
        )

    for fold_id, (train_indices, test_indices) in enumerate(
        split_iterator,
        start=1,
    ):
        train_classes = np.unique(
            benefit_labels[train_indices]
        )

        if len(train_classes) != 2:
            raise ValueError(
                f"{method_name} fold {fold_id} training data "
                "does not contain both benefit classes."
            )

        router = make_router(seed + fold_id)
        router.fit(
            features[train_indices],
            benefit_labels[train_indices],
        )

        oof_probabilities[test_indices] = (
            router.predict_proba(
                features[test_indices]
            )[:, 1]
        )

        fold_ids[test_indices] = fold_id

    group_sizes = Counter(groups.tolist())

    duplicate_groups = {
        group
        for group, size in group_sizes.items()
        if size > 1
    }

    group_folds: dict[str, set[int]] = defaultdict(set)

    for group, fold_id in zip(
        groups.tolist(),
        fold_ids.tolist(),
    ):
        group_folds[group].add(fold_id)

    split_duplicate_groups = sum(
        len(group_folds[group]) > 1
        for group in duplicate_groups
    )

    result: dict[str, float | int | str] = {
        "method": method_name,
        "examples": total,
        "unique_groups": len(group_sizes),
        "duplicate_groups": len(duplicate_groups),
        "split_duplicate_groups": split_duplicate_groups,
        "router_roc_auc": float(
            roc_auc_score(
                benefit_labels,
                oof_probabilities,
            )
        ),
        "router_average_precision": float(
            average_precision_score(
                benefit_labels,
                oof_probabilities,
            )
        ),
        "disprm_accuracy": routing_accuracy(
            labels,
            dis_predictions,
            gen_predictions,
            set(),
        ),
    }

    learned_order = sorted(
        range(total),
        key=lambda index: oof_probabilities[index],
        reverse=True,
    )

    for budget_percent in budgets:
        budget = round(total * budget_percent / 100)
        routed_indices = set(learned_order[:budget])

        result[
            f"learned_accuracy_{budget_percent}"
        ] = routing_accuracy(
            labels,
            dis_predictions,
            gen_predictions,
            routed_indices,
        )

    return result


def main() -> None:
    args = parse_args()

    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(
            "--threshold must be between 0 and 1."
        )

    for budget in args.budgets:
        if not 0 <= budget <= 100:
            raise ValueError(
                f"Invalid routing budget: {budget}"
            )

    dis_records = index_records(
        read_records(args.disprm),
        "DisPRM",
    )
    gen_records = index_records(
        read_records(args.genprm),
        "GenPRM",
    )

    common_keys = sorted(
        set(dis_records) & set(gen_records)
    )

    if len(common_keys) != len(dis_records):
        raise ValueError(
            "Some DisPRM records have no GenPRM match."
        )

    if len(common_keys) != len(gen_records):
        raise ValueError(
            "Some GenPRM records have no DisPRM match."
        )

    ordered_dis_records = [
        dis_records[key]
        for key in common_keys
    ]
    ordered_gen_records = [
        gen_records[key]
        for key in common_keys
    ]

    example_ids = [
        key[0]
        for key in common_keys
    ]

    groups, split_name = load_question_groups(
        dataset_name=args.dataset,
        requested_split=args.split,
        example_ids=example_ids,
    )

    labels = np.asarray(
        [
            int(record["label"])
            for record in ordered_dis_records
        ],
        dtype=np.int64,
    )

    gen_labels = np.asarray(
        [
            int(record["label"])
            for record in ordered_gen_records
        ],
        dtype=np.int64,
    )

    if not np.array_equal(labels, gen_labels):
        raise ValueError(
            "DisPRM and GenPRM labels do not match."
        )

    dis_scores = np.asarray(
        [
            float(record["disprm_score"])
            for record in ordered_dis_records
        ],
        dtype=np.float64,
    )

    dis_predictions = (
        dis_scores >= args.threshold
    ).astype(np.int64)

    gen_predictions = np.asarray(
        [
            int(record["genprm_prediction"])
            for record in ordered_gen_records
        ],
        dtype=np.int64,
    )

    dis_correct = dis_predictions == labels
    gen_correct = gen_predictions == labels

    benefit_labels = (
        (~dis_correct) & gen_correct
    ).astype(np.int64)

    positive_count = int(benefit_labels.sum())
    negative_count = len(benefit_labels) - positive_count

    if min(positive_count, negative_count) < args.folds:
        raise ValueError(
            "Insufficient benefit class counts for "
            f"{args.folds}-fold evaluation."
        )

    features = np.asarray(
        [
            build_features(
                record=record,
                score=float(score),
                threshold=args.threshold,
            )
            for record, score in zip(
                ordered_dis_records,
                dis_scores,
            )
        ],
        dtype=np.float64,
    )

    standard_splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    grouped_splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    results = [
        evaluate_splitter(
            method_name="standard_stratified",
            splitter=standard_splitter,
            grouped=False,
            features=features,
            benefit_labels=benefit_labels,
            groups=groups,
            labels=labels,
            dis_predictions=dis_predictions,
            gen_predictions=gen_predictions,
            budgets=args.budgets,
            seed=args.seed,
        ),
        evaluate_splitter(
            method_name="grouped_by_original_question",
            splitter=grouped_splitter,
            grouped=True,
            features=features,
            benefit_labels=benefit_labels,
            groups=groups,
            labels=labels,
            dis_predictions=dis_predictions,
            gen_predictions=gen_predictions,
            budgets=args.budgets,
            seed=args.seed,
        ),
    ]

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(results[0].keys()),
        )
        writer.writeheader()
        writer.writerows(results)

    group_sizes = Counter(groups.tolist())
    duplicate_group_count = sum(
        size > 1
        for size in group_sizes.values()
    )
    duplicate_record_count = sum(
        size
        for size in group_sizes.values()
        if size > 1
    )

    print(f"Examples:                  {len(labels)}")
    print(f"Dataset split:             {split_name}")
    print(f"Original-question groups:  {len(group_sizes)}")
    print(f"Duplicated groups:         {duplicate_group_count}")
    print(f"Records in duplicate groups: {duplicate_record_count}")
    print(f"Benefit-positive examples: {positive_count}")
    print()

    budget_headers = " | ".join(
        f"Learned {budget}%"
        for budget in args.budgets
    )

    print(
        "Method                       | Split groups | "
        f"ROC-AUC | AP    | {budget_headers}"
    )
    print("-" * (67 + 14 * len(args.budgets)))

    for result in results:
        budget_values = " | ".join(
            f"{float(result[f'learned_accuracy_{budget}']):>11.3f}"
            for budget in args.budgets
        )

        print(
            f"{str(result['method']):28s} | "
            f"{int(result['split_duplicate_groups']):>12d} | "
            f"{float(result['router_roc_auc']):>7.3f} | "
            f"{float(result['router_average_precision']):>5.3f} | "
            f"{budget_values}"
        )

    print()
    print(f"Summary saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()