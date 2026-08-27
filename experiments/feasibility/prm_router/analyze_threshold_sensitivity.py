"""Evaluate PRM routing robustness across DisPRM thresholds.

This script reuses existing ReasonEval and GenPRM outputs. It does not run
either language model. For each threshold, it reconstructs benefit labels,
performs five-fold out-of-fold router evaluation, and saves one summary CSV.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

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
    / "threshold_sensitivity.csv"
)

DEFAULT_THRESHOLDS = [0.50, 0.80, 0.90, 0.95, 0.96, 0.98]
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
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=DEFAULT_BUDGETS,
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
        "--random-trials",
        type=int,
        default=1000,
    )

    return parser.parse_args()


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


def validate_arguments(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")

    if args.random_trials < 1:
        raise ValueError("--random-trials must be at least 1.")

    for threshold in args.thresholds:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Threshold must be between 0 and 1: {threshold}"
            )

    for budget in args.budgets:
        if not 0 <= budget <= 100:
            raise ValueError(
                f"Budget must be between 0 and 100: {budget}"
            )


def evaluate_threshold(
    threshold: float,
    ordered_dis_records: list[dict[str, Any]],
    labels: np.ndarray,
    dis_scores: np.ndarray,
    gen_predictions: np.ndarray,
    budgets: list[int],
    folds: int,
    seed: int,
    random_trials: int,
) -> dict[str, float | int]:
    total = len(labels)
    all_indices = list(range(total))

    dis_predictions = (
        dis_scores >= threshold
    ).astype(np.int64)

    dis_correct = dis_predictions == labels
    gen_correct = gen_predictions == labels

    benefit_labels = (
        (~dis_correct) & gen_correct
    ).astype(np.int64)

    harmful_mask = dis_correct & (~gen_correct)

    beneficial_indices = set(
        np.flatnonzero(benefit_labels == 1).tolist()
    )
    harmful_indices = set(
        np.flatnonzero(harmful_mask).tolist()
    )

    positive_count = int(benefit_labels.sum())
    negative_count = total - positive_count

    if min(positive_count, negative_count) < folds:
        raise ValueError(
            f"Threshold {threshold:.4f} has insufficient class counts "
            f"for {folds}-fold CV: positive={positive_count}, "
            f"negative={negative_count}."
        )

    features = np.asarray(
        [
            build_features(
                record=record,
                score=float(score),
                threshold=threshold,
            )
            for record, score in zip(
                ordered_dis_records,
                dis_scores,
            )
        ],
        dtype=np.float64,
    )

    cross_validation = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )

    oof_probabilities = np.zeros(
        total,
        dtype=np.float64,
    )

    for fold_id, (train_indices, test_indices) in enumerate(
        cross_validation.split(features, benefit_labels),
        start=1,
    ):
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

    roc_auc = float(
        roc_auc_score(
            benefit_labels,
            oof_probabilities,
        )
    )

    average_precision = float(
        average_precision_score(
            benefit_labels,
            oof_probabilities,
        )
    )

    base_accuracy = routing_accuracy(
        labels,
        dis_predictions,
        gen_predictions,
        set(),
    )

    gen_accuracy = routing_accuracy(
        labels,
        dis_predictions,
        gen_predictions,
        set(all_indices),
    )

    learned_order = sorted(
        all_indices,
        key=lambda index: oof_probabilities[index],
        reverse=True,
    )

    uncertainty_order = sorted(
        all_indices,
        key=lambda index: abs(
            dis_scores[index] - threshold
        ),
    )

    result: dict[str, float | int] = {
        "threshold": threshold,
        "examples": total,
        "disprm_accuracy": base_accuracy,
        "genprm_accuracy": gen_accuracy,
        "beneficial_calls": positive_count,
        "harmful_calls": len(harmful_indices),
        "neutral_calls": (
            total
            - positive_count
            - len(harmful_indices)
        ),
        "oracle_accuracy": (
            base_accuracy + positive_count / total
        ),
        "router_roc_auc": roc_auc,
        "router_average_precision": average_precision,
        "benefit_positive_rate": positive_count / total,
    }

    for budget_percent in budgets:
        budget = round(
            total * budget_percent / 100
        )

        learned_routed = set(
            learned_order[:budget]
        )
        uncertainty_routed = set(
            uncertainty_order[:budget]
        )

        learned_accuracy = routing_accuracy(
            labels,
            dis_predictions,
            gen_predictions,
            learned_routed,
        )

        uncertainty_accuracy = routing_accuracy(
            labels,
            dis_predictions,
            gen_predictions,
            uncertainty_routed,
        )

        random_generator = random.Random(
            seed
            + int(round(threshold * 10_000))
            + budget_percent
        )

        random_accuracies = []

        for _ in range(random_trials):
            random_routed = set(
                random_generator.sample(
                    all_indices,
                    budget,
                )
            )

            random_accuracies.append(
                routing_accuracy(
                    labels,
                    dis_predictions,
                    gen_predictions,
                    random_routed,
                )
            )

        result[
            f"learned_accuracy_{budget_percent}"
        ] = learned_accuracy

        result[
            f"uncertainty_accuracy_{budget_percent}"
        ] = uncertainty_accuracy

        result[
            f"random_accuracy_{budget_percent}"
        ] = mean(random_accuracies)

        result[
            f"learned_beneficial_{budget_percent}"
        ] = len(
            learned_routed & beneficial_indices
        )

        result[
            f"learned_harmful_{budget_percent}"
        ] = len(
            learned_routed & harmful_indices
        )

    return result


def main() -> None:
    args = parse_args()
    validate_arguments(args)

    dis_records = index_records(
        read_records(args.disprm),
        model_name="DisPRM",
    )

    gen_records = index_records(
        read_records(args.genprm),
        model_name="GenPRM",
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

    if not common_keys:
        raise ValueError("No paired records were found.")

    ordered_dis_records = [
        dis_records[key]
        for key in common_keys
    ]

    ordered_gen_records = [
        gen_records[key]
        for key in common_keys
    ]

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

    gen_predictions = np.asarray(
        [
            int(record["genprm_prediction"])
            for record in ordered_gen_records
        ],
        dtype=np.int64,
    )

    results = []

    for threshold in args.thresholds:
        result = evaluate_threshold(
            threshold=threshold,
            ordered_dis_records=ordered_dis_records,
            labels=labels,
            dis_scores=dis_scores,
            gen_predictions=gen_predictions,
            budgets=args.budgets,
            folds=args.folds,
            seed=args.seed,
            random_trials=args.random_trials,
        )

        results.append(result)

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

    display_budget = (
        20
        if 20 in args.budgets
        else args.budgets[0]
    )

    print(f"Examples: {len(labels)}")
    print(f"Thresholds: {len(results)}")
    print(
        f"Displayed routing budget: "
        f"{display_budget}%"
    )
    print()

    print(
        "Threshold | DisPRM | Benefit/Harm | "
        "ROC-AUC | AP    | Learned | Uncertainty | Random | Oracle"
    )
    print("-" * 93)

    for result in results:
        threshold = float(result["threshold"])
        dis_accuracy = float(
            result["disprm_accuracy"]
        )
        beneficial = int(
            result["beneficial_calls"]
        )
        harmful = int(
            result["harmful_calls"]
        )
        roc_auc = float(
            result["router_roc_auc"]
        )
        average_precision = float(
            result["router_average_precision"]
        )
        learned = float(
            result[
                f"learned_accuracy_{display_budget}"
            ]
        )
        uncertainty = float(
            result[
                f"uncertainty_accuracy_{display_budget}"
            ]
        )
        random_accuracy = float(
            result[
                f"random_accuracy_{display_budget}"
            ]
        )
        oracle = float(
            result["oracle_accuracy"]
        )

        print(
            f"{threshold:>9.2f} | "
            f"{dis_accuracy:>6.3f} | "
            f"{beneficial:>3}/{harmful:<3}      | "
            f"{roc_auc:>7.3f} | "
            f"{average_precision:>5.3f} | "
            f"{learned:>7.3f} | "
            f"{uncertainty:>11.3f} | "
            f"{random_accuracy:>6.3f} | "
            f"{oracle:>6.3f}"
        )

    print()
    print(
        "Summary saved to: "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()