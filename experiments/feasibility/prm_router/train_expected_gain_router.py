"""Evaluate a harm-aware expected-gain router with grouped OOF predictions.

This script reuses saved ReasonEval and second-stage verifier outputs. It does
not run either verifier. It compares a binary benefit-only router with a
three-class router that predicts beneficial, neutral, and harmful calls.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analyze_grouped_cv import index_records, load_question_groups
from train_benefit_router import (
    build_features,
    make_router,
    read_records,
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
    / "expected_gain_router.json"
)

DATASET_NAME = "hitsmy/PRMBench_Preview"
DEFAULT_BUDGETS = [0, 10, 20, 30, 40, 50, 75, 100]
GAIN_CLASSES = [-1, 0, 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disprm", type=Path, default=DEFAULT_DISPRM)
    parser.add_argument("--genprm", type=Path, default=DEFAULT_GENPRM)
    parser.add_argument(
        "--second-stage",
        type=Path,
        default=None,
        help=(
            "Optional second-stage verifier output. "
            "Overrides --genprm when supplied."
        ),
    )
    parser.add_argument(
        "--second-stage-kind",
        choices=("genprm", "pathfinder"),
        default="genprm",
        help="Schema used by the second-stage verifier output.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=str, default=DATASET_NAME)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.96)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=DEFAULT_BUDGETS,
    )
    return parser.parse_args()


def make_expected_gain_router(seed: int) -> Pipeline:
    """Create an unweighted multinomial model with probabilistic outputs.

    Class weights are intentionally not balanced: expected gain uses probability
    differences, so preserving the observed class priors is preferable here.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.random_trials < 1:
        raise ValueError("--random-trials must be at least 1.")
    for budget in args.budgets:
        if not 0 <= budget <= 100:
            raise ValueError(f"Invalid routing budget: {budget}")


def ordered_indices(scores: np.ndarray) -> list[int]:
    """Sort by descending score with a stable index tie-breaker."""
    return sorted(
        range(len(scores)),
        key=lambda index: (-float(scores[index]), index),
    )


def main() -> None:
    args = parse_args()
    validate_args(args)

    second_stage_path = (
        args.second_stage
        if args.second_stage is not None
        else args.genprm
    )

    if args.second_stage_kind == "genprm":
        second_stage_name = "GenPRM"
        prediction_field = "genprm_prediction"
        runtime_field = "genprm_runtime_seconds"
    else:
        second_stage_name = "PathFinder"
        prediction_field = "pathfinder_prediction"
        runtime_field = "pathfinder_runtime_seconds"

    dis_records = index_records(read_records(args.disprm), "DisPRM")
    gen_records = index_records(
        read_records(second_stage_path),
        second_stage_name,
    )
    common_keys = sorted(set(dis_records) & set(gen_records))

    if len(common_keys) != len(dis_records):
        raise ValueError(
            f"Some DisPRM records have no {second_stage_name} match."
        )
    if len(common_keys) != len(gen_records):
        raise ValueError(
            f"Some {second_stage_name} records have no DisPRM match."
        )

    ordered_dis_records = [dis_records[key] for key in common_keys]
    ordered_gen_records = [gen_records[key] for key in common_keys]
    example_ids = [key[0] for key in common_keys]

    groups, split_name = load_question_groups(
        dataset_name=args.dataset,
        requested_split=args.split,
        example_ids=example_ids,
    )

    labels = np.asarray(
        [int(record["label"]) for record in ordered_dis_records],
        dtype=np.int64,
    )
    gen_labels = np.asarray(
        [int(record["label"]) for record in ordered_gen_records],
        dtype=np.int64,
    )
    if not np.array_equal(labels, gen_labels):
        raise ValueError(
            f"DisPRM and {second_stage_name} labels do not match."
        )

    dis_scores = np.asarray(
        [float(record["disprm_score"]) for record in ordered_dis_records],
        dtype=np.float64,
    )
    dis_predictions = (dis_scores >= args.threshold).astype(np.int64)
    gen_predictions = np.asarray(
        [
            int(record[prediction_field])
            for record in ordered_gen_records
        ],
        dtype=np.int64,
    )

    dis_correct = dis_predictions == labels
    gen_correct = gen_predictions == labels
    gain_labels = gen_correct.astype(np.int64) - dis_correct.astype(np.int64)
    benefit_labels = (gain_labels == 1).astype(np.int64)
    harm_labels = (gain_labels == -1).astype(np.int64)

    class_counts = {
        gain_class: int(np.sum(gain_labels == gain_class))
        for gain_class in GAIN_CLASSES
    }
    if min(class_counts.values()) < args.folds:
        raise ValueError(
            "Insufficient gain-class counts for "
            f"{args.folds}-fold grouped evaluation."
        )

    features = np.asarray(
        [
            build_features(
                record=record,
                score=float(score),
                threshold=args.threshold,
            )
            for record, score in zip(ordered_dis_records, dis_scores)
        ],
        dtype=np.float64,
    )

    splitter = StratifiedGroupKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    total = len(common_keys)
    gain_probabilities = np.zeros((total, 3), dtype=np.float64)
    benefit_only_probabilities = np.zeros(total, dtype=np.float64)
    fold_ids = np.zeros(total, dtype=np.int64)
    class_to_column = {label: index for index, label in enumerate(GAIN_CLASSES)}

    for fold_id, (train_indices, test_indices) in enumerate(
        splitter.split(features, gain_labels, groups),
        start=1,
    ):
        if set(np.unique(gain_labels[train_indices])) != set(GAIN_CLASSES):
            raise ValueError(
                f"Fold {fold_id} training data lacks a gain class."
            )
        if len(np.unique(benefit_labels[train_indices])) != 2:
            raise ValueError(
                f"Fold {fold_id} training data lacks a benefit class."
            )

        gain_router = make_expected_gain_router(args.seed + fold_id)
        gain_router.fit(features[train_indices], gain_labels[train_indices])
        fold_probabilities = gain_router.predict_proba(features[test_indices])

        gain_classes = gain_router.named_steps["classifier"].classes_
        for source_column, gain_class in enumerate(gain_classes):
            target_column = class_to_column[int(gain_class)]
            gain_probabilities[test_indices, target_column] = (
                fold_probabilities[:, source_column]
            )

        benefit_router = make_router(args.seed + fold_id)
        benefit_router.fit(
            features[train_indices],
            benefit_labels[train_indices],
        )
        benefit_only_probabilities[test_indices] = (
            benefit_router.predict_proba(features[test_indices])[:, 1]
        )
        fold_ids[test_indices] = fold_id

    harm_probability = gain_probabilities[:, class_to_column[-1]]
    neutral_probability = gain_probabilities[:, class_to_column[0]]
    benefit_probability = gain_probabilities[:, class_to_column[1]]
    expected_gain = benefit_probability - harm_probability

    metrics = {
        "benefit_only_roc_auc": float(
            roc_auc_score(benefit_labels, benefit_only_probabilities)
        ),
        "benefit_only_average_precision": float(
            average_precision_score(
                benefit_labels,
                benefit_only_probabilities,
            )
        ),
        "harm_aware_benefit_roc_auc": float(
            roc_auc_score(benefit_labels, benefit_probability)
        ),
        "harm_aware_benefit_average_precision": float(
            average_precision_score(benefit_labels, benefit_probability)
        ),
        "harm_aware_harm_roc_auc": float(
            roc_auc_score(harm_labels, harm_probability)
        ),
        "harm_aware_harm_average_precision": float(
            average_precision_score(harm_labels, harm_probability)
        ),
    }

    expected_gain_order = ordered_indices(expected_gain)
    benefit_only_order = ordered_indices(benefit_only_probabilities)
    uncertainty_order = sorted(
        range(total),
        key=lambda index: (
            abs(float(dis_scores[index]) - args.threshold),
            index,
        ),
    )

    beneficial_indices = set(np.flatnonzero(gain_labels == 1).tolist())
    harmful_indices = set(np.flatnonzero(gain_labels == -1).tolist())
    base_accuracy = routing_accuracy(
        labels,
        dis_predictions,
        gen_predictions,
        set(),
    )

    dis_runtime = mean(
        float(record["disprm_runtime_seconds"])
        for record in ordered_dis_records
    )
    gen_runtime = mean(
        float(record[runtime_field])
        for record in ordered_gen_records
    )

    print(f"Examples:                         {total}")
    print(f"Dataset split:                    {split_name}")
    print(f"Original-question groups:         {len(set(groups.tolist()))}")
    print(f"DisPRM threshold:                 {args.threshold:.2f}")
    print(f"Beneficial calls (+1):            {class_counts[1]}")
    print(f"Neutral calls (0):                {class_counts[0]}")
    print(f"Harmful calls (-1):               {class_counts[-1]}")
    print()
    print("Grouped OOF ranking metrics")
    print("-" * 72)
    print(
        "Benefit-only router, benefit AP: "
        f"{metrics['benefit_only_average_precision']:.4f}"
    )
    print(
        "Expected-gain router, benefit AP: "
        f"{metrics['harm_aware_benefit_average_precision']:.4f}"
    )
    print(
        "Expected-gain router, harm AP:    "
        f"{metrics['harm_aware_harm_average_precision']:.4f}"
    )
    print()
    print(
        "Budget | Expected gain | Benefit only | Uncertainty | "
        "Random mean+/-std | Oracle | EG B/H | BO B/H | Cost"
    )
    print("-" * 112)

    rng = random.Random(args.seed)
    budget_results: list[dict[str, Any]] = []

    for budget_percent in args.budgets:
        budget = round(total * budget_percent / 100)
        expected_routed = set(expected_gain_order[:budget])
        benefit_routed = set(benefit_only_order[:budget])
        uncertainty_routed = set(uncertainty_order[:budget])

        expected_accuracy = routing_accuracy(
            labels, dis_predictions, gen_predictions, expected_routed
        )
        benefit_accuracy = routing_accuracy(
            labels, dis_predictions, gen_predictions, benefit_routed
        )
        uncertainty_accuracy = routing_accuracy(
            labels, dis_predictions, gen_predictions, uncertainty_routed
        )

        random_accuracies = [
            routing_accuracy(
                labels,
                dis_predictions,
                gen_predictions,
                set(rng.sample(range(total), budget)),
            )
            for _ in range(args.random_trials)
        ]

        expected_beneficial = len(expected_routed & beneficial_indices)
        expected_harmful = len(expected_routed & harmful_indices)
        benefit_beneficial = len(benefit_routed & beneficial_indices)
        benefit_harmful = len(benefit_routed & harmful_indices)
        oracle_accuracy = (
            base_accuracy
            + min(budget, len(beneficial_indices)) / total
        )
        cost_ratio = (
            (
                dis_runtime
                + (budget / total) * gen_runtime
            )
            / dis_runtime
            if args.second_stage_kind == "genprm"
            else None
        )

        result = {
            "budget_percent": budget_percent,
            "routed_examples": budget,
            "expected_gain_accuracy": expected_accuracy,
            "benefit_only_accuracy": benefit_accuracy,
            "uncertainty_accuracy": uncertainty_accuracy,
            "random_accuracy_mean": mean(random_accuracies),
            "random_accuracy_std": pstdev(random_accuracies),
            "oracle_accuracy": oracle_accuracy,
            "expected_gain_beneficial": expected_beneficial,
            "expected_gain_harmful": expected_harmful,
            "benefit_only_beneficial": benefit_beneficial,
            "benefit_only_harmful": benefit_harmful,
            "cascade_cost_ratio": cost_ratio,
        }
        budget_results.append(result)

        random_text = (
            f"{result['random_accuracy_mean']:.3f}"
            f"+/-{result['random_accuracy_std']:.3f}"
        )
        cost_text = (
            f"{cost_ratio:.2f}x"
            if cost_ratio is not None
            else "n/a"
        )

        print(
            f"{budget_percent:>5}% | "
            f"{expected_accuracy:>13.3f} | "
            f"{benefit_accuracy:>12.3f} | "
            f"{uncertainty_accuracy:>11.3f} | "
            f"{random_text:>17} | "
            f"{oracle_accuracy:>6.3f} | "
            f"{expected_beneficial:02d}/{expected_harmful:02d} | "
            f"{benefit_beneficial:02d}/{benefit_harmful:02d} | "
            f"{cost_text:>5}"
        )

    output_records = []
    for index, item_key in enumerate(common_keys):
        output_records.append(
            {
                "example_id": item_key[0],
                "current_step": item_key[1],
                "label": int(labels[index]),
                "fold": int(fold_ids[index]),
                "disprm_score": float(dis_scores[index]),
                "disprm_prediction": int(dis_predictions[index]),
                "second_stage_prediction": int(
                    gen_predictions[index]
                ),
                "gain_label": int(gain_labels[index]),
                "probability_harmful": float(harm_probability[index]),
                "probability_neutral": float(neutral_probability[index]),
                "probability_beneficial": float(benefit_probability[index]),
                "predicted_expected_gain": float(expected_gain[index]),
                "benefit_only_probability": float(
                    benefit_only_probabilities[index]
                ),
            }
        )

    output = {
        "config": {
            "disprm": str(args.disprm),
            "genprm": (
                str(second_stage_path)
                if args.second_stage_kind == "genprm"
                else None
            ),
            "second_stage": str(second_stage_path),
            "second_stage_kind": args.second_stage_kind,
            "dataset": args.dataset,
            "dataset_split": split_name,
            "threshold": args.threshold,
            "folds": args.folds,
            "seed": args.seed,
            "grouping": "normalized_original_question",
        },
        "class_counts": {
            "harmful": class_counts[-1],
            "neutral": class_counts[0],
            "beneficial": class_counts[1],
        },
        "metrics": metrics,
        "budgets": budget_results,
        "records": output_records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print()
    print(f"Results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
