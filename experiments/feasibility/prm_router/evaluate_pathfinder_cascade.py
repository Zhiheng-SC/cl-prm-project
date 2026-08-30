"""Evaluate pre-call routing and post-call PathFinder arbitration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from train_benefit_router import (
    build_features,
    read_records,
    routing_accuracy,
)
from train_expected_gain_router import make_expected_gain_router


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DISPRM = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "reasoneval_feasibility_100.jsonl"
)

DEFAULT_PATHFINDER = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "pathfinder"
    / "pathfinder_feasibility_100.jsonl"
)

DEFAULT_PRE_ROUTER = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "robustness"
    / "pathfinder_expected_gain_router.json"
)

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "robustness"
    / "pathfinder_cascade_arbitration.json"
)

DEFAULT_BUDGETS = [0, 10, 20, 30, 40, 50, 75, 100]
GAIN_CLASSES = [-1, 0, 1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--disprm",
        type=Path,
        default=DEFAULT_DISPRM,
    )
    parser.add_argument(
        "--pathfinder",
        type=Path,
        default=DEFAULT_PATHFINDER,
    )
    parser.add_argument(
        "--pre-router",
        type=Path,
        default=DEFAULT_PRE_ROUTER,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=DEFAULT_BUDGETS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def record_key(
    record: dict[str, Any],
) -> tuple[str, int]:
    return (
        str(record["example_id"]),
        int(record["current_step"]),
    )


def index_records(
    records: list[dict[str, Any]],
    source_name: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    indexed = {}

    for record in records:
        key = record_key(record)

        if key in indexed:
            raise ValueError(
                f"Duplicate {source_name} record: {key}"
            )

        indexed[key] = record

    return indexed


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_descending_order(
    scores: np.ndarray,
) -> list[int]:
    return sorted(
        range(len(scores)),
        key=lambda index: (
            -float(scores[index]),
            index,
        ),
    )


def cross_fitted_gain_probabilities(
    features: np.ndarray,
    gain_labels: np.ndarray,
    fold_ids: np.ndarray,
    seed: int,
) -> np.ndarray:
    probabilities = np.zeros(
        (len(features), 3),
        dtype=np.float64,
    )

    class_to_column = {
        gain_class: index
        for index, gain_class in enumerate(GAIN_CLASSES)
    }

    for fold_id in sorted(np.unique(fold_ids)):
        test_indices = np.flatnonzero(
            fold_ids == fold_id
        )
        train_indices = np.flatnonzero(
            fold_ids != fold_id
        )

        training_classes = set(
            np.unique(gain_labels[train_indices]).tolist()
        )

        if training_classes != set(GAIN_CLASSES):
            raise ValueError(
                f"Fold {fold_id} lacks a gain class: "
                f"{training_classes}"
            )

        model = make_expected_gain_router(
            seed + int(fold_id)
        )
        model.fit(
            features[train_indices],
            gain_labels[train_indices],
        )

        fold_probabilities = model.predict_proba(
            features[test_indices]
        )

        model_classes = model.named_steps[
            "classifier"
        ].classes_

        for source_column, gain_class in enumerate(
            model_classes
        ):
            target_column = class_to_column[
                int(gain_class)
            ]
            probabilities[
                test_indices,
                target_column,
            ] = fold_probabilities[:, source_column]

    return probabilities


def replacement_statistics(
    labels: np.ndarray,
    dis_predictions: np.ndarray,
    pathfinder_predictions: np.ndarray,
    replaced_indices: set[int],
    beneficial_indices: set[int],
    harmful_indices: set[int],
) -> dict[str, Any]:
    return {
        "accuracy": routing_accuracy(
            labels,
            dis_predictions,
            pathfinder_predictions,
            replaced_indices,
        ),
        "replacements": len(replaced_indices),
        "beneficial_replacements": len(
            replaced_indices & beneficial_indices
        ),
        "harmful_replacements": len(
            replaced_indices & harmful_indices
        ),
    }


def main() -> None:
    args = parse_args()

    for budget in args.budgets:
        if not 0 <= budget <= 100:
            raise ValueError(
                f"Invalid budget: {budget}"
            )

    dis_records = index_records(
        read_records(args.disprm),
        "DisPRM",
    )
    pathfinder_records = index_records(
        read_records(args.pathfinder),
        "PathFinder",
    )

    pre_router_payload = read_json(args.pre_router)
    pre_router_records = index_records(
        pre_router_payload["records"],
        "pre-router",
    )

    common_keys = sorted(
        set(dis_records)
        & set(pathfinder_records)
        & set(pre_router_records)
    )

    if not (
        len(common_keys)
        == len(dis_records)
        == len(pathfinder_records)
        == len(pre_router_records)
    ):
        raise ValueError(
            "DisPRM, PathFinder, and pre-router "
            "records do not match."
        )

    ordered_dis = [
        dis_records[key]
        for key in common_keys
    ]
    ordered_pathfinder = [
        pathfinder_records[key]
        for key in common_keys
    ]
    ordered_pre_router = [
        pre_router_records[key]
        for key in common_keys
    ]

    threshold = float(
        pre_router_payload["config"]["threshold"]
    )

    labels = np.asarray(
        [
            int(record["label"])
            for record in ordered_dis
        ],
        dtype=np.int64,
    )

    pathfinder_labels = np.asarray(
        [
            int(record["label"])
            for record in ordered_pathfinder
        ],
        dtype=np.int64,
    )

    if not np.array_equal(
        labels,
        pathfinder_labels,
    ):
        raise ValueError(
            "DisPRM and PathFinder labels do not match."
        )

    dis_scores = np.asarray(
        [
            float(record["disprm_score"])
            for record in ordered_dis
        ],
        dtype=np.float64,
    )

    dis_predictions = (
        dis_scores >= threshold
    ).astype(np.int64)

    pathfinder_predictions = np.asarray(
        [
            int(record["pathfinder_prediction"])
            for record in ordered_pathfinder
        ],
        dtype=np.int64,
    )

    dis_correct = dis_predictions == labels
    pathfinder_correct = (
        pathfinder_predictions == labels
    )

    gain_labels = (
        pathfinder_correct.astype(np.int64)
        - dis_correct.astype(np.int64)
    )

    saved_gain_labels = np.asarray(
        [
            int(record["gain_label"])
            for record in ordered_pre_router
        ],
        dtype=np.int64,
    )

    if not np.array_equal(
        gain_labels,
        saved_gain_labels,
    ):
        raise ValueError(
            "Recomputed gain labels do not match "
            "the saved pre-router output."
        )

    fold_ids = np.asarray(
        [
            int(record["fold"])
            for record in ordered_pre_router
        ],
        dtype=np.int64,
    )

    pre_expected_gain = np.asarray(
        [
            float(
                record["predicted_expected_gain"]
            )
            for record in ordered_pre_router
        ],
        dtype=np.float64,
    )

    generic_features = np.asarray(
        [
            build_features(
                record=record,
                score=float(score),
                threshold=threshold,
            )
            for record, score in zip(
                ordered_dis,
                dis_scores,
            )
        ],
        dtype=np.float64,
    )

    overall_features = np.asarray(
        [
            [
                float(
                    record[
                        "pathfinder_official_score"
                    ]
                ),
                float(
                    record[
                        "pathfinder_gate_passed"
                    ]
                ),
                float(
                    record[
                        "pathfinder_prediction"
                    ]
                ),
            ]
            for record in ordered_pathfinder
        ],
        dtype=np.float64,
    )

    fine_grained_features = np.asarray(
        [
            [
                float(
                    record[
                        "pathfinder_math_probability"
                    ]
                ),
                float(
                    record[
                        "pathfinder_consistency_probability"
                    ]
                ),
                float(
                    record[
                        "pathfinder_optimality_probability"
                    ]
                ),
                float(
                    record[
                        "pathfinder_math_prediction"
                    ]
                ),
                float(
                    record[
                        "pathfinder_consistency_prediction"
                    ]
                ),
            ]
            for record in ordered_pathfinder
        ],
        dtype=np.float64,
    )

    feature_sets = {
        "overall": np.column_stack(
            [
                generic_features,
                overall_features,
            ]
        ),
        "fine_grained": np.column_stack(
            [
                generic_features,
                fine_grained_features,
            ]
        ),
        "all_signals": np.column_stack(
            [
                generic_features,
                overall_features,
                fine_grained_features,
            ]
        ),
    }

    benefit_labels = (
        gain_labels == 1
    ).astype(np.int64)

    harm_labels = (
        gain_labels == -1
    ).astype(np.int64)

    class_to_column = {
        gain_class: index
        for index, gain_class in enumerate(
            GAIN_CLASSES
        )
    }

    post_call_results = {}

    print(f"Examples:                  {len(labels)}")
    print(f"DisPRM threshold:          {threshold:.2f}")
    print(
        "DisPRM accuracy:           "
        f"{np.mean(dis_correct):.4f}"
    )
    print(
        "PathFinder accuracy:       "
        f"{np.mean(pathfinder_correct):.4f}"
    )
    print(
        "Beneficial calls:          "
        f"{int(np.sum(benefit_labels))}"
    )
    print(
        "Harmful calls:             "
        f"{int(np.sum(harm_labels))}"
    )
    print()

    print("Post-call grouped OOF metrics")
    print("-" * 70)
    print(
        "Feature set       | Benefit ROC | Benefit AP "
        "| Harm ROC | Harm AP"
    )
    print("-" * 70)

    for feature_name, features in feature_sets.items():
        probabilities = (
            cross_fitted_gain_probabilities(
                features=features,
                gain_labels=gain_labels,
                fold_ids=fold_ids,
                seed=args.seed,
            )
        )

        harm_probability = probabilities[
            :,
            class_to_column[-1],
        ]
        benefit_probability = probabilities[
            :,
            class_to_column[1],
        ]
        expected_gain = (
            benefit_probability
            - harm_probability
        )

        metrics = {
            "benefit_roc_auc": float(
                roc_auc_score(
                    benefit_labels,
                    benefit_probability,
                )
            ),
            "benefit_average_precision": float(
                average_precision_score(
                    benefit_labels,
                    benefit_probability,
                )
            ),
            "harm_roc_auc": float(
                roc_auc_score(
                    harm_labels,
                    harm_probability,
                )
            ),
            "harm_average_precision": float(
                average_precision_score(
                    harm_labels,
                    harm_probability,
                )
            ),
        }

        post_call_results[feature_name] = {
            "probabilities": probabilities,
            "expected_gain": expected_gain,
            "metrics": metrics,
        }

        print(
            f"{feature_name:<17} | "
            f"{metrics['benefit_roc_auc']:>11.3f} | "
            f"{metrics['benefit_average_precision']:>10.3f} | "
            f"{metrics['harm_roc_auc']:>8.3f} | "
            f"{metrics['harm_average_precision']:>7.3f}"
        )

    pre_order = stable_descending_order(
        pre_expected_gain
    )

    beneficial_indices = set(
        np.flatnonzero(
            gain_labels == 1
        ).tolist()
    )

    harmful_indices = set(
        np.flatnonzero(
            gain_labels == -1
        ).tolist()
    )

    base_accuracy = float(
        np.mean(dis_correct)
    )

    print()
    print("Budgeted PathFinder cascade")
    print("-" * 105)
    print(
        "Budget | Always | Overall | Fine | All | "
        "Oracle | Fine B/H | All B/H"
    )
    print("-" * 105)

    budget_results = []

    for budget_percent in args.budgets:
        budget = round(
            len(labels)
            * budget_percent
            / 100
        )

        called_indices = set(
            pre_order[:budget]
        )

        always_statistics = (
            replacement_statistics(
                labels=labels,
                dis_predictions=dis_predictions,
                pathfinder_predictions=(
                    pathfinder_predictions
                ),
                replaced_indices=called_indices,
                beneficial_indices=(
                    beneficial_indices
                ),
                harmful_indices=harmful_indices,
            )
        )

        result = {
            "budget_percent": budget_percent,
            "called_examples": budget,
            "always_replace": always_statistics,
            "arbitration": {},
        }

        for feature_name, model_result in (
            post_call_results.items()
        ):
            expected_gain = model_result[
                "expected_gain"
            ]

            replaced_indices = {
                index
                for index in called_indices
                if expected_gain[index] > 0.0
            }

            result["arbitration"][
                feature_name
            ] = replacement_statistics(
                labels=labels,
                dis_predictions=dis_predictions,
                pathfinder_predictions=(
                    pathfinder_predictions
                ),
                replaced_indices=replaced_indices,
                beneficial_indices=(
                    beneficial_indices
                ),
                harmful_indices=harmful_indices,
            )

        oracle_accuracy = (
            base_accuracy
            + min(
                budget,
                len(beneficial_indices),
            )
            / len(labels)
        )

        result["oracle_accuracy"] = (
            oracle_accuracy
        )
        budget_results.append(result)

        overall = result["arbitration"][
            "overall"
        ]
        fine = result["arbitration"][
            "fine_grained"
        ]
        all_signals = result["arbitration"][
            "all_signals"
        ]

        print(
            f"{budget_percent:>5}% | "
            f"{always_statistics['accuracy']:>6.3f} | "
            f"{overall['accuracy']:>7.3f} | "
            f"{fine['accuracy']:>4.3f} | "
            f"{all_signals['accuracy']:>4.3f} | "
            f"{oracle_accuracy:>6.3f} | "
            f"{fine['beneficial_replacements']:02d}/"
            f"{fine['harmful_replacements']:02d} | "
            f"{all_signals['beneficial_replacements']:02d}/"
            f"{all_signals['harmful_replacements']:02d}"
        )

    output_records = []

    for index, item_key in enumerate(common_keys):
        output_record = {
            "example_id": item_key[0],
            "current_step": item_key[1],
            "label": int(labels[index]),
            "fold": int(fold_ids[index]),
            "gain_label": int(
                gain_labels[index]
            ),
            "pre_call_expected_gain": float(
                pre_expected_gain[index]
            ),
        }

        for feature_name, model_result in (
            post_call_results.items()
        ):
            output_record[
                f"{feature_name}_post_expected_gain"
            ] = float(
                model_result[
                    "expected_gain"
                ][index]
            )

        output_records.append(output_record)

    output = {
        "config": {
            "disprm": str(args.disprm),
            "pathfinder": str(
                args.pathfinder
            ),
            "pre_router": str(
                args.pre_router
            ),
            "threshold": threshold,
            "seed": args.seed,
            "post_call_acceptance_rule": (
                "predicted_expected_gain > 0"
            ),
        },
        "class_counts": {
            "beneficial": int(
                np.sum(gain_labels == 1)
            ),
            "neutral": int(
                np.sum(gain_labels == 0)
            ),
            "harmful": int(
                np.sum(gain_labels == -1)
            ),
        },
        "post_call_metrics": {
            feature_name: result["metrics"]
            for feature_name, result in (
                post_call_results.items()
            )
        },
        "budgets": budget_results,
        "records": output_records,
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Results saved to: "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()