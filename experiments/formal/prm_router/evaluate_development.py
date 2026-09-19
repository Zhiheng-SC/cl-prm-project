"""Select the formal PRM router on train/validation data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from cl_prm.data.records import (
    align_verifier_records as align,
    correctness_arrays,
)
from cl_prm.evaluation.cost import cost_features, make_cost_predictor
from cl_prm.evaluation.routing import (
    GAIN_CLASSES,
    gain_probabilities,
    make_expected_gain_router,
    router_features,
    routed_accuracy,
    safe_fraction,
    top_budget_indices,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments" / "prm_router_formal.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "formal"
    / "development_selection.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--train-reasoneval", type=Path, required=True)
    parser.add_argument("--train-pathfinder", type=Path, required=True)
    parser.add_argument("--validation-reasoneval", type=Path, required=True)
    parser.add_argument("--validation-pathfinder", type=Path, required=True)
    parser.add_argument(
        "--reason-eval-threshold",
        type=float,
        required=True,
        help="Validation-selected ReasonEval threshold.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def bin_diagnostic(
    values: np.ndarray,
    gain_labels: np.ndarray,
    bins: int,
) -> list[dict[str, Any]]:
    if len(values) == 0:
        return []
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if len(edges) <= 1:
        return []

    results = []
    for index in range(len(edges) - 1):
        left, right = float(edges[index]), float(edges[index + 1])
        if index == len(edges) - 2:
            mask = (values >= left) & (values <= right)
        else:
            mask = (values >= left) & (values < right)
        n = int(mask.sum())
        if n == 0:
            continue
        results.append(
            {
                "bin": index,
                "left": left,
                "right": right,
                "examples": n,
                "beneficial_rate": float(np.mean(gain_labels[mask] == 1)),
                "harmful_rate": float(np.mean(gain_labels[mask] == -1)),
            }
        )
    return results


def category_diagnostic(
    records: list[dict[str, Any]],
    gain_labels: np.ndarray,
    field: str,
) -> list[dict[str, Any]]:
    categories = sorted({str(row.get(field, "unknown")) for row in records})
    results = []
    for category in categories:
        mask = np.asarray(
            [str(row.get(field, "unknown")) == category for row in records],
            dtype=bool,
        )
        n = int(mask.sum())
        results.append(
            {
                "category": category,
                "examples": n,
                "beneficial_rate": float(np.mean(gain_labels[mask] == 1)),
                "harmful_rate": float(np.mean(gain_labels[mask] == -1)),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.reason_eval_threshold <= 1.0:
        raise ValueError("--reason-eval-threshold must be in [0, 1].")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    routing_config = config["routing"]
    cost_config = routing_config["cost_aware"]
    diagnostic_config = config["evaluation"]["diagnostic_analyses"]

    train_base, train_second = align(
        args.train_reasoneval,
        args.train_pathfinder,
    )
    val_base, val_second = align(
        args.validation_reasoneval,
        args.validation_pathfinder,
    )

    train = correctness_arrays(
        train_base,
        train_second,
        args.reason_eval_threshold,
    )
    val = correctness_arrays(
        val_base,
        val_second,
        args.reason_eval_threshold,
    )

    missing_classes = set(GAIN_CLASSES) - set(np.unique(train["gain_labels"]).tolist())
    if missing_classes:
        raise ValueError(
            f"Training split is missing gain classes: {sorted(missing_classes)}"
        )

    x_train = router_features(
        train_base,
        train["scores"],
        args.reason_eval_threshold,
    )
    x_val = router_features(
        val_base,
        val["scores"],
        args.reason_eval_threshold,
    )

    router = make_expected_gain_router(seed=int(config["formal_splits"]["seed"]))
    router.fit(x_train, train["gain_labels"])
    val_probabilities = gain_probabilities(router, x_val)

    class_to_column = {label: index for index, label in enumerate(GAIN_CLASSES)}
    p_harm = val_probabilities[:, class_to_column[-1]]
    p_benefit = val_probabilities[:, class_to_column[1]]

    x_cost_train = cost_features(train_base)
    x_cost_val = cost_features(val_base)
    train_runtime = np.asarray(
        [float(row["pathfinder_runtime_seconds"]) for row in train_second],
        dtype=np.float64,
    )
    val_runtime = np.asarray(
        [float(row["pathfinder_runtime_seconds"]) for row in val_second],
        dtype=np.float64,
    )

    cost_predictor = make_cost_predictor(alpha=float(cost_config["ridge_alpha"]))
    cost_predictor.fit(x_cost_train, train_runtime)
    predicted_val_runtime = np.maximum(cost_predictor.predict(x_cost_val), 0.0)

    train_median_runtime = float(np.median(train_runtime))
    if train_median_runtime <= 0.0:
        raise ValueError("Median training PathFinder runtime must be positive.")
    normalized_predicted_cost = predicted_val_runtime / train_median_runtime

    primary_budget = float(routing_config["primary_call_budget"])
    candidates = []
    for lambda_h in cost_config["lambda_h_candidates"]:
        for mu in cost_config["mu_candidates"]:
            utility = (
                p_benefit
                - float(lambda_h) * p_harm
                - float(mu) * normalized_predicted_cost
            )
            selected = top_budget_indices(utility, primary_budget)
            accuracy = routed_accuracy(
                val["labels"],
                val["base_predictions"],
                val["second_predictions"],
                selected,
            )
            realized_extra_runtime = float(np.sum(val_runtime[selected]))
            candidates.append(
                {
                    "lambda_h": float(lambda_h),
                    "mu": float(mu),
                    "validation_accuracy": accuracy,
                    "called_examples": int(len(selected)),
                    "measured_pathfinder_runtime_seconds": realized_extra_runtime,
                    "mean_predicted_pathfinder_runtime_seconds": (
                        float(np.mean(predicted_val_runtime[selected]))
                        if len(selected)
                        else 0.0
                    ),
                }
            )

    candidates.sort(
        key=lambda row: (
            -row["validation_accuracy"],
            row["measured_pathfinder_runtime_seconds"],
            row["lambda_h"],
            row["mu"],
        )
    )
    selected_config = candidates[0]

    base_accuracy = float(np.mean(val["base_correct"]))
    second_accuracy = float(np.mean(val["second_correct"]))
    oracle_accuracy = float(np.mean(val["base_correct"] | val["second_correct"]))
    expected_gain = p_benefit - p_harm
    expected_gain_selected = top_budget_indices(expected_gain, primary_budget)
    expected_gain_accuracy = routed_accuracy(
        val["labels"],
        val["base_predictions"],
        val["second_predictions"],
        expected_gain_selected,
    )
    captured_headroom = safe_fraction(
        expected_gain_accuracy - base_accuracy,
        oracle_accuracy - base_accuracy,
    )

    step_position = np.asarray(
        [
            float(
                row.get(
                    "step_position",
                    int(row["current_step"]) / max(int(row.get("total_steps", 1)), 1),
                )
            )
            for row in val_base
        ],
        dtype=np.float64,
    )
    input_length = np.asarray(
        [float(row.get("disprm_input_tokens", 0)) for row in val_base],
        dtype=np.float64,
    )

    output = {
        "schema_version": 1,
        "scope": "formal_train_validation_only",
        "reason_eval_threshold": float(args.reason_eval_threshold),
        "primary_call_budget": primary_budget,
        "validation": {
            "examples": len(val_base),
            "reason_eval_accuracy": base_accuracy,
            "pathfinder_accuracy": second_accuracy,
            "oracle_accuracy": oracle_accuracy,
            "beneficial_calls": int(np.sum(val["gain_labels"] == 1)),
            "neutral_calls": int(np.sum(val["gain_labels"] == 0)),
            "harmful_calls": int(np.sum(val["gain_labels"] == -1)),
            "expected_gain_accuracy": expected_gain_accuracy,
            "captured_oracle_headroom": captured_headroom,
        },
        "cost_predictor": {
            "model": cost_config["predictor_model"],
            "ridge_alpha": float(cost_config["ridge_alpha"]),
            "train_median_pathfinder_runtime_seconds": train_median_runtime,
            "validation_mae_seconds": float(
                np.mean(np.abs(predicted_val_runtime - val_runtime))
            ),
            "validation_mean_actual_runtime_seconds": float(np.mean(val_runtime)),
            "validation_mean_predicted_runtime_seconds": float(
                np.mean(predicted_val_runtime)
            ),
        },
        "cost_aware_selection": {
            "selection_rule": cost_config["selection_rule"],
            "selected": selected_config,
            "candidates": candidates,
        },
        "diagnostics": {
            "reason_eval_confidence": bin_diagnostic(
                val["scores"],
                val["gain_labels"],
                int(diagnostic_config["confidence_bins"]),
            ),
            "step_position": bin_diagnostic(
                step_position,
                val["gain_labels"],
                int(diagnostic_config["step_position_bins"]),
            ),
            "input_length": bin_diagnostic(
                input_length,
                val["gain_labels"],
                int(diagnostic_config["input_length_bins"]),
            ),
            "error_type": category_diagnostic(
                val_base,
                val["gain_labels"],
                str(diagnostic_config["error_type_field"]),
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Formal development selection complete")
    print(f"Validation examples:       {len(val_base)}")
    print(f"ReasonEval accuracy:       {base_accuracy:.4f}")
    print(f"PathFinder accuracy:       {second_accuracy:.4f}")
    print(f"Oracle accuracy:           {oracle_accuracy:.4f}")
    print(f"Expected-gain @ 20%:       {expected_gain_accuracy:.4f}")
    print(
        "Selected cost-aware:      "
        f"lambda_h={selected_config['lambda_h']:.3g}, "
        f"mu={selected_config['mu']:.3g}, "
        f"accuracy={selected_config['validation_accuracy']:.4f}"
    )
    print(f"Output:                    {args.output}")


if __name__ == "__main__":
    main()
