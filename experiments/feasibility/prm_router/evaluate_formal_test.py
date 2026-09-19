"""Evaluate the frozen PRM Router exactly once on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_formal_development import (
    align,
    correctness_arrays,
    cost_features,
    gain_probabilities,
    make_cost_predictor,
    make_expected_gain_router,
    router_features,
    safe_fraction,
    top_budget_indices,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments" / "prm_router_formal.json"
DEFAULT_THRESHOLD = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "formal"
    / "validation"
    / "reasoneval_threshold.json"
)
DEFAULT_SELECTION = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "formal"
    / "development_selection.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "formal"
    / "test"
    / "formal_test_results.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--threshold-selection", type=Path, default=DEFAULT_THRESHOLD)
    parser.add_argument("--development-selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--train-reasoneval", type=Path, required=True)
    parser.add_argument("--train-pathfinder", type=Path, required=True)
    parser.add_argument("--test-reasoneval", type=Path, required=True)
    parser.add_argument("--test-pathfinder", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--confirm-protocol-frozen",
        action="store_true",
        help="Required acknowledgement that validation choices are frozen.",
    )
    return parser.parse_args()


def make_binary_predictor(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


def final_predictions(
    base_predictions: np.ndarray,
    second_predictions: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    result = base_predictions.copy()
    result[selected] = second_predictions[selected]
    return result


def selected_runtime(
    base_runtime: np.ndarray,
    second_runtime: np.ndarray,
    selected: np.ndarray,
) -> float:
    return float(np.sum(base_runtime) + np.sum(second_runtime[selected]))


def bootstrap_group_metric(
    values: np.ndarray,
    group_ids: np.ndarray,
    samples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    groups = sorted(set(group_ids.tolist()))
    indices_by_group = {
        group: np.flatnonzero(group_ids == group)
        for group in groups
    }
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)

    for sample_id in range(samples):
        drawn = rng.choice(groups, size=len(groups), replace=True)
        indices = np.concatenate([indices_by_group[group] for group in drawn])
        estimates[sample_id] = float(np.mean(values[indices]))

    alpha = (1.0 - confidence_level) / 2.0
    return {
        "estimate": float(np.mean(values)),
        "ci_low": float(np.quantile(estimates, alpha)),
        "ci_high": float(np.quantile(estimates, 1.0 - alpha)),
    }


def main() -> None:
    args = parse_args()
    if not args.confirm_protocol_frozen:
        raise ValueError(
            "Held-out evaluation is locked. Freeze all validation-selected "
            "choices first, then add --confirm-protocol-frozen."
        )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    threshold_payload = json.loads(
        args.threshold_selection.read_text(encoding="utf-8")
    )
    development_payload = json.loads(
        args.development_selection.read_text(encoding="utf-8")
    )

    if threshold_payload.get("selection_split") != "validation":
        raise ValueError("ReasonEval threshold was not selected on validation.")
    if development_payload.get("scope") != "formal_train_validation_only":
        raise ValueError("Development selection has an unexpected scope.")

    threshold = float(threshold_payload["selected_threshold"])
    primary_budget = float(config["routing"]["primary_call_budget"])
    seed = int(config["formal_splits"]["seed"])

    selected_cost = development_payload["cost_aware_selection"]["selected"]
    lambda_h = float(selected_cost["lambda_h"])
    mu = float(selected_cost["mu"])

    train_base, train_second = align(
        args.train_reasoneval,
        args.train_pathfinder,
    )
    test_base, test_second = align(
        args.test_reasoneval,
        args.test_pathfinder,
    )

    formal_splits = {str(row.get("formal_split", "")) for row in test_base}
    if formal_splits - {"test", ""}:
        raise ValueError(
            "Formal test evaluator received non-test ReasonEval records: "
            f"{sorted(formal_splits)}"
        )

    train = correctness_arrays(train_base, train_second, threshold)
    test = correctness_arrays(test_base, test_second, threshold)

    x_train = router_features(train_base, train["scores"], threshold)
    x_test = router_features(test_base, test["scores"], threshold)

    # Primary pair-specific expected-gain router. The model is fit only on the
    # formal training split; validation was used for selection, not refitting.
    gain_router = make_expected_gain_router(seed=seed)
    gain_router.fit(x_train, train["gain_labels"])
    gain_prob = gain_probabilities(gain_router, x_test)
    p_harm = gain_prob[:, 0]
    p_benefit = gain_prob[:, 2]
    expected_gain = p_benefit - p_harm

    # Difficulty-only baseline: predict whether ReasonEval is wrong.
    train_failure = (~train["base_correct"]).astype(np.int64)
    if len(np.unique(train_failure)) < 2:
        raise ValueError("Training split must contain both ReasonEval success and failure.")
    failure_model = make_binary_predictor(seed)
    failure_model.fit(x_train, train_failure)
    failure_score = failure_model.predict_proba(x_test)[:, 1]

    # Benefit-only baseline: model whether PathFinder can correct ReasonEval,
    # without explicitly penalizing harmful replacements.
    train_benefit = (train["gain_labels"] == 1).astype(np.int64)
    if len(np.unique(train_benefit)) < 2:
        raise ValueError("Training split must contain beneficial and non-beneficial calls.")
    benefit_model = make_binary_predictor(seed + 1)
    benefit_model.fit(x_train, train_benefit)
    benefit_score = benefit_model.predict_proba(x_test)[:, 1]

    # Heuristic baselines available before the PathFinder call.
    uncertainty_score = -np.abs(test["scores"] - threshold)
    low_score = -test["scores"]
    random_seed = int(config["routing"].get("random_baseline_seed", seed))
    random_score = np.random.default_rng(random_seed).random(len(test_base))

    # Cost-aware secondary method. Runtime is predicted from pre-call features;
    # realized test runtime is never used in the routing score.
    cost_config = config["routing"]["cost_aware"]
    cost_predictor = make_cost_predictor(alpha=float(cost_config["ridge_alpha"]))
    train_cost_x = cost_features(train_base)
    test_cost_x = cost_features(test_base)
    train_pf_runtime = np.asarray(
        [float(row["pathfinder_runtime_seconds"]) for row in train_second],
        dtype=np.float64,
    )
    cost_predictor.fit(train_cost_x, train_pf_runtime)
    predicted_test_runtime = np.maximum(
        cost_predictor.predict(test_cost_x),
        0.0,
    )
    train_median_runtime = float(np.median(train_pf_runtime))
    normalized_cost = predicted_test_runtime / train_median_runtime
    cost_utility = p_benefit - lambda_h * p_harm - mu * normalized_cost

    routing_scores = {
        "random": random_score,
        "low_score": low_score,
        "uncertainty": uncertainty_score,
        "failure_prediction": failure_score,
        "benefit_only": benefit_score,
        "expected_gain": expected_gain,
        "cost_aware": cost_utility,
    }
    selected_indices = {
        name: top_budget_indices(score, primary_budget)
        for name, score in routing_scores.items()
    }

    predictions = {
        "reasoneval_only": test["base_predictions"],
        "pathfinder_only": test["second_predictions"],
    }
    for name, selected in selected_indices.items():
        predictions[name] = final_predictions(
            test["base_predictions"],
            test["second_predictions"],
            selected,
        )

    labels = test["labels"]
    base_runtime = np.asarray(
        [float(row["disprm_runtime_seconds"]) for row in test_base],
        dtype=np.float64,
    )
    pf_runtime = np.asarray(
        [float(row["pathfinder_runtime_seconds"]) for row in test_second],
        dtype=np.float64,
    )

    runtime = {
        "reasoneval_only_total_seconds": float(np.sum(base_runtime)),
        "pathfinder_only_total_seconds": float(np.sum(pf_runtime)),
    }
    for name, selected in selected_indices.items():
        runtime[f"{name}_cascade_total_seconds"] = selected_runtime(
            base_runtime,
            pf_runtime,
            selected,
        )

    group_ids = np.asarray(
        [
            str(
                row.get("original_question_group")
                or row.get("original_question")
                or row["example_id"]
            )
            for row in test_base
        ],
        dtype=object,
    )

    bootstrap_samples = int(config["evaluation"]["bootstrap_samples"])
    confidence_level = float(config["evaluation"]["confidence_level"])

    accuracy = {}
    correct = {}
    for offset, (name, pred) in enumerate(predictions.items()):
        indicator = (pred == labels).astype(np.float64)
        correct[name] = indicator
        accuracy[name] = bootstrap_group_metric(
            indicator,
            group_ids,
            bootstrap_samples,
            confidence_level,
            seed + offset,
        )

    paired_differences = {}
    for offset, baseline in enumerate(
        ["reasoneval_only", "uncertainty", "failure_prediction"],
        start=100,
    ):
        difference = correct["expected_gain"] - correct[baseline]
        paired_differences[f"expected_gain_minus_{baseline}"] = (
            bootstrap_group_metric(
                difference,
                group_ids,
                bootstrap_samples,
                confidence_level,
                seed + offset,
            )
        )

    secondary_budgets = [
        float(value) for value in config["routing"].get("secondary_call_budgets", [])
    ]
    all_budgets = sorted(set([primary_budget, *secondary_budgets]))
    routing_curves: dict[str, list[dict[str, float | int]]] = {}
    for name, score in routing_scores.items():
        rows = []
        for budget in all_budgets:
            selected = top_budget_indices(score, budget)
            pred = final_predictions(
                test["base_predictions"],
                test["second_predictions"],
                selected,
            )
            rows.append(
                {
                    "call_budget": float(budget),
                    "called_examples": int(len(selected)),
                    "accuracy": float(np.mean(pred == labels)),
                    "cascade_total_seconds": selected_runtime(
                        base_runtime, pf_runtime, selected
                    ),
                    "beneficial_selected": int(
                        np.sum(test["gain_labels"][selected] == 1)
                    ),
                    "harmful_selected": int(
                        np.sum(test["gain_labels"][selected] == -1)
                    ),
                }
            )
        routing_curves[name] = rows

    oracle_accuracy = float(
        np.mean(test["base_correct"] | test["second_correct"])
    )
    expected_gain_accuracy = float(
        np.mean(predictions["expected_gain"] == labels)
    )
    base_accuracy = float(np.mean(test["base_correct"]))
    captured_headroom = safe_fraction(
        expected_gain_accuracy - base_accuracy,
        oracle_accuracy - base_accuracy,
    )

    output = {
        "schema_version": 1,
        "scope": "held_out_test_once",
        "protocol": {
            "reason_eval_threshold": threshold,
            "primary_call_budget": primary_budget,
            "secondary_call_budgets": secondary_budgets,
            "random_baseline_seed": random_seed,
            "lambda_h": lambda_h,
            "mu": mu,
            "router_fit_split": "train",
            "selection_split": "validation",
            "evaluation_split": "test",
            "bootstrap_unit": config["evaluation"]["bootstrap_unit"],
            "bootstrap_samples": bootstrap_samples,
            "confidence_level": confidence_level,
        },
        "counts": {
            "examples": len(test_base),
            "original_question_groups": len(set(group_ids.tolist())),
            "beneficial_calls": int(np.sum(test["gain_labels"] == 1)),
            "neutral_calls": int(np.sum(test["gain_labels"] == 0)),
            "harmful_calls": int(np.sum(test["gain_labels"] == -1)),
        },
        "accuracy": accuracy,
        "paired_differences": paired_differences,
        "runtime": runtime,
        "routing_curves": routing_curves,
        "oracle": {
            "pairwise_accuracy": oracle_accuracy,
            "captured_headroom_expected_gain": captured_headroom,
        },
        "cost_aware": {
            "ridge_alpha": float(cost_config["ridge_alpha"]),
            "mean_predicted_pathfinder_runtime_seconds": float(
                np.mean(predicted_test_runtime)
            ),
            "train_median_pathfinder_runtime_seconds": train_median_runtime,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("Frozen held-out PRM Router evaluation complete")
    print(f"Test examples:              {len(test_base)}")
    print(f"ReasonEval:                 {accuracy['reasoneval_only']['estimate']:.4f}")
    print(f"PathFinder:                 {accuracy['pathfinder_only']['estimate']:.4f}")
    print(f"Random @ 20%:               {accuracy['random']['estimate']:.4f}")
    print(f"Low score @ 20%:            {accuracy['low_score']['estimate']:.4f}")
    print(f"Uncertainty @ 20%:          {accuracy['uncertainty']['estimate']:.4f}")
    print(
        f"Failure prediction @ 20%:   "
        f"{accuracy['failure_prediction']['estimate']:.4f}"
    )
    print(f"Benefit only @ 20%:         {accuracy['benefit_only']['estimate']:.4f}")
    print(f"Expected gain @ 20%:        {accuracy['expected_gain']['estimate']:.4f}")
    print(f"Cost-aware @ 20%:           {accuracy['cost_aware']['estimate']:.4f}")
    print(f"Oracle:                     {oracle_accuracy:.4f}")
    print(f"Output:                     {args.output}")


if __name__ == "__main__":
    main()
