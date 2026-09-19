"""Validation for the frozen formal PRM Router configuration."""

from __future__ import annotations

from typing import Any

from cl_prm.evaluation.cost import COST_FEATURE_NAMES
from cl_prm.evaluation.routing import FEATURE_NAMES


def _require_equal(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ValueError(
            f"Unsupported formal config for {field}: "
            f"expected {expected!r}, found {actual!r}."
        )


def validate_formal_config(config: dict[str, Any]) -> None:
    """Reject config values that the current formal implementation cannot honor.

    The JSON config is a frozen protocol contract, not a free-form tuning file.
    Several implementation choices are intentionally fixed in code; validating
    them here prevents a silent mismatch between the declared protocol and the
    executed experiment.
    """
    dataset = config["dataset"]
    splits = config["formal_splits"]
    runtime = config["runtime"]
    routing = config["routing"]
    cost = routing["cost_aware"]
    threshold = config["base_threshold"]
    evaluation = config["evaluation"]
    test_lock = evaluation["test_lock"]

    _require_equal(dataset["group_field"], "original_question", "dataset.group_field")
    _require_equal(
        bool(dataset["exclude_pilot_groups"]),
        True,
        "dataset.exclude_pilot_groups",
    )
    _require_equal(float(splits["correct_fraction"]), 0.5, "formal_splits.correct_fraction")
    _require_equal(
        bool(splits["preserve_error_type_coverage"]),
        True,
        "formal_splits.preserve_error_type_coverage",
    )

    _require_equal(
        bool(runtime["exclude_model_loading"]),
        True,
        "runtime.exclude_model_loading",
    )
    _require_equal(
        bool(runtime["cuda_synchronize"]),
        True,
        "runtime.cuda_synchronize",
    )
    _require_equal(
        bool(runtime["same_hardware_required"]),
        True,
        "runtime.same_hardware_required",
    )

    _require_equal(
        threshold["selection_split"],
        "validation",
        "base_threshold.selection_split",
    )
    _require_equal(
        threshold["selection_metric"],
        "balanced_accuracy",
        "base_threshold.selection_metric",
    )

    _require_equal(routing["model"], "logistic_regression", "routing.model")
    _require_equal(
        list(routing["pre_call_features"]),
        FEATURE_NAMES,
        "routing.pre_call_features",
    )
    _require_equal(
        set(routing["targets"]),
        {"base_failure", "beneficial", "beneficial_neutral_harmful"},
        "routing.targets",
    )

    primary_budget = float(routing["primary_call_budget"])
    if not 0.0 <= primary_budget <= 1.0:
        raise ValueError("routing.primary_call_budget must be in [0, 1].")

    secondary_budgets = [float(value) for value in routing["secondary_call_budgets"]]
    if any(not 0.0 <= value <= 1.0 for value in secondary_budgets):
        raise ValueError("Every routing.secondary_call_budgets value must be in [0, 1].")
    if len(set(secondary_budgets)) != len(secondary_budgets):
        raise ValueError("routing.secondary_call_budgets must not contain duplicates.")

    _require_equal(bool(cost["enabled"]), True, "routing.cost_aware.enabled")
    _require_equal(
        cost["cost_target"],
        "pathfinder_runtime_seconds",
        "routing.cost_aware.cost_target",
    )
    _require_equal(
        cost["predictor_model"],
        "ridge_regression",
        "routing.cost_aware.predictor_model",
    )
    _require_equal(
        list(cost["predictor_features"]),
        COST_FEATURE_NAMES,
        "routing.cost_aware.predictor_features",
    )
    _require_equal(
        cost["normalization"],
        "divide_by_train_median_pathfinder_runtime",
        "routing.cost_aware.normalization",
    )
    _require_equal(
        cost["selection_split"],
        "validation",
        "routing.cost_aware.selection_split",
    )
    if float(cost["ridge_alpha"]) < 0.0:
        raise ValueError("routing.cost_aware.ridge_alpha must be non-negative.")
    if not cost["lambda_h_candidates"] or not cost["mu_candidates"]:
        raise ValueError("Cost-aware lambda_h and mu candidate grids must be non-empty.")

    _require_equal(
        evaluation["primary_metric"],
        "accuracy_at_20_percent_call_budget",
        "evaluation.primary_metric",
    )
    _require_equal(
        evaluation["bootstrap_unit"],
        "original_question",
        "evaluation.bootstrap_unit",
    )
    _require_equal(
        bool(evaluation["same_hardware_runtime_required"]),
        True,
        "evaluation.same_hardware_runtime_required",
    )
    _require_equal(
        bool(test_lock["require_protocol_frozen"]),
        True,
        "evaluation.test_lock.require_protocol_frozen",
    )
    _require_equal(
        bool(test_lock["primary_comparison_locked_before_test"]),
        True,
        "evaluation.test_lock.primary_comparison_locked_before_test",
    )
    _require_equal(
        bool(test_lock["no_test_based_hyperparameter_selection"]),
        True,
        "evaluation.test_lock.no_test_based_hyperparameter_selection",
    )
