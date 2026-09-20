"""Reusable feature construction and lightweight routing models."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


GAIN_CLASSES = [-1, 0, 1]

FEATURE_NAMES = [
    "disprm_score",
    "distance_to_threshold",
    "disprm_predicted_correct",
    "step_position",
    "current_step",
    "total_steps",
    "question_characters",
    "disprm_input_tokens",
    "current_step_characters",
]


def build_features(
    record: dict[str, Any],
    score: float,
    threshold: float,
) -> list[float]:
    steps = record.get("steps", [])
    question = str(record.get("question", ""))
    current_step_text = str(
        record.get("current_step_text", steps[-1] if steps else "")
    )
    prefix_text = "\n".join(str(step) for step in steps)
    current_step = int(record["current_step"])
    total_steps = int(record.get("total_steps", current_step))
    step_position = float(
        record.get("step_position", current_step / max(total_steps, 1))
    )
    predicted_correct = int(score >= threshold)

    return [
        score,
        abs(score - threshold),
        float(predicted_correct),
        step_position,
        float(current_step),
        float(total_steps),
        float(len(question)),
        float(record.get("disprm_input_tokens", 0)),
        float(len(current_step_text)),
    ]


def router_features(
    records: list[dict[str, Any]],
    scores: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return np.asarray(
        [
            build_features(record=row, score=float(score), threshold=threshold)
            for row, score in zip(records, scores)
        ],
        dtype=np.float64,
    )


def make_router(seed: int) -> Pipeline:
    """Create the balanced binary router used by feasibility baselines."""
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


def make_binary_predictor(seed: int) -> Pipeline:
    """Alias with a formal-evaluation name for the same binary model."""
    return make_router(seed)


def make_expected_gain_router(seed: int) -> Pipeline:
    """Create the fixed unweighted multinomial gain router."""
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


def gain_probabilities(
    model: Pipeline,
    features: np.ndarray,
) -> np.ndarray:
    """Return class probabilities in the fixed [-1, 0, +1] column order."""
    raw = model.predict_proba(features)
    classes = model.named_steps["classifier"].classes_
    output = np.zeros((len(features), 3), dtype=np.float64)
    class_to_column = {label: index for index, label in enumerate(GAIN_CLASSES)}
    for source_column, label in enumerate(classes):
        output[:, class_to_column[int(label)]] = raw[:, source_column]
    return output


def routing_accuracy(
    labels: np.ndarray,
    base_predictions: np.ndarray,
    second_predictions: np.ndarray,
    routed_indices: set[int],
) -> float:
    """Compatibility helper for feasibility scripts using a set of indices."""
    final = base_predictions.copy()
    if routed_indices:
        selected = np.asarray(sorted(routed_indices), dtype=np.int64)
        final[selected] = second_predictions[selected]
    return float(np.mean(final == labels))


def routed_accuracy(
    labels: np.ndarray,
    base_predictions: np.ndarray,
    second_predictions: np.ndarray,
    selected: np.ndarray,
) -> float:
    final = final_predictions(base_predictions, second_predictions, selected)
    return float(np.mean(final == labels))


def top_budget_indices(scores: np.ndarray, fraction: float) -> np.ndarray:
    count = int(round(len(scores) * fraction))
    order = np.argsort(-scores, kind="stable")
    return order[:count]


def safe_fraction(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def final_predictions(
    base_predictions: np.ndarray,
    second_predictions: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    result = base_predictions.copy()
    result[selected] = second_predictions[selected]
    return result
