"""Pre-call runtime prediction and cascade-cost helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


COST_FEATURE_NAMES = [
    "disprm_input_tokens",
    "current_step",
    "total_steps",
    "step_position",
    "question_characters",
    "prefix_characters",
    "current_step_characters",
]


def cost_features(records: list[dict[str, Any]]) -> np.ndarray:
    features = []
    for row in records:
        steps = row.get("steps", [])
        question = str(row.get("question", ""))
        current_step_text = str(
            row.get("current_step_text", steps[-1] if steps else "")
        )
        prefix_text = "\n".join(str(step) for step in steps)
        current_step = int(row["current_step"])
        total_steps = int(row.get("total_steps", current_step))
        step_position = float(
            row.get("step_position", current_step / max(total_steps, 1))
        )
        features.append(
            [
                float(row.get("disprm_input_tokens", 0)),
                float(current_step),
                float(total_steps),
                step_position,
                float(len(question)),
                float(len(prefix_text)),
                float(len(current_step_text)),
            ]
        )
    return np.asarray(features, dtype=np.float64)


def make_cost_predictor(alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )


def selected_runtime(
    base_runtime: np.ndarray,
    second_runtime: np.ndarray,
    selected: np.ndarray,
) -> float:
    """Sequential cascade cost: all base calls plus selected second-stage calls."""
    return float(np.sum(base_runtime) + np.sum(second_runtime[selected]))
