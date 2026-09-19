"""Threshold-selection helpers."""

from __future__ import annotations

import numpy as np


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores)
    if len(unique) == 1:
        return np.asarray([0.0, float(unique[0]), 1.0], dtype=np.float64)

    midpoints = (unique[:-1] + unique[1:]) / 2.0
    values = np.concatenate(
        [
            np.asarray([0.0], dtype=np.float64),
            unique,
            midpoints,
            np.asarray([1.0], dtype=np.float64),
        ]
    )
    return np.unique(np.clip(values, 0.0, 1.0))
