"""Grouped bootstrap utilities for formal evaluation."""

from __future__ import annotations

import numpy as np


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
