"""Select the formal ReasonEval decision threshold on validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from train_benefit_router import read_records


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments" / "prm_router_formal.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "formal"
    / "validation"
    / "reasoneval_threshold.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validation-reasoneval", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    metric = str(config["base_threshold"]["selection_metric"])
    split = str(config["base_threshold"]["selection_split"])
    if split != "validation":
        raise ValueError("Formal threshold selection must use validation only.")
    if metric != "balanced_accuracy":
        raise ValueError(
            "This selector currently implements balanced_accuracy only; "
            f"config requested {metric!r}."
        )

    records = read_records(args.validation_reasoneval)
    if not records:
        raise ValueError("Validation ReasonEval output is empty.")

    formal_splits = {str(row.get("formal_split", "")) for row in records}
    if formal_splits - {"validation", ""}:
        raise ValueError(
            "Threshold selector received non-validation records: "
            f"{sorted(formal_splits)}"
        )

    labels = np.asarray([int(row["label"]) for row in records], dtype=np.int64)
    scores = np.asarray(
        [float(row["disprm_score"]) for row in records],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(scores)):
        raise ValueError("ReasonEval validation scores contain non-finite values.")

    rows = []
    for threshold in candidate_thresholds(scores):
        predictions = (scores >= threshold).astype(np.int64)
        balanced = float(balanced_accuracy_score(labels, predictions))
        accuracy = float(np.mean(predictions == labels))
        rows.append(
            {
                "threshold": float(threshold),
                "balanced_accuracy": balanced,
                "accuracy": accuracy,
            }
        )

    # Deterministic tie-break: highest balanced accuracy, then threshold closest
    # to 0.5, then the smaller threshold.
    rows.sort(
        key=lambda row: (
            -row["balanced_accuracy"],
            abs(row["threshold"] - 0.5),
            row["threshold"],
        )
    )
    selected = rows[0]

    output = {
        "schema_version": 1,
        "selection_split": "validation",
        "selection_metric": "balanced_accuracy",
        "examples": len(records),
        "selected_threshold": selected["threshold"],
        "selected_balanced_accuracy": selected["balanced_accuracy"],
        "selected_accuracy": selected["accuracy"],
        "tie_break": "closest_to_0.5_then_smaller_threshold",
        "candidate_count": len(rows),
        "top_candidates": rows[:20],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("ReasonEval threshold selection complete")
    print(f"Validation examples:        {len(records)}")
    print(f"Selected threshold:         {selected['threshold']:.8f}")
    print(f"Balanced accuracy:          {selected['balanced_accuracy']:.4f}")
    print(f"Plain accuracy:             {selected['accuracy']:.4f}")
    print(f"Output:                     {args.output}")


if __name__ == "__main__":
    main()
