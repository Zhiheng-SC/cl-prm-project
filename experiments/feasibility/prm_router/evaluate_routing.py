"""Evaluate budgeted routing baselines."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DISPRM = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "reasoneval_feasibility_100.json"
)

DEFAULT_GENPRM = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "genprm_feasibility_100.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disprm", type=Path, default=DEFAULT_DISPRM)
    parser.add_argument("--genprm", type=Path, default=DEFAULT_GENPRM)
    parser.add_argument("--threshold", type=float, default=0.96)
    parser.add_argument("--random-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        return [data]

    raise ValueError(f"Unsupported JSON structure in {path}")


def key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["example_id"]), int(row["current_step"])


def route_accuracy(
    dis_predictions: list[int],
    gen_predictions: list[int],
    labels: list[int],
    routed_indices: set[int],
) -> float:
    final_predictions = [
        gen_predictions[index]
        if index in routed_indices
        else dis_predictions[index]
        for index in range(len(labels))
    ]

    return mean(
        prediction == label
        for prediction, label in zip(final_predictions, labels)
    )


def main() -> None:
    args = parse_args()

    dis_rows = {key(row): row for row in read_records(args.disprm)}
    gen_rows = {key(row): row for row in read_records(args.genprm)}

    common_keys = sorted(set(dis_rows) & set(gen_rows))

    if len(common_keys) != len(dis_rows) or len(common_keys) != len(gen_rows):
        raise ValueError("DisPRM and GenPRM records do not match.")

    labels: list[int] = []
    dis_scores: list[float] = []
    dis_predictions: list[int] = []
    gen_predictions: list[int] = []
    dis_runtimes: list[float] = []
    gen_runtimes: list[float] = []

    for item_key in common_keys:
        dis_row = dis_rows[item_key]
        gen_row = gen_rows[item_key]

        label = int(dis_row["label"])

        if label != int(gen_row["label"]):
            raise ValueError(f"Label mismatch for {item_key}")

        score = float(dis_row["disprm_score"])

        labels.append(label)
        dis_scores.append(score)
        dis_predictions.append(int(score >= args.threshold))
        gen_predictions.append(int(gen_row["genprm_prediction"]))
        dis_runtimes.append(float(dis_row["disprm_runtime_seconds"]))
        gen_runtimes.append(float(gen_row["genprm_runtime_seconds"]))

    total = len(labels)
    indices = list(range(total))

    beneficial = {
        index
        for index in indices
        if dis_predictions[index] != labels[index]
        and gen_predictions[index] == labels[index]
    }

    harmful = {
        index
        for index in indices
        if dis_predictions[index] == labels[index]
        and gen_predictions[index] != labels[index]
    }

    neutral = set(indices) - beneficial - harmful

    base_accuracy = route_accuracy(
        dis_predictions,
        gen_predictions,
        labels,
        set(),
    )

    gen_accuracy = route_accuracy(
        dis_predictions,
        gen_predictions,
        labels,
        set(indices),
    )

    mean_dis_runtime = mean(dis_runtimes)
    mean_gen_runtime = mean(gen_runtimes)

    # Lowest score: route examples that DisPRM considers least correct.
    low_score_order = sorted(
        indices,
        key=lambda index: dis_scores[index],
    )

    # Margin uncertainty: route examples closest to the decision threshold.
    uncertainty_order = sorted(
        indices,
        key=lambda index: abs(
            dis_scores[index] - args.threshold
        ),
    )

    print(f"Examples:                  {total}")
    print(f"DisPRM threshold:          {args.threshold:.2f}")
    print(f"DisPRM-only accuracy:      {base_accuracy:.4f}")
    print(f"GenPRM-only accuracy:      {gen_accuracy:.4f}")
    print(f"Beneficial calls:          {len(beneficial)}")
    print(f"Harmful calls:             {len(harmful)}")
    print(f"Neutral calls:             {len(neutral)}")
    print()
    print(
        "Budget | Low score | Uncertainty | Random mean±std | "
        "Oracle | Cascade cost"
    )
    print("-" * 86)

    rng = random.Random(args.seed)

    for budget_percent in [0, 10, 20, 30, 40, 50, 75, 100]:
        budget = round(total * budget_percent / 100)

        low_score_routed = set(low_score_order[:budget])
        uncertainty_routed = set(uncertainty_order[:budget])

        low_score_accuracy = route_accuracy(
            dis_predictions,
            gen_predictions,
            labels,
            low_score_routed,
        )

        uncertainty_accuracy = route_accuracy(
            dis_predictions,
            gen_predictions,
            labels,
            uncertainty_routed,
        )

        random_accuracies = []

        for _ in range(args.random_trials):
            random_routed = set(rng.sample(indices, budget))
            random_accuracies.append(
                route_accuracy(
                    dis_predictions,
                    gen_predictions,
                    labels,
                    random_routed,
                )
            )

        # Oracle may call GenPRM on up to the available budget,
        # selecting beneficial calls and avoiding harmful calls.
        oracle_gain = min(budget, len(beneficial)) / total
        oracle_accuracy = base_accuracy + oracle_gain

        cascade_runtime = (
            mean_dis_runtime
            + (budget / total) * mean_gen_runtime
        )
        cascade_cost_ratio = cascade_runtime / mean_dis_runtime

        random_text = (
            f"{mean(random_accuracies):.3f}"
            f"±{pstdev(random_accuracies):.3f}"
        )

        print(
            f"{budget_percent:>5}% | "
            f"{low_score_accuracy:>9.3f} | "
            f"{uncertainty_accuracy:>11.3f} | "
            f"{random_text:>15} | "
            f"{oracle_accuracy:>6.3f} | "
            f"{cascade_cost_ratio:>8.2f}x"
        )


if __name__ == "__main__":
    main()