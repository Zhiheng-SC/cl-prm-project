"""Train and evaluate a lightweight benefit-aware PRM router.

The router predicts whether invoking GenPRM will correct a mistake made by
the discriminative PRM. Performance is evaluated with out-of-fold predictions
to avoid evaluating each sample with a model trained on that same sample.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "benefit_router_oof.json"
)

FEATURE_NAMES = [
    "disprm_score",
    "distance_to_threshold",
    "disprm_predicted_correct",
    "step_position",
    "current_step",
    "total_steps",
    "question_characters",
    "prefix_characters",
    "current_step_characters",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disprm", type=Path, default=DEFAULT_DISPRM)
    parser.add_argument("--genprm", type=Path, default=DEFAULT_GENPRM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.96)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-trials", type=int, default=1000)
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


def record_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["example_id"]), int(record["current_step"])


def build_features(
    record: dict[str, Any],
    score: float,
    threshold: float,
) -> list[float]:
    steps = record.get("steps", [])
    question = str(record.get("question", ""))
    current_step_text = str(
        record.get(
            "current_step_text",
            steps[-1] if steps else "",
        )
    )
    prefix_text = "\n".join(str(step) for step in steps)

    current_step = int(record["current_step"])
    total_steps = int(record.get("total_steps", current_step))

    step_position = float(
        record.get(
            "step_position",
            current_step / max(total_steps, 1),
        )
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
        float(len(prefix_text)),
        float(len(current_step_text)),
    ]


def make_router(seed: int) -> Pipeline:
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


def routing_accuracy(
    labels: np.ndarray,
    dis_predictions: np.ndarray,
    gen_predictions: np.ndarray,
    routed_indices: set[int],
) -> float:
    final_predictions = dis_predictions.copy()

    if routed_indices:
        selected = np.array(sorted(routed_indices), dtype=int)
        final_predictions[selected] = gen_predictions[selected]

    return float(np.mean(final_predictions == labels))


def main() -> None:
    args = parse_args()

    dis_records = {
        record_key(record): record
        for record in read_records(args.disprm)
    }
    gen_records = {
        record_key(record): record
        for record in read_records(args.genprm)
    }

    common_keys = sorted(set(dis_records) & set(gen_records))

    if len(common_keys) != len(dis_records):
        raise ValueError("Some DisPRM records have no GenPRM match.")

    if len(common_keys) != len(gen_records):
        raise ValueError("Some GenPRM records have no DisPRM match.")

    features: list[list[float]] = []
    labels: list[int] = []
    dis_scores: list[float] = []
    dis_predictions: list[int] = []
    gen_predictions: list[int] = []
    benefit_labels: list[int] = []
    dis_runtimes: list[float] = []
    gen_runtimes: list[float] = []

    for item_key in common_keys:
        dis_record = dis_records[item_key]
        gen_record = gen_records[item_key]

        label = int(dis_record["label"])

        if label != int(gen_record["label"]):
            raise ValueError(f"Label mismatch for {item_key}")

        dis_score = float(dis_record["disprm_score"])
        dis_prediction = int(dis_score >= args.threshold)
        gen_prediction = int(gen_record["genprm_prediction"])

        dis_correct = dis_prediction == label
        gen_correct = gen_prediction == label

        # Positive only when replacing DisPRM with GenPRM improves the result.
        benefit = int((not dis_correct) and gen_correct)

        features.append(
            build_features(
                record=dis_record,
                score=dis_score,
                threshold=args.threshold,
            )
        )
        labels.append(label)
        dis_scores.append(dis_score)
        dis_predictions.append(dis_prediction)
        gen_predictions.append(gen_prediction)
        benefit_labels.append(benefit)
        dis_runtimes.append(
            float(dis_record["disprm_runtime_seconds"])
        )
        gen_runtimes.append(
            float(gen_record["genprm_runtime_seconds"])
        )

    x = np.asarray(features, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    dis_scores_array = np.asarray(dis_scores, dtype=np.float64)
    dis_predictions_array = np.asarray(
        dis_predictions,
        dtype=np.int64,
    )
    gen_predictions_array = np.asarray(
        gen_predictions,
        dtype=np.int64,
    )
    benefit_array = np.asarray(benefit_labels, dtype=np.int64)

    positive_count = int(benefit_array.sum())
    negative_count = len(benefit_array) - positive_count

    if positive_count < args.folds:
        raise ValueError(
            "Not enough positive benefit examples for the requested folds."
        )

    cross_validation = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.seed,
    )

    oof_probabilities = np.zeros(len(common_keys), dtype=np.float64)
    fold_ids = np.zeros(len(common_keys), dtype=np.int64)

    for fold_id, (train_indices, test_indices) in enumerate(
        cross_validation.split(x, benefit_array),
        start=1,
    ):
        router = make_router(args.seed + fold_id)
        router.fit(x[train_indices], benefit_array[train_indices])

        oof_probabilities[test_indices] = router.predict_proba(
            x[test_indices]
        )[:, 1]
        fold_ids[test_indices] = fold_id

    roc_auc = roc_auc_score(benefit_array, oof_probabilities)
    average_precision = average_precision_score(
        benefit_array,
        oof_probabilities,
    )

    total = len(common_keys)
    all_indices = list(range(total))

    beneficial_indices = {
        index
        for index in all_indices
        if benefit_array[index] == 1
    }

    harmful_indices = {
        index
        for index in all_indices
        if (
            dis_predictions_array[index] == labels_array[index]
            and gen_predictions_array[index] != labels_array[index]
        )
    }

    learned_order = sorted(
        all_indices,
        key=lambda index: oof_probabilities[index],
        reverse=True,
    )

    uncertainty_order = sorted(
        all_indices,
        key=lambda index: abs(
            dis_scores_array[index] - args.threshold
        ),
    )

    base_accuracy = routing_accuracy(
        labels_array,
        dis_predictions_array,
        gen_predictions_array,
        set(),
    )

    mean_dis_runtime = mean(dis_runtimes)
    mean_gen_runtime = mean(gen_runtimes)

    print(f"Examples:                    {total}")
    print(f"Benefit-positive examples:   {positive_count}")
    print(f"Benefit-negative examples:   {negative_count}")
    print(f"Positive rate:               {positive_count / total:.4f}")
    print(f"OOF ROC-AUC:                 {roc_auc:.4f}")
    print(f"OOF average precision:       {average_precision:.4f}")
    print(f"Random AP baseline:          {positive_count / total:.4f}")
    print()
    print(
        "Budget | Learned | Uncertainty | Random mean±std | "
        "Oracle | Benefit/harm | Cost"
    )
    print("-" * 94)

    rng = random.Random(args.seed)

    for budget_percent in [0, 10, 20, 30, 40, 50, 75, 100]:
        budget = round(total * budget_percent / 100)

        learned_routed = set(learned_order[:budget])
        uncertainty_routed = set(uncertainty_order[:budget])

        learned_accuracy = routing_accuracy(
            labels_array,
            dis_predictions_array,
            gen_predictions_array,
            learned_routed,
        )

        uncertainty_accuracy = routing_accuracy(
            labels_array,
            dis_predictions_array,
            gen_predictions_array,
            uncertainty_routed,
        )

        random_accuracies = []

        for _ in range(args.random_trials):
            random_routed = set(rng.sample(all_indices, budget))
            random_accuracies.append(
                routing_accuracy(
                    labels_array,
                    dis_predictions_array,
                    gen_predictions_array,
                    random_routed,
                )
            )

        learned_beneficial = len(
            learned_routed & beneficial_indices
        )
        learned_harmful = len(
            learned_routed & harmful_indices
        )

        oracle_gain = min(budget, len(beneficial_indices)) / total
        oracle_accuracy = base_accuracy + oracle_gain

        cascade_runtime = (
            mean_dis_runtime
            + (budget / total) * mean_gen_runtime
        )
        cost_ratio = cascade_runtime / mean_dis_runtime

        random_text = (
            f"{mean(random_accuracies):.3f}"
            f"±{pstdev(random_accuracies):.3f}"
        )

        benefit_harm_text = (
            f"{learned_beneficial:02d}/{learned_harmful:02d}"
        )

        print(
            f"{budget_percent:>5}% | "
            f"{learned_accuracy:>7.3f} | "
            f"{uncertainty_accuracy:>11.3f} | "
            f"{random_text:>15} | "
            f"{oracle_accuracy:>6.3f} | "
            f"{benefit_harm_text:>12} | "
            f"{cost_ratio:>4.2f}x"
        )

    # Fit once on all data only to inspect standardized coefficients.
    # This full-data fit is not used for the reported OOF performance.
    inspection_router = make_router(args.seed)
    inspection_router.fit(x, benefit_array)

    coefficients = inspection_router.named_steps[
        "classifier"
    ].coef_[0]

    print()
    print("Full-data standardized coefficients (inspection only):")

    for feature_name, coefficient in sorted(
        zip(FEATURE_NAMES, coefficients),
        key=lambda item: abs(item[1]),
        reverse=True,
    ):
        print(f"  {feature_name:28s} {coefficient:+.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    output_records = []

    for index, item_key in enumerate(common_keys):
        output_records.append(
            {
                "example_id": item_key[0],
                "current_step": item_key[1],
                "label": int(labels_array[index]),
                "disprm_score": float(dis_scores_array[index]),
                "disprm_prediction": int(
                    dis_predictions_array[index]
                ),
                "genprm_prediction": int(
                    gen_predictions_array[index]
                ),
                "benefit_label": int(benefit_array[index]),
                "router_oof_probability": float(
                    oof_probabilities[index]
                ),
                "fold": int(fold_ids[index]),
            }
        )

    args.output.write_text(
        json.dumps(
            output_records,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(f"OOF predictions saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()