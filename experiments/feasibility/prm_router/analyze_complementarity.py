"""Analyze complementarity between ReasonEval and GenPRM."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DISPRM = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "reasoneval_smoke_test.jsonl"
)

DEFAULT_GENPRM = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "genprm_smoke_test.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--disprm", type=Path, default=DEFAULT_DISPRM)
    parser.add_argument("--genprm", type=Path, default=DEFAULT_GENPRM)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read either a JSON array or JSONL records."""

    text = path.read_text(encoding="utf-8-sig").strip()

    if not text:
        return []

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

    raise ValueError(
        f"Unsupported JSON structure in {path}: "
        f"{type(data).__name__}"
    )


def make_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["example_id"]), int(row["current_step"])


def index_rows(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}

    for row in rows:
        key = make_key(row)

        if key in indexed:
            raise ValueError(f"Duplicate record: {key}")

        indexed[key] = row

    return indexed


def print_confusion_matrix(
    name: str,
    labels: list[int],
    predictions: list[int],
) -> None:
    tp = sum(
        label == 1 and prediction == 1
        for label, prediction in zip(labels, predictions)
    )
    fn = sum(
        label == 1 and prediction == 0
        for label, prediction in zip(labels, predictions)
    )
    fp = sum(
        label == 0 and prediction == 1
        for label, prediction in zip(labels, predictions)
    )
    tn = sum(
        label == 0 and prediction == 0
        for label, prediction in zip(labels, predictions)
    )

    accuracy = (tp + tn) / len(labels)

    print(f"{name} confusion matrix")
    print("                 Pred correct  Pred error")
    print(f"Actually correct       {tp:2d}          {fn:2d}")
    print(f"Actually error         {fp:2d}          {tn:2d}")
    print(f"Accuracy: {accuracy:.4f}")
    print()


def analyze_threshold(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    disprm_threshold: float,
) -> dict[str, list[tuple[str, int]]]:
    groups: dict[str, list[tuple[str, int]]] = {
        "both_correct": [],
        "disprm_only": [],
        "genprm_only": [],
        "both_wrong": [],
    }

    for dis_row, gen_row in pairs:
        label = int(dis_row["label"])

        dis_prediction = int(
            float(dis_row["disprm_score"]) >= disprm_threshold
        )
        gen_prediction = int(gen_row["genprm_prediction"])

        dis_correct = dis_prediction == label
        gen_correct = gen_prediction == label

        key = make_key(dis_row)

        if dis_correct and gen_correct:
            groups["both_correct"].append(key)
        elif dis_correct and not gen_correct:
            groups["disprm_only"].append(key)
        elif not dis_correct and gen_correct:
            groups["genprm_only"].append(key)
        else:
            groups["both_wrong"].append(key)

    total = len(pairs)
    dis_accuracy = (
        len(groups["both_correct"]) + len(groups["disprm_only"])
    ) / total
    gen_accuracy = (
        len(groups["both_correct"]) + len(groups["genprm_only"])
    ) / total
    oracle_accuracy = 1 - len(groups["both_wrong"]) / total

    print("=" * 72)
    print(f"Complementarity with DisPRM threshold = {disprm_threshold:.2f}")
    print("=" * 72)
    print(f"Both correct:                  {len(groups['both_correct']):2d}")
    print(
        "DisPRM correct, GenPRM wrong: "
        f"{len(groups['disprm_only']):2d}  "
        "(potentially harmful calls)"
    )
    print(
        "DisPRM wrong, GenPRM correct: "
        f"{len(groups['genprm_only']):2d}  "
        "(potentially beneficial calls)"
    )
    print(f"Both wrong:                    {len(groups['both_wrong']):2d}")
    print()
    print(f"DisPRM accuracy:               {dis_accuracy:.4f}")
    print(f"GenPRM accuracy:               {gen_accuracy:.4f}")
    print(f"Oracle routing accuracy:       {oracle_accuracy:.4f}")
    print(
        "Maximum oracle gain over DisPRM: "
        f"+{len(groups['genprm_only'])}/{total}"
    )
    print()

    print("Potentially beneficial calls:")
    if groups["genprm_only"]:
        for example_id, current_step in groups["genprm_only"]:
            print(f"  {example_id}, step={current_step}")
    else:
        print("  None")

    print()
    print("Potentially harmful calls:")
    if groups["disprm_only"]:
        for example_id, current_step in groups["disprm_only"]:
            print(f"  {example_id}, step={current_step}")
    else:
        print("  None")

    print()

    return groups


def main() -> None:
    args = parse_args()

    dis_rows = read_jsonl(args.disprm)
    gen_rows = read_jsonl(args.genprm)

    dis_index = index_rows(dis_rows)
    gen_index = index_rows(gen_rows)

    common_keys = sorted(set(dis_index) & set(gen_index))

    missing_from_genprm = sorted(set(dis_index) - set(gen_index))
    missing_from_disprm = sorted(set(gen_index) - set(dis_index))

    if missing_from_genprm:
        raise ValueError(
            f"{len(missing_from_genprm)} records are missing from GenPRM."
        )

    if missing_from_disprm:
        raise ValueError(
            f"{len(missing_from_disprm)} records are missing from DisPRM."
        )

    pairs = [
        (dis_index[key], gen_index[key])
        for key in common_keys
    ]

    for dis_row, gen_row in pairs:
        if int(dis_row["label"]) != int(gen_row["label"]):
            raise ValueError(
                f"Label mismatch for record {make_key(dis_row)}"
            )

    labels = [
        int(dis_row["label"])
        for dis_row, _ in pairs
    ]

    dis_default_predictions = [
        int(float(dis_row["disprm_score"]) >= 0.5)
        for dis_row, _ in pairs
    ]

    gen_predictions = [
        int(gen_row["genprm_prediction"])
        for _, gen_row in pairs
    ]

    parsed_judgements = Counter(
        gen_row["genprm_parsed_judgement"]
        for _, gen_row in pairs
    )

    judgement_mapping = {"Yes": 1, "No": 0}

    judgement_prediction_mismatches = sum(
        judgement_mapping.get(
            gen_row["genprm_parsed_judgement"]
        )
        != int(gen_row["genprm_prediction"])
        for _, gen_row in pairs
        if gen_row["genprm_parsed_judgement"] in judgement_mapping
    )

    judgement_prediction_mismatches = sum(
        judgement_mapping.get(
            gen_row["genprm_parsed_judgement"]
        )
        != int(gen_row["genprm_prediction"])
        for _, gen_row in pairs
        if gen_row["genprm_parsed_judgement"] in judgement_mapping
    )

    gen_scores = [
        float(gen_row["genprm_score"])
        for _, gen_row in pairs
    ]

    dis_runtimes = [
        float(dis_row["disprm_runtime_seconds"])
        for dis_row, _ in pairs
    ]

    gen_runtimes = [
        float(gen_row["genprm_runtime_seconds"])
        for _, gen_row in pairs
    ]

    print(f"Paired examples: {len(pairs)}")
    print()

    print("GenPRM sampled judgement parsing:")
    for judgement, count in sorted(parsed_judgements.items()):
        print(f"  {judgement}: {count}")
    print(
        "  Sampled/probability mismatches: "
        f"{judgement_prediction_mismatches}"
    )
    print()

    print("GenPRM scores:")
    print(f"  Minimum:       {min(gen_scores):.6f}")
    print(f"  Maximum:       {max(gen_scores):.6f}")
    print(f"  Mean:          {mean(gen_scores):.6f}")
    print(f"  Unique values: {len(set(gen_scores))}")
    print()

    print("Runtime:")
    print(f"  Mean DisPRM:   {mean(dis_runtimes):.4f}s")
    print(f"  Mean GenPRM:   {mean(gen_runtimes):.4f}s")
    print(
        f"  Slowdown:      "
        f"{mean(gen_runtimes) / mean(dis_runtimes):.2f}x"
    )
    print()

    print_confusion_matrix(
        "ReasonEval, threshold 0.50",
        labels,
        dis_default_predictions,
    )

    print_confusion_matrix(
        "GenPRM",
        labels,
        gen_predictions,
    )

    # Default threshold used by the original ReasonEval run.
    analyze_threshold(pairs, disprm_threshold=0.50)

    # Exploratory threshold selected during the separate 20-example pilot.
    # It remains fixed when analyzing the 100-example feasibility subset.
    analyze_threshold(pairs, disprm_threshold=0.96)


if __name__ == "__main__":
    main()