"""Offline post-hoc threshold sensitivity for RE -> second-stage PRM routing.

This analysis never runs model inference. It thresholds saved raw verifier
scores, selects each second-stage threshold on validation only, refits the
existing expected-gain router on train, selects lambda/mu on validation, and
evaluates the two requested configurations on test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, r2_score

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cl_prm.data.records import align_verifier_records  # noqa: E402
from cl_prm.evaluation.bootstrap import bootstrap_group_metric  # noqa: E402
from cl_prm.evaluation.cost import cost_features, make_cost_predictor  # noqa: E402
from cl_prm.evaluation.routing import (  # noqa: E402
    GAIN_CLASSES,
    final_predictions,
    gain_probabilities,
    make_expected_gain_router,
    router_features,
    safe_fraction,
    top_budget_indices,
)
from cl_prm.evaluation.thresholds import candidate_thresholds  # noqa: E402
from cl_prm.utils.formal_config import validate_formal_config  # noqa: E402


EXPECTED_SPLIT_COUNTS = {"train": 600, "validation": 200, "test": 400}
ORIGINAL_THRESHOLD = 0.5
TIE_BREAK = "maximize_balanced_accuracy_then_closest_to_0.5_then_smaller_threshold"
SCOPE = "post_hoc_threshold_sensitivity_after_test_results_seen"

VERIFIERS: dict[str, dict[str, str]] = {
    "pathfinder": {
        "display_name": "PathFinder",
        "score_field": "pathfinder_official_score",
        "prediction_field": "pathfinder_prediction",
        "runtime_field": "pathfinder_runtime_seconds",
    },
    "math_prm": {
        "display_name": "MathPRM",
        "score_field": "extension_score",
        "prediction_field": "extension_prediction",
        "runtime_field": "extension_runtime_seconds",
    },
    "skywork_prm": {
        "display_name": "Skywork",
        "score_field": "extension_score",
        "prediction_field": "extension_prediction",
        "runtime_field": "extension_runtime_seconds",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT.parent / "cl_finalproject_hg",
        help="Local checkout of cl-prm-team/cl-prm-artifacts.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Must resolve to <artifacts-dir>/extensions/threshold_sensitivity.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def validate_output_root(artifact_root: Path, output_root: Path) -> None:
    expected = (artifact_root / "extensions" / "threshold_sensitivity").resolve()
    if output_root.resolve() != expected:
        raise ValueError(
            "Threshold-sensitivity outputs must be written only to "
            f"{expected}; received {output_root.resolve()}."
        )
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing threshold-sensitivity outputs: {output_root}"
        )


def second_stage_path(artifact_root: Path, verifier: str, split: str) -> Path:
    if verifier == "pathfinder":
        return artifact_root / "formal" / split / "pathfinder.jsonl"
    return artifact_root / "extensions" / "full" / split / f"{verifier}.jsonl"


def validate_metadata(path: Path, split: str, expected_count: int) -> dict[str, Any]:
    metadata_path = path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError(f"Inference artifact is not complete: {metadata_path}")
    source = metadata.get("input", {})
    if int(source.get("selected_examples", -1)) != expected_count:
        raise ValueError(f"Wrong {split} count in {metadata_path}")
    if Path(str(source.get("path", ""))).name != f"{split}.jsonl":
        raise ValueError(f"Wrong split source in {metadata_path}")
    expected_sha256 = metadata.get("output", {}).get("sha256")
    checkout_sha256 = sha256(path)
    verification = "exact_bytes"
    if checkout_sha256 != expected_sha256:
        # Git for Windows may materialize tracked JSONL text with CRLF even
        # though the inference sidecar hashes the original Linux LF bytes.
        # Accept only that exact, reversible normalization; any other byte
        # difference remains a hard failure.
        normalized_sha256 = sha256_bytes(path.read_bytes().replace(b"\r\n", b"\n"))
        if normalized_sha256 != expected_sha256:
            raise ValueError(f"Output SHA-256 differs from sidecar: {path}")
        verification = "normalized_crlf_to_lf"
    metadata["_local_hash_verification"] = {
        "checkout_sha256": checkout_sha256,
        "sidecar_sha256": expected_sha256,
        "method": verification,
    }
    return metadata


def load_split(
    artifact_root: Path, verifier: str, split: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    expected_count = EXPECTED_SPLIT_COUNTS[split]
    re_path = artifact_root / "formal" / split / "reasoneval.jsonl"
    verifier_path = second_stage_path(artifact_root, verifier, split)
    re_metadata = validate_metadata(re_path, split, expected_count)
    verifier_metadata = validate_metadata(verifier_path, split, expected_count)
    if re_metadata["input"]["sha256"] != verifier_metadata["input"]["sha256"]:
        raise ValueError(f"RE and {verifier} {split} input hashes differ")

    base, second = align_verifier_records(re_path, verifier_path)
    if len(base) != expected_count:
        raise ValueError(f"Expected {expected_count} aligned {split} rows, got {len(base)}")
    if {str(row.get("formal_split")) for row in base + second} != {split}:
        raise ValueError(f"Unexpected formal_split value in {split}")

    spec = VERIFIERS[verifier]
    labels = np.asarray([int(row["label"]) for row in base], dtype=np.int64)
    second_labels = np.asarray([int(row["label"]) for row in second], dtype=np.int64)
    if not np.array_equal(labels, second_labels):
        raise ValueError(f"RE and {verifier} labels differ in {split}")
    scores = np.asarray([float(row[spec["score_field"]]) for row in second])
    runtimes = np.asarray([float(row[spec["runtime_field"]]) for row in second])
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"Non-finite {verifier} score in {split}")
    if not np.all(np.isfinite(runtimes)) or np.any(runtimes <= 0):
        raise ValueError(f"Invalid {verifier} runtime in {split}")
    if verifier != "pathfinder" and (np.any(scores < 0) or np.any(scores > 1)):
        raise ValueError(f"Out-of-range {verifier} score in {split}")
    stored = np.asarray([int(row[spec["prediction_field"]]) for row in second])
    if not np.array_equal(stored, (scores >= ORIGINAL_THRESHOLD).astype(np.int64)):
        raise ValueError(f"Stored {verifier} predictions do not reproduce threshold 0.5")
    if verifier == "pathfinder":
        failed_gate = np.asarray([not bool(row["pathfinder_gate_passed"]) for row in second])
        if not np.all(scores[failed_gate] == -1.0):
            raise ValueError("PathFinder failed gates must retain saved final score -1")
    else:
        if any(row.get("extension_verifier") != verifier for row in second):
            raise ValueError(f"Wrong extension_verifier field for {verifier}")
        if any(row.get("extension_status") != "ok" for row in second):
            raise ValueError(f"Non-ok {verifier} inference status in {split}")

    provenance = {
        "split": split,
        "input_sha256": re_metadata["input"]["sha256"],
        "reasoneval": {
            "path": re_path.relative_to(artifact_root).as_posix(),
            **re_metadata["_local_hash_verification"],
        },
        "second_stage": {
            "path": verifier_path.relative_to(artifact_root).as_posix(),
            **verifier_metadata["_local_hash_verification"],
        },
    }
    return base, second, provenance


def threshold_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for threshold in candidate_thresholds(scores):
        predictions = (scores >= threshold).astype(np.int64)
        rows.append(
            {
                "threshold": float(threshold),
                "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
                "accuracy": float(np.mean(predictions == labels)),
                "positive_predictions": int(np.sum(predictions)),
            }
        )
    selected = sorted(
        rows,
        key=lambda row: (
            -row["balanced_accuracy"],
            abs(row["threshold"] - ORIGINAL_THRESHOLD),
            row["threshold"],
        ),
    )[0]
    return sorted(rows, key=lambda row: row["threshold"]), selected


def metric_row(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(np.int64)
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(predictions == labels)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "positive_predictions": int(np.sum(predictions)),
    }


def arrays_for_threshold(
    base: list[dict[str, Any]],
    second: list[dict[str, Any]],
    verifier: str,
    re_threshold: float,
    second_threshold: float,
) -> dict[str, np.ndarray]:
    spec = VERIFIERS[verifier]
    labels = np.asarray([int(row["label"]) for row in base], dtype=np.int64)
    re_scores = np.asarray([float(row["disprm_score"]) for row in base])
    re_predictions = (re_scores >= re_threshold).astype(np.int64)
    second_scores = np.asarray([float(row[spec["score_field"]]) for row in second])
    second_predictions = (second_scores >= second_threshold).astype(np.int64)
    gain = (
        (second_predictions == labels).astype(np.int64)
        - (re_predictions == labels).astype(np.int64)
    )
    runtime = np.asarray([float(row[spec["runtime_field"]]) for row in second])
    return {
        "labels": labels,
        "re_scores": re_scores,
        "re_predictions": re_predictions,
        "second_scores": second_scores,
        "second_predictions": second_predictions,
        "gain": gain,
        "runtime": runtime,
    }


def gain_counts(gain: np.ndarray) -> dict[str, int]:
    return {
        "beneficial": int(np.sum(gain == 1)),
        "neutral": int(np.sum(gain == 0)),
        "harmful": int(np.sum(gain == -1)),
    }


def split_metrics(
    arrays: dict[str, np.ndarray], selected: np.ndarray, base_runtime: np.ndarray
) -> dict[str, Any]:
    routed = final_predictions(
        arrays["re_predictions"], arrays["second_predictions"], selected
    )
    re_correct = arrays["re_predictions"] == arrays["labels"]
    second_correct = arrays["second_predictions"] == arrays["labels"]
    selected_gain = arrays["gain"][selected]
    oracle_accuracy = float(np.mean(re_correct | second_correct))
    re_accuracy = float(np.mean(re_correct))
    routed_accuracy = float(np.mean(routed == arrays["labels"]))
    return {
        "reasoneval_accuracy": re_accuracy,
        "second_stage_accuracy": float(np.mean(second_correct)),
        "second_stage_balanced_accuracy": float(
            balanced_accuracy_score(arrays["labels"], arrays["second_predictions"])
        ),
        "routed_accuracy": routed_accuracy,
        "pairwise_oracle_accuracy": oracle_accuracy,
        "captured_oracle_headroom": safe_fraction(
            routed_accuracy - re_accuracy, oracle_accuracy - re_accuracy
        ),
        "available_outcomes": gain_counts(arrays["gain"]),
        "selected_calls": int(len(selected)),
        "selected_outcomes": gain_counts(selected_gain),
        "second_stage_total_runtime_seconds": float(np.sum(arrays["runtime"])),
        "selected_extra_runtime_seconds": float(np.sum(arrays["runtime"][selected])),
        "selected_cascade_runtime_seconds": float(
            np.sum(base_runtime) + np.sum(arrays["runtime"][selected])
        ),
    }


def evaluate_router(
    verifier: str,
    records: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    re_threshold: float,
    second_threshold: float,
    formal: dict[str, Any],
    bootstrap_offset: int,
) -> dict[str, Any]:
    split_arrays = {
        split: arrays_for_threshold(
            base, second, verifier, re_threshold, second_threshold
        )
        for split, (base, second) in records.items()
    }
    train = split_arrays["train"]
    validation = split_arrays["validation"]
    test = split_arrays["test"]
    missing = set(GAIN_CLASSES) - set(np.unique(train["gain"]).tolist())
    if missing:
        raise ValueError(
            f"{verifier} threshold {second_threshold} train split lacks gain classes: {sorted(missing)}"
        )

    seed = int(formal["formal_splits"]["seed"])
    budget = float(formal["routing"]["primary_call_budget"])
    cost_settings = formal["routing"]["cost_aware"]
    train_base = records["train"][0]
    validation_base = records["validation"][0]
    test_base = records["test"][0]

    router = make_expected_gain_router(seed=seed)
    router.fit(
        router_features(train_base, train["re_scores"], re_threshold), train["gain"]
    )
    validation_probabilities = gain_probabilities(
        router,
        router_features(validation_base, validation["re_scores"], re_threshold),
    )
    test_probabilities = gain_probabilities(
        router, router_features(test_base, test["re_scores"], re_threshold)
    )

    cost_model = make_cost_predictor(alpha=float(cost_settings["ridge_alpha"]))
    cost_model.fit(cost_features(train_base), train["runtime"])
    predicted_validation_runtime = np.maximum(
        cost_model.predict(cost_features(validation_base)), 0.0
    )
    predicted_test_runtime = np.maximum(cost_model.predict(cost_features(test_base)), 0.0)
    median_runtime = float(np.median(train["runtime"]))
    if median_runtime <= 0:
        raise ValueError("Training median second-stage runtime must be positive")

    candidates = []
    for lambda_h in cost_settings["lambda_h_candidates"]:
        for mu in cost_settings["mu_candidates"]:
            utility = (
                validation_probabilities[:, 2]
                - float(lambda_h) * validation_probabilities[:, 0]
                - float(mu) * predicted_validation_runtime / median_runtime
            )
            selected = top_budget_indices(utility, budget)
            routed = final_predictions(
                validation["re_predictions"], validation["second_predictions"], selected
            )
            candidates.append(
                {
                    "lambda_h": float(lambda_h),
                    "mu": float(mu),
                    "validation_accuracy": float(np.mean(routed == validation["labels"])),
                    "measured_extra_runtime_seconds": float(
                        np.sum(validation["runtime"][selected])
                    ),
                    "called_examples": int(len(selected)),
                }
            )
    candidates.sort(
        key=lambda row: (
            -row["validation_accuracy"],
            row["measured_extra_runtime_seconds"],
            row["lambda_h"],
            row["mu"],
        )
    )
    selected_parameters = candidates[0]
    lambda_h = selected_parameters["lambda_h"]
    mu = selected_parameters["mu"]
    validation_utility = (
        validation_probabilities[:, 2]
        - lambda_h * validation_probabilities[:, 0]
        - mu * predicted_validation_runtime / median_runtime
    )
    test_utility = (
        test_probabilities[:, 2]
        - lambda_h * test_probabilities[:, 0]
        - mu * predicted_test_runtime / median_runtime
    )
    validation_selected = top_budget_indices(validation_utility, budget)
    test_selected = top_budget_indices(test_utility, budget)
    if len(validation_selected) != 40 or len(test_selected) != 80:
        raise ValueError("Expected exactly 40 validation and 80 test routed calls")

    validation_base_runtime = np.asarray(
        [float(row["disprm_runtime_seconds"]) for row in validation_base]
    )
    test_base_runtime = np.asarray(
        [float(row["disprm_runtime_seconds"]) for row in test_base]
    )
    validation_metrics = split_metrics(
        validation, validation_selected, validation_base_runtime
    )
    test_metrics = split_metrics(test, test_selected, test_base_runtime)
    test_routed = final_predictions(
        test["re_predictions"], test["second_predictions"], test_selected
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
    paired = (test_routed == test["labels"]).astype(float) - (
        test["re_predictions"] == test["labels"]
    ).astype(float)
    test_metrics["paired_difference_vs_re"] = bootstrap_group_metric(
        paired,
        group_ids,
        int(formal["evaluation"]["bootstrap_samples"]),
        float(formal["evaluation"]["confidence_level"]),
        seed + bootstrap_offset,
    )
    return {
        "second_stage_threshold": float(second_threshold),
        "train_gain_counts": gain_counts(train["gain"]),
        "router_selection": {
            "rule": cost_settings["selection_rule"],
            "selected": selected_parameters,
            "candidates": candidates,
        },
        "cost_predictor": {
            "model": "ridge_regression",
            "ridge_alpha": float(cost_settings["ridge_alpha"]),
            "train_median_runtime_seconds": median_runtime,
            "validation_mae_seconds": float(
                np.mean(np.abs(predicted_validation_runtime - validation["runtime"]))
            ),
            "validation_r2": float(
                r2_score(validation["runtime"], predicted_validation_runtime)
            ),
            "median_baseline_mae_seconds": float(
                np.mean(np.abs(median_runtime - validation["runtime"]))
            ),
        },
        "validation": validation_metrics,
        "test": test_metrics,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def original_reproduction_checks(
    artifact_root: Path,
    verifier: str,
    standalone_rows: list[dict[str, Any]],
    router_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Require the 0.5 condition to reproduce the already archived analysis."""
    original = next(row for row in standalone_rows if row["condition"] == "original_0.5")
    if verifier == "pathfinder":
        development = json.loads(
            (artifact_root / "evaluation/router/pathfinder_development_selection.json").read_text(
                encoding="utf-8"
            )
        )
        test = json.loads(
            (artifact_root / "evaluation/router/formal_test_results.json").read_text(
                encoding="utf-8"
            )
        )
        expected = {
            "standalone_validation_accuracy": development["validation"]["pathfinder_accuracy"],
            "standalone_test_accuracy": test["accuracy"]["pathfinder_only"]["estimate"],
            "validation_routed_accuracy": development["cost_aware_selection"]["selected"][
                "validation_accuracy"
            ],
            "test_routed_accuracy": test["accuracy"]["cost_aware"]["estimate"],
            "lambda_h": development["cost_aware_selection"]["selected"]["lambda_h"],
            "mu": development["cost_aware_selection"]["selected"]["mu"],
        }
        source_paths = [
            "evaluation/router/pathfinder_development_selection.json",
            "evaluation/router/formal_test_results.json",
        ]
    else:
        evaluation_root = artifact_root / "extensions" / "full" / "evaluation"
        development = json.loads(
            (evaluation_root / f"{verifier}_development_selection.json").read_text(
                encoding="utf-8"
            )
        )
        test = json.loads(
            (evaluation_root / f"{verifier}_test_results.json").read_text(encoding="utf-8")
        )
        expected = {
            "standalone_validation_accuracy": development["validation"][
                "second_stage_accuracy"
            ],
            "standalone_test_accuracy": test["accuracy"]["second_stage_only"],
            "validation_routed_accuracy": development["selected"]["validation_accuracy"],
            "test_routed_accuracy": test["accuracy"]["selected_router"],
            "lambda_h": development["selected"]["lambda_h"],
            "mu": development["selected"]["mu"],
        }
        source_paths = [
            f"extensions/full/evaluation/{verifier}_development_selection.json",
            f"extensions/full/evaluation/{verifier}_test_results.json",
        ]

    selected_parameters = router_result["router_selection"]["selected"]
    actual = {
        "standalone_validation_accuracy": original["validation_accuracy"],
        "standalone_test_accuracy": original["test_accuracy"],
        "validation_routed_accuracy": router_result["validation"]["routed_accuracy"],
        "test_routed_accuracy": router_result["test"]["routed_accuracy"],
        "lambda_h": selected_parameters["lambda_h"],
        "mu": selected_parameters["mu"],
    }
    checks = []
    for name, expected_value in expected.items():
        actual_value = actual[name]
        passed = bool(np.isclose(actual_value, expected_value, rtol=0, atol=1e-12))
        checks.append(
            {
                "verifier": verifier,
                "check": name,
                "expected": float(expected_value),
                "actual": float(actual_value),
                "passed": passed,
                "archived_sources": source_paths,
            }
        )
    if not all(check["passed"] for check in checks):
        failed = [check["check"] for check in checks if not check["passed"]]
        raise ValueError(f"Original 0.5 reproduction failed for {verifier}: {failed}")
    return checks


def main() -> None:
    args = parse_args()
    artifact_root = args.artifacts_dir.expanduser().resolve()
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root
        else artifact_root / "extensions" / "threshold_sensitivity"
    )
    validate_output_root(artifact_root, output_root)

    formal_path = REPO_ROOT / "configs" / "experiments" / "prm_router_formal.json"
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    validate_formal_config(formal)
    re_threshold_path = artifact_root / "evaluation" / "router" / "reasoneval_threshold.json"
    re_threshold_payload = json.loads(re_threshold_path.read_text(encoding="utf-8"))
    if re_threshold_payload.get("selection_split") != "validation":
        raise ValueError("ReasonEval threshold must remain validation-selected")
    re_threshold = float(re_threshold_payload["selected_threshold"])

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".threshold_sensitivity_", dir=output_root.parent)
    )
    summary_rows: list[dict[str, Any]] = []
    reproduction_checks: list[dict[str, Any]] = []
    try:
        for verifier_index, (verifier, spec) in enumerate(VERIFIERS.items()):
            records: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
            provenance = []
            for split in EXPECTED_SPLIT_COUNTS:
                base, second, split_provenance = load_split(
                    artifact_root, verifier, split
                )
                records[split] = (base, second)
                provenance.append(split_provenance)

            validation_labels = np.asarray(
                [int(row["label"]) for row in records["validation"][0]]
            )
            validation_scores = np.asarray(
                [float(row[spec["score_field"]]) for row in records["validation"][1]]
            )
            test_labels = np.asarray([int(row["label"]) for row in records["test"][0]])
            test_scores = np.asarray(
                [float(row[spec["score_field"]]) for row in records["test"][1]]
            )
            curve, selected = threshold_curve(validation_labels, validation_scores)
            selected_threshold = float(selected["threshold"])

            standalone_rows = []
            for condition, threshold in (
                ("original_0.5", ORIGINAL_THRESHOLD),
                ("validation_selected", selected_threshold),
            ):
                val_metrics = metric_row(validation_labels, validation_scores, threshold)
                test_metrics = metric_row(test_labels, test_scores, threshold)
                standalone_rows.append(
                    {
                        "condition": condition,
                        "threshold": threshold,
                        "validation_accuracy": val_metrics["accuracy"],
                        "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
                        "validation_positive_predictions": val_metrics["positive_predictions"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                        "test_positive_predictions": test_metrics["positive_predictions"],
                    }
                )

            router_results = {}
            for condition_index, (condition, threshold) in enumerate(
                (("original_0.5", ORIGINAL_THRESHOLD), ("validation_selected", selected_threshold))
            ):
                router_results[condition] = evaluate_router(
                    verifier,
                    records,
                    re_threshold,
                    threshold,
                    formal,
                    bootstrap_offset=300 + 10 * verifier_index + condition_index,
                )

            router_comparison_rows = []
            for condition, result in router_results.items():
                row = next(item for item in standalone_rows if item["condition"] == condition)
                selected_parameters = result["router_selection"]["selected"]
                router_comparison_rows.append(
                    {
                        "condition": condition,
                        "second_stage_threshold": result["second_stage_threshold"],
                        "standalone_validation_accuracy": row["validation_accuracy"],
                        "standalone_test_accuracy": row["test_accuracy"],
                        "lambda_h": selected_parameters["lambda_h"],
                        "mu": selected_parameters["mu"],
                        "validation_routed_accuracy": result["validation"]["routed_accuracy"],
                        "test_routed_accuracy": result["test"]["routed_accuracy"],
                        "test_selected_calls": result["test"]["selected_calls"],
                        "test_beneficial_selected": result["test"]["selected_outcomes"]["beneficial"],
                        "test_neutral_selected": result["test"]["selected_outcomes"]["neutral"],
                        "test_harmful_selected": result["test"]["selected_outcomes"]["harmful"],
                        "test_selected_extra_runtime_seconds": result["test"]["selected_extra_runtime_seconds"],
                    }
                )

            reproduction_checks.extend(
                original_reproduction_checks(
                    artifact_root,
                    verifier,
                    standalone_rows,
                    router_results["original_0.5"],
                )
            )

            selected_payload = {
                "schema_version": 1,
                "scope": SCOPE,
                "verifier": verifier,
                "display_name": spec["display_name"],
                "selection_split": "validation",
                "selection_metric": "balanced_accuracy",
                "tie_break": TIE_BREAK,
                "candidate_generation": "unique_scores_midpoints_and_0_1_endpoints",
                "candidate_count": len(curve),
                "original_threshold": ORIGINAL_THRESHOLD,
                "selected_threshold": selected_threshold,
                "selected_balanced_accuracy": selected["balanced_accuracy"],
                "selected_accuracy": selected["accuracy"],
                "top_candidates": sorted(
                    curve,
                    key=lambda row: (
                        -row["balanced_accuracy"],
                        abs(row["threshold"] - ORIGINAL_THRESHOLD),
                        row["threshold"],
                    ),
                )[:20],
                "pathfinder_constraint": (
                    "threshold_saved_pathfinder_official_score_only_internal_gates_unchanged"
                    if verifier == "pathfinder"
                    else None
                ),
            }
            verifier_root = temporary_root / verifier
            write_csv(verifier_root / "threshold_curve.csv", curve)
            write_json(verifier_root / "selected_threshold.json", selected_payload)
            write_csv(verifier_root / "standalone_comparison.csv", standalone_rows)
            write_csv(verifier_root / "router_comparison.csv", router_comparison_rows)
            write_json(
                verifier_root / "router_results.json",
                {
                    "schema_version": 1,
                    "scope": SCOPE,
                    "verifier": verifier,
                    "reason_eval_threshold": re_threshold,
                    "primary_call_budget": float(formal["routing"]["primary_call_budget"]),
                    "router_fit_split": "train",
                    "parameter_selection_split": "validation",
                    "evaluation_split": "test",
                    "results": router_results,
                },
            )
            write_json(
                verifier_root / "metadata.json",
                {
                    "schema_version": 1,
                    "scope": SCOPE,
                    "post_hoc": True,
                    "test_results_seen_before_analysis": True,
                    "offline_only_no_inference": True,
                    "verifier": verifier,
                    "raw_score_field": spec["score_field"],
                    "stored_predictions_used_for_new_thresholds": False,
                    "reason_eval_threshold": re_threshold,
                    "tie_break": TIE_BREAK,
                    "source_artifacts": provenance,
                    "code": {
                        "git_head_before_analysis": git_head(),
                        "script": Path(__file__).relative_to(REPO_ROOT).as_posix(),
                        "script_sha256": sha256(Path(__file__)),
                        "formal_config": formal_path.relative_to(REPO_ROOT).as_posix(),
                        "formal_config_sha256": sha256(formal_path),
                        "reason_eval_threshold_artifact": re_threshold_path.relative_to(
                            artifact_root
                        ).as_posix(),
                        "reason_eval_threshold_artifact_sha256": sha256(re_threshold_path),
                    },
                },
            )

            for row in router_comparison_rows:
                summary_rows.append({"verifier": spec["display_name"], **row})

        original_by_verifier = {
            row["verifier"]: row
            for row in summary_rows
            if row["condition"] == "original_0.5"
        }
        for row in summary_rows:
            original = original_by_verifier[row["verifier"]]
            row["standalone_validation_accuracy_delta_vs_0.5"] = (
                row["standalone_validation_accuracy"]
                - original["standalone_validation_accuracy"]
            )
            row["standalone_test_accuracy_delta_vs_0.5"] = (
                row["standalone_test_accuracy"] - original["standalone_test_accuracy"]
            )
            row["test_routed_accuracy_delta_vs_0.5"] = (
                row["test_routed_accuracy"] - original["test_routed_accuracy"]
            )
        write_csv(temporary_root / "comparison_summary.csv", summary_rows)
        write_json(
            temporary_root / "reproduction_gate.json",
            {
                "schema_version": 1,
                "scope": SCOPE,
                "purpose": "verify_original_0.5_analysis_before_interpreting_sensitivity",
                "passed": all(check["passed"] for check in reproduction_checks),
                "checks_passed": sum(check["passed"] for check in reproduction_checks),
                "checks_total": len(reproduction_checks),
                "checks": reproduction_checks,
            },
        )
        (temporary_root / "README.md").write_text(
            "# Post-hoc second-stage threshold sensitivity\n\n"
            "This directory is an offline post-hoc sensitivity analysis performed after "
            "the main test results were seen. Thresholds were selected using validation "
            "balanced accuracy only. Ties use the same fixed rule for all verifiers: "
            "closest to 0.5, then the smaller threshold. ReasonEval keeps its existing "
            "validation-selected threshold. PathFinder changes only the threshold applied "
            "to its saved final gated score; its internal gates are unchanged. No model "
            "inference was rerun.\n",
            encoding="utf-8",
        )
        temporary_root.replace(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    print(f"Wrote post-hoc threshold sensitivity outputs to {output_root}")
    for row in summary_rows:
        print(
            f"{row['verifier']:10s} {row['condition']:19s} "
            f"threshold={row['second_stage_threshold']:.8g} "
            f"standalone_test={row['standalone_test_accuracy']:.4f} "
            f"routed_test={row['test_routed_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
