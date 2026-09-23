"""Descriptive analysis of the frozen ReasonEval--PathFinder test result.

This script refits only the already-frozen train-only routing models and first
checks that their held-out selections reproduce the one-time formal evaluator.
It never changes features, thresholds, utility weights, or routing budgets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from cl_prm.data.records import align_verifier_records as align
from cl_prm.data.records import correctness_arrays
from cl_prm.evaluation.cost import cost_features, make_cost_predictor, selected_runtime
from cl_prm.evaluation.routing import (
    final_predictions,
    gain_probabilities,
    make_binary_predictor,
    make_expected_gain_router,
    router_features,
    top_budget_indices,
)
from cl_prm.utils.formal_config import validate_formal_config


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_HF_ROOT = REPO_ROOT.parent / "cl_finalproject_hg"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "prm_router" / "formal" / "analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze the already-frozen held-out PRM Router artifacts."
    )
    parser.add_argument(
        "--hf-root",
        type=Path,
        default=DEFAULT_HF_ROOT,
        help="Local clone of cl-prm-team/cl-prm-artifacts.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "experiments" / "prm_router_formal.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_canonical_lf(path: Path) -> str:
    """Hash text artifacts after undoing Windows CRLF checkout conversion."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_tracked_dirty(path: Path) -> bool | None:
    result = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def fit_calibration(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Fit unpenalized logistic calibration on clipped prediction logits."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logits = np.log(p / (1.0 - p))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept, slope = parameters
        linear = intercept + slope * logits
        fitted = expit(linear)
        loss = float(np.sum(np.logaddexp(0.0, linear) - y * linear))
        residual = fitted - y
        gradient = np.asarray(
            [np.sum(residual), np.sum(residual * logits)], dtype=np.float64
        )
        return loss, gradient

    result = minimize(
        fun=lambda parameters: objective(parameters)[0],
        x0=np.asarray([0.0, 1.0], dtype=np.float64),
        jac=lambda parameters: objective(parameters)[1],
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"Calibration fit failed: {result.message}")
    return {
        "intercept": float(result.x[0]),
        "slope": float(result.x[1]),
        "converged": True,
    }


def reliability_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 5,
) -> list[dict[str, float | int]]:
    """Return fixed quantile-bin reliability summaries."""
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if len(y) != len(p):
        raise ValueError("Labels and probabilities must have equal length.")
    if len(y) == 0 or bins <= 0:
        return []

    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) <= 1:
        return []
    rows: list[dict[str, float | int]] = []
    for index in range(len(edges) - 1):
        left, right = float(edges[index]), float(edges[index + 1])
        if index == len(edges) - 2:
            mask = (p >= left) & (p <= right)
        else:
            mask = (p >= left) & (p < right)
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": index,
                "left": left,
                "right": right,
                "examples": int(np.sum(mask)),
                "mean_predicted_probability": float(np.mean(p[mask])),
                "observed_frequency": float(np.mean(y[mask])),
            }
        )
    return rows


def binary_router_diagnostics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 5,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    precision, recall, _ = precision_recall_curve(y, p)
    return {
        "examples": int(len(y)),
        "positives": int(np.sum(y)),
        "prevalence": float(np.mean(y)),
        "average_precision": float(average_precision_score(y, p)),
        "pr_auc_trapezoidal": float(auc(recall, precision)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "calibration": fit_calibration(y, p),
        "reliability_bins": reliability_bins(y, p, bins=bins),
    }


def selection_outcome(
    name: str,
    selected: np.ndarray,
    gain_labels: np.ndarray,
) -> dict[str, int | str]:
    labels = gain_labels[selected]
    beneficial = int(np.sum(labels == 1))
    harmful = int(np.sum(labels == -1))
    neutral = int(np.sum(labels == 0))
    return {
        "method": name,
        "selected_examples": int(len(selected)),
        "beneficial": beneficial,
        "neutral": neutral,
        "harmful": harmful,
        "net_gain": beneficial - harmful,
    }


def overlap_partitions(
    expected_gain_selected: np.ndarray,
    failure_selected: np.ndarray,
    gain_labels: np.ndarray,
) -> dict[str, Any]:
    size = len(gain_labels)
    eg_mask = np.zeros(size, dtype=bool)
    fp_mask = np.zeros(size, dtype=bool)
    eg_mask[expected_gain_selected] = True
    fp_mask[failure_selected] = True
    masks = {
        "both": eg_mask & fp_mask,
        "expected_gain_only": eg_mask & ~fp_mask,
        "failure_prediction_only": ~eg_mask & fp_mask,
        "neither": ~eg_mask & ~fp_mask,
    }
    rows = []
    for partition, mask in masks.items():
        labels = gain_labels[mask]
        beneficial = int(np.sum(labels == 1))
        harmful = int(np.sum(labels == -1))
        rows.append(
            {
                "partition": partition,
                "examples": int(np.sum(mask)),
                "beneficial": beneficial,
                "neutral": int(np.sum(labels == 0)),
                "harmful": harmful,
                "net_gain": beneficial - harmful,
            }
        )
    intersection = int(np.sum(eg_mask & fp_mask))
    union = int(np.sum(eg_mask | fp_mask))
    return {
        "expected_gain_selected": int(np.sum(eg_mask)),
        "failure_prediction_selected": int(np.sum(fp_mask)),
        "intersection": intersection,
        "overlap_fraction_of_budget": float(intersection / np.sum(eg_mask)),
        "jaccard": float(intersection / union),
        "partitions": rows,
        "analysis_status": "post_hoc_descriptive",
    }


def quantile_gain_diagnostic(
    name: str,
    values: np.ndarray,
    gain_labels: np.ndarray,
    bins: int,
) -> list[dict[str, Any]]:
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) <= 1:
        return []
    rows = []
    for index in range(len(edges) - 1):
        left, right = float(edges[index]), float(edges[index + 1])
        if index == len(edges) - 2:
            mask = (values >= left) & (values <= right)
        else:
            mask = (values >= left) & (values < right)
        if not np.any(mask):
            continue
        labels = gain_labels[mask]
        rows.append(
            {
                "diagnostic": name,
                "bin": index,
                "left": left,
                "right": right,
                "examples": int(np.sum(mask)),
                "beneficial": int(np.sum(labels == 1)),
                "neutral": int(np.sum(labels == 0)),
                "harmful": int(np.sum(labels == -1)),
                "beneficial_rate": float(np.mean(labels == 1)),
                "harmful_rate": float(np.mean(labels == -1)),
            }
        )
    return rows


def categorical_gain_diagnostic(
    records: list[dict[str, Any]],
    gain_labels: np.ndarray,
    field: str,
) -> list[dict[str, Any]]:
    rows = []
    categories = sorted({str(record.get(field, "unknown")) for record in records})
    for category in categories:
        mask = np.asarray(
            [str(record.get(field, "unknown")) == category for record in records],
            dtype=bool,
        )
        labels = gain_labels[mask]
        rows.append(
            {
                "diagnostic": "error_type",
                "category": category,
                "examples": int(np.sum(mask)),
                "beneficial": int(np.sum(labels == 1)),
                "neutral": int(np.sum(labels == 0)),
                "harmful": int(np.sum(labels == -1)),
                "beneficial_rate": float(np.mean(labels == 1)),
                "harmful_rate": float(np.mean(labels == -1)),
            }
        )
    return rows


def gate_check(
    checks: list[dict[str, Any]],
    name: str,
    actual: float | int,
    expected: float | int,
    tolerance: float = 1e-12,
) -> None:
    passed = bool(abs(float(actual) - float(expected)) <= tolerance)
    checks.append(
        {
            "name": name,
            "actual": actual,
            "expected": expected,
            "tolerance": tolerance,
            "passed": passed,
        }
    )


def gate_check_equal(
    checks: list[dict[str, Any]],
    name: str,
    actual: Any,
    expected: Any,
) -> None:
    checks.append(
        {
            "name": name,
            "actual": actual,
            "expected": expected,
            "tolerance": None,
            "passed": bool(actual == expected),
        }
    )


def analyze(config_path: Path, hf_root: Path) -> dict[str, Any]:
    paths = {
        "train_reasoneval": hf_root / "formal" / "train" / "reasoneval.jsonl",
        "train_pathfinder": hf_root / "formal" / "train" / "pathfinder.jsonl",
        "test_reasoneval": hf_root / "formal" / "test" / "reasoneval.jsonl",
        "test_pathfinder": hf_root / "formal" / "test" / "pathfinder.jsonl",
        "threshold": hf_root / "evaluation" / "router" / "reasoneval_threshold.json",
        "development_selection": hf_root
        / "evaluation"
        / "router"
        / "pathfinder_development_selection.json",
        "formal_results": hf_root
        / "evaluation"
        / "router"
        / "formal_test_results.json",
        "test_reasoneval_metadata": hf_root
        / "formal"
        / "test"
        / "reasoneval.metadata.json",
        "test_pathfinder_metadata": hf_root
        / "formal"
        / "test"
        / "pathfinder.metadata.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required frozen artifacts: {missing}")

    config = read_json(config_path)
    validate_formal_config(config)
    threshold_payload = read_json(paths["threshold"])
    development_payload = read_json(paths["development_selection"])
    formal_results = read_json(paths["formal_results"])
    re_metadata = read_json(paths["test_reasoneval_metadata"])
    pf_metadata = read_json(paths["test_pathfinder_metadata"])

    if threshold_payload.get("selection_split") != "validation":
        raise ValueError("ReasonEval threshold was not selected on validation.")
    if development_payload.get("scope") != "formal_train_validation_only":
        raise ValueError("Development selection scope is not train/validation only.")
    if formal_results.get("scope") != "held_out_test_once":
        raise ValueError("Formal result does not identify a one-time held-out evaluation.")

    threshold = float(threshold_payload["selected_threshold"])
    primary_budget = float(config["routing"]["primary_call_budget"])
    seed = int(config["formal_splits"]["seed"])
    selected_cost = development_payload["cost_aware_selection"]["selected"]
    lambda_h = float(selected_cost["lambda_h"])
    mu = float(selected_cost["mu"])

    train_base, train_second = align(
        paths["train_reasoneval"], paths["train_pathfinder"]
    )
    test_base, test_second = align(paths["test_reasoneval"], paths["test_pathfinder"])
    train = correctness_arrays(train_base, train_second, threshold)
    test = correctness_arrays(test_base, test_second, threshold)
    x_train = router_features(train_base, train["scores"], threshold)
    x_test = router_features(test_base, test["scores"], threshold)

    gain_router = make_expected_gain_router(seed=seed)
    gain_router.fit(x_train, train["gain_labels"])
    gain_probability = gain_probabilities(gain_router, x_test)
    p_harm = gain_probability[:, 0]
    p_benefit = gain_probability[:, 2]
    expected_gain = p_benefit - p_harm

    train_failure = (~train["base_correct"]).astype(np.int64)
    failure_model = make_binary_predictor(seed)
    failure_model.fit(x_train, train_failure)
    failure_score = failure_model.predict_proba(x_test)[:, 1]

    train_benefit = (train["gain_labels"] == 1).astype(np.int64)
    benefit_model = make_binary_predictor(seed + 1)
    benefit_model.fit(x_train, train_benefit)
    benefit_score = benefit_model.predict_proba(x_test)[:, 1]

    uncertainty_score = -np.abs(test["scores"] - threshold)
    low_score = -test["scores"]
    random_seed = int(config["routing"].get("random_baseline_seed", seed))
    random_score = np.random.default_rng(random_seed).random(len(test_base))

    cost_config = config["routing"]["cost_aware"]
    cost_predictor = make_cost_predictor(alpha=float(cost_config["ridge_alpha"]))
    train_cost_x = cost_features(train_base)
    test_cost_x = cost_features(test_base)
    train_pf_runtime = np.asarray(
        [float(record["pathfinder_runtime_seconds"]) for record in train_second],
        dtype=np.float64,
    )
    cost_predictor.fit(train_cost_x, train_pf_runtime)
    predicted_test_runtime = np.maximum(cost_predictor.predict(test_cost_x), 0.0)
    normalized_cost = predicted_test_runtime / float(np.median(train_pf_runtime))
    cost_utility = p_benefit - lambda_h * p_harm - mu * normalized_cost

    routing_scores = {
        "random": random_score,
        "low_score": low_score,
        "uncertainty": uncertainty_score,
        "failure_prediction": failure_score,
        "benefit_only": benefit_score,
        "expected_gain": expected_gain,
        "cost_aware": cost_utility,
    }
    selected = {
        name: top_budget_indices(score, primary_budget)
        for name, score in routing_scores.items()
    }

    labels = test["labels"]
    predictions = {
        "reasoneval_only": test["base_predictions"],
        "pathfinder_only": test["second_predictions"],
    }
    for name, indices in selected.items():
        predictions[name] = final_predictions(
            test["base_predictions"], test["second_predictions"], indices
        )

    accuracies = {
        name: float(np.mean(prediction == labels))
        for name, prediction in predictions.items()
    }
    base_runtime = np.asarray(
        [float(record["disprm_runtime_seconds"]) for record in test_base],
        dtype=np.float64,
    )
    pf_runtime = np.asarray(
        [float(record["pathfinder_runtime_seconds"]) for record in test_second],
        dtype=np.float64,
    )
    runtimes = {
        name: selected_runtime(base_runtime, pf_runtime, indices)
        for name, indices in selected.items()
    }
    group_ids = np.asarray(
        [
            str(
                record.get("original_question_group")
                or record.get("original_question")
                or record["example_id"]
            )
            for record in test_base
        ],
        dtype=object,
    )

    outcomes = [
        selection_outcome(name, selected[name], test["gain_labels"])
        for name in [
            "random",
            "low_score",
            "uncertainty",
            "failure_prediction",
            "benefit_only",
            "expected_gain",
            "cost_aware",
        ]
    ]
    outcome_by_name = {row["method"]: row for row in outcomes}

    checks: list[dict[str, Any]] = []
    protocol = formal_results["protocol"]
    counts = formal_results["counts"]
    gate_check(checks, "test_examples", len(test_base), 400, 0.0)
    gate_check(checks, "test_groups", len(set(group_ids.tolist())), 242, 0.0)
    gate_check(checks, "threshold", threshold, 0.9469655402936041)
    gate_check(checks, "lambda_h", lambda_h, 1.5)
    gate_check(checks, "mu", mu, 0.5)
    gate_check(checks, "primary_budget", primary_budget, 0.2)
    gate_check(checks, "protocol_threshold", protocol["reason_eval_threshold"], threshold)
    gate_check_equal(checks, "reasoneval_metadata_status", re_metadata.get("status"), "completed")
    gate_check_equal(checks, "pathfinder_metadata_status", pf_metadata.get("status"), "completed")
    gate_check_equal(
        checks,
        "formal_input_sha256_match",
        re_metadata.get("input", {}).get("sha256"),
        pf_metadata.get("input", {}).get("sha256"),
    )
    gate_check_equal(
        checks,
        "formal_environment_match",
        re_metadata.get("environment"),
        pf_metadata.get("environment"),
    )
    gate_check_equal(
        checks,
        "formal_code_commit_match",
        re_metadata.get("git", {}).get("commit"),
        pf_metadata.get("git", {}).get("commit"),
    )
    gate_check_equal(
        checks,
        "reasoneval_output_sha256",
        sha256_canonical_lf(paths["test_reasoneval"]),
        re_metadata.get("output", {}).get("sha256"),
    )
    gate_check_equal(
        checks,
        "pathfinder_output_sha256",
        sha256_canonical_lf(paths["test_pathfinder"]),
        pf_metadata.get("output", {}).get("sha256"),
    )
    gate_check(checks, "total_beneficial", int(np.sum(test["gain_labels"] == 1)), counts["beneficial_calls"], 0.0)
    gate_check(checks, "total_neutral", int(np.sum(test["gain_labels"] == 0)), counts["neutral_calls"], 0.0)
    gate_check(checks, "total_harmful", int(np.sum(test["gain_labels"] == -1)), counts["harmful_calls"], 0.0)
    for name, accuracy in accuracies.items():
        gate_check(
            checks,
            f"accuracy_{name}",
            accuracy,
            formal_results["accuracy"][name]["estimate"],
        )
    for name, runtime in runtimes.items():
        gate_check(
            checks,
            f"runtime_{name}",
            runtime,
            formal_results["runtime"][f"{name}_cascade_total_seconds"],
            1e-9,
        )
    gate_check(checks, "expected_gain_selected", len(selected["expected_gain"]), 80, 0.0)
    gate_check(checks, "cost_aware_selected", len(selected["cost_aware"]), 80, 0.0)
    gate_check(checks, "expected_gain_beneficial", outcome_by_name["expected_gain"]["beneficial"], 54, 0.0)
    gate_check(checks, "expected_gain_neutral", outcome_by_name["expected_gain"]["neutral"], 12, 0.0)
    gate_check(checks, "expected_gain_harmful", outcome_by_name["expected_gain"]["harmful"], 14, 0.0)

    failed = [check for check in checks if not check["passed"]]
    if failed:
        details = "; ".join(
            f"{check['name']}: {check['actual']} != {check['expected']}"
            for check in failed
        )
        raise AssertionError(f"Reproduction gate failed; analysis stopped. {details}")

    diagnostic_config = config["evaluation"]["diagnostic_analyses"]
    diagnostics = {
        "reason_eval_confidence": quantile_gain_diagnostic(
            "reason_eval_confidence",
            test["scores"],
            test["gain_labels"],
            int(diagnostic_config["confidence_bins"]),
        ),
        "step_position": quantile_gain_diagnostic(
            "step_position",
            np.asarray([float(record["step_position"]) for record in test_base]),
            test["gain_labels"],
            int(diagnostic_config["step_position_bins"]),
        ),
        "input_length": quantile_gain_diagnostic(
            "input_length",
            np.asarray(
                [float(record["disprm_input_tokens"]) for record in test_base]
            ),
            test["gain_labels"],
            int(diagnostic_config["input_length_bins"]),
        ),
        "error_type": categorical_gain_diagnostic(
            test_base,
            test["gain_labels"],
            str(diagnostic_config["error_type_field"]),
        ),
    }

    metadata_commits = sorted(
        {
            str(re_metadata.get("git", {}).get("commit", "")),
            str(pf_metadata.get("git", {}).get("commit", "")),
        }
        - {""}
    )
    return {
        "schema_version": 1,
        "scope": "frozen_held_out_descriptive_analysis",
        "reproduction_gate": {"status": "PASS", "checks": checks},
        "protocol": {
            "reason_eval_threshold": threshold,
            "lambda_h": lambda_h,
            "mu": mu,
            "primary_call_budget": primary_budget,
            "router_fit_split": "train",
            "selection_split": "validation",
            "evaluation_split": "test",
        },
        "provenance": {
            "analysis_code_base_commit": git_commit(REPO_ROOT),
            "analysis_code_tracked_worktree_dirty": git_tracked_dirty(REPO_ROOT),
            "analysis_script_sha256": {
                "analyze_test_statistics.py": sha256_file(Path(__file__)),
                "make_final_tables.py": sha256_file(
                    Path(__file__).with_name("make_final_tables.py")
                ),
                "make_final_figures.py": sha256_file(
                    Path(__file__).with_name("make_final_figures.py")
                ),
            },
            "hf_artifacts_commit": git_commit(hf_root),
            "formal_inference_code_commits": metadata_commits,
            "input_sha256": {
                name: sha256_file(path) for name, path in paths.items()
            },
            "formal_jsonl_canonical_lf_sha256": {
                "test_reasoneval": sha256_canonical_lf(
                    paths["test_reasoneval"]
                ),
                "test_pathfinder": sha256_canonical_lf(
                    paths["test_pathfinder"]
                ),
            },
            "hash_note": (
                "Formal metadata hashes use LF bytes. Windows Git may check out "
                "JSONL with CRLF, so the reproduction gate compares canonical LF bytes "
                "while also recording raw working-tree hashes."
            ),
        },
        "counts": {
            "examples": len(test_base),
            "original_question_groups": len(set(group_ids.tolist())),
            "beneficial": int(np.sum(test["gain_labels"] == 1)),
            "neutral": int(np.sum(test["gain_labels"] == 0)),
            "harmful": int(np.sum(test["gain_labels"] == -1)),
        },
        "reconstructed_accuracy": accuracies,
        "captured_oracle_headroom_20pct": {
            name: float(
                (accuracies[name] - accuracies["reasoneval_only"])
                / (
                    float(formal_results["oracle"]["pairwise_accuracy"])
                    - accuracies["reasoneval_only"]
                )
            )
            for name in routing_scores
        },
        "selection_outcomes_20pct": outcomes,
        "router_diagnostics": {
            "analysis_status": "additional_descriptive_router_diagnostics",
            "beneficial_vs_rest": binary_router_diagnostics(
                (test["gain_labels"] == 1).astype(np.int64), p_benefit
            ),
            "harmful_vs_rest": binary_router_diagnostics(
                (test["gain_labels"] == -1).astype(np.int64), p_harm
            ),
        },
        "expected_gain_vs_failure_prediction_overlap": overlap_partitions(
            selected["expected_gain"],
            selected["failure_prediction"],
            test["gain_labels"],
        ),
        "pre_specified_diagnostics": diagnostics,
    }


def main() -> None:
    args = parse_args()
    payload = analyze(args.config.resolve(), args.hf_root.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "final_statistics.json"
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Reproduction gate: PASS")
    print(f"Checks:            {len(payload['reproduction_gate']['checks'])}")
    print("Expected gain:     0.8175 (54 beneficial, 12 neutral, 14 harmful)")
    print(f"Output:            {output}")


if __name__ == "__main__":
    main()
