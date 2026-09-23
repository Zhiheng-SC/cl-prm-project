"""Train a separate RE->alternative router on train, select on validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cl_prm.data.records import align_verifier_records  # noqa: E402
from cl_prm.evaluation.cost import cost_features, make_cost_predictor  # noqa: E402
from cl_prm.evaluation.routing import (  # noqa: E402
    GAIN_CLASSES, gain_probabilities, make_expected_gain_router,
    routed_accuracy, router_features, top_budget_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("genprm_official", "math_prm", "skywork_prm"), required=True)
    parser.add_argument("--train-reasoneval", type=Path, required=True)
    parser.add_argument("--validation-reasoneval", type=Path, required=True)
    parser.add_argument("--extension-root", type=Path, default=REPO_ROOT / "outputs/prm_router/extensions/full")
    parser.add_argument("--reason-eval-threshold", type=float, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def arrays(base: list[dict], second: list[dict], threshold: float) -> dict[str, np.ndarray]:
    for a, b in zip(base, second):
        if int(a["label"]) != int(b["label"]):
            raise ValueError("Ground-truth labels differ between verifiers.")
        if b.get("extension_verifier") is None or b.get("extension_runtime_seconds") is None:
            raise ValueError("Expected complete official extension inference records.")
        if b.get("extension_status") != "ok" or b.get("extension_prediction") is None:
            raise ValueError("Invalid model judgments found; report failures and choose a fixed policy before routing.")
    labels = np.asarray([int(r["label"]) for r in base])
    scores = np.asarray([float(r["disprm_score"]) for r in base])
    base_pred = (scores >= threshold).astype(int)
    second_pred = np.asarray([int(r["extension_prediction"]) for r in second])
    gain = (second_pred == labels).astype(int) - (base_pred == labels).astype(int)
    runtime = np.asarray([float(r["extension_runtime_seconds"]) for r in second])
    if not np.all(np.isfinite(runtime)) or np.any(runtime <= 0):
        raise ValueError("Missing or invalid measured runtime.")
    return dict(labels=labels, scores=scores, base=base_pred, second=second_pred,
                gain=gain, runtime=runtime)


def main() -> None:
    args = parse_args()
    if not 0 <= args.reason_eval_threshold <= 1:
        raise ValueError("RE threshold must be in [0, 1].")
    tr_base, tr_second = align_verifier_records(
        args.train_reasoneval, args.extension_root / "train" / f"{args.model}.jsonl"
    )
    va_base, va_second = align_verifier_records(
        args.validation_reasoneval, args.extension_root / "validation" / f"{args.model}.jsonl"
    )
    if any(r["extension_verifier"] != args.model for r in tr_second + va_second):
        raise ValueError("Verifier field does not match selected extension model.")
    tr = arrays(tr_base, tr_second, args.reason_eval_threshold)
    va = arrays(va_base, va_second, args.reason_eval_threshold)
    if set(np.unique(tr["gain"])) != set(GAIN_CLASSES):
        raise ValueError("Training split lacks a gain class; do not fit the multinomial router.")

    router = make_expected_gain_router(seed=2026)
    router.fit(router_features(tr_base, tr["scores"], args.reason_eval_threshold), tr["gain"])
    probs = gain_probabilities(router, router_features(va_base, va["scores"], args.reason_eval_threshold))
    p_harm, p_benefit = probs[:, 0], probs[:, 2]
    cost = make_cost_predictor(alpha=1.0)
    cost.fit(cost_features(tr_base), tr["runtime"])
    pred_cost = np.maximum(cost.predict(cost_features(va_base)), 0.0)
    median_cost = float(np.median(tr["runtime"]))
    # Match the formal RE->PF candidate grid and call budget while refitting
    # every learned component for this model.
    formal = json.loads((REPO_ROOT / "configs/experiments/prm_router_formal.json").read_text())
    settings = formal["routing"]["cost_aware"]
    budget = float(formal["routing"]["primary_call_budget"])
    candidates = []
    for lambda_h in settings["lambda_h_candidates"]:
        for mu in settings["mu_candidates"]:
            utility = p_benefit - float(lambda_h) * p_harm - float(mu) * pred_cost / median_cost
            selected = top_budget_indices(utility, budget)
            candidates.append({
                "lambda_h": lambda_h, "mu": mu,
                "validation_accuracy": routed_accuracy(va["labels"], va["base"], va["second"], selected),
                "measured_extra_runtime_seconds": float(va["runtime"][selected].sum()),
                "called_examples": len(selected),
            })
    candidates.sort(key=lambda r: (-r["validation_accuracy"], r["measured_extra_runtime_seconds"], r["lambda_h"], r["mu"]))
    output = {
        "scope": "exploratory_train_validation_only", "model": args.model,
        "reason_eval_threshold": args.reason_eval_threshold, "call_budget": budget,
        "train_examples": len(tr_base), "validation_examples": len(va_base),
        "validation": {
            "reasoneval_accuracy": float(np.mean(va["base"] == va["labels"])),
            "second_stage_accuracy": float(np.mean(va["second"] == va["labels"])),
            "oracle_accuracy": float(np.mean((va["base"] == va["labels"]) | (va["second"] == va["labels"]))),
            "beneficial": int(np.sum(va["gain"] == 1)),
            "neutral": int(np.sum(va["gain"] == 0)),
            "harmful": int(np.sum(va["gain"] == -1)),
        },
        "cost_predictor": {
            "model": "ridge_regression", "validation_mae_seconds": float(np.mean(np.abs(pred_cost - va["runtime"]))),
            "validation_r2": float(r2_score(va["runtime"], pred_cost)),
            "median_baseline_mae_seconds": float(np.mean(np.abs(median_cost - va["runtime"]))),
            "train_median_runtime_seconds": median_cost,
        },
        "selected": candidates[0], "candidates": candidates,
    }
    destination = args.output or args.extension_root.parent / "evaluation" / f"{args.model}_development.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "validation": output["validation"],
                      "selected": output["selected"], "cost_predictor": output["cost_predictor"]}, indent=2))
    print(f"Saved to {destination}")


if __name__ == "__main__":
    main()
