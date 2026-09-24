#!/usr/bin/env python3
"""Exploratory RE -> 256-token GenPRM routing with predicted per-call cost.

Place beside genprm_cost_sweep.py. Reads only train/validation artifacts and
the cost-tuning script's validation_predictions.jsonl. Does not access test.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from genprm_cost_sweep import read_split

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))
from cl_prm.evaluation.routing import (  # noqa: E402
    GAIN_CLASSES, gain_probabilities, make_expected_gain_router,
    router_features, top_budget_indices,
)


def load_predictions(path: Path, pairs: list[tuple[dict, dict]], column: str) -> np.ndarray:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    by_id = {row["example_id"]: row for row in rows}
    ids = [g["example_id"] for g, _ in pairs]
    if len(rows) != len(by_id) or set(ids) != set(by_id):
        raise ValueError("Cost predictions must match all validation example_id values exactly")
    result = []
    for g, _ in pairs:
        row = by_id[g["example_id"]]
        if abs(float(row["actual"]) - float(g["genprm_runtime_seconds"])) > 1e-5:
            raise ValueError(f"Cost prediction target differs from GenPRM runtime: {g['example_id']}")
        result.append(float(row[column]))
    output = np.asarray(result, dtype=float)
    if not np.isfinite(output).all():
        raise ValueError("Non-finite cost prediction")
    return np.maximum(output, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifacts-dir", type=Path)
    source.add_argument("--artifacts-zip", type=Path)
    parser.add_argument("--predictions", type=Path, required=True,
                        help="cost_tune_256/full/validation_predictions.jsonl, target=runtime")
    parser.add_argument("--model-columns", nargs="+",
                        default=["selected", "mean", "ridge_re"],
                        help="Columns in validation_predictions.jsonl to compare")
    parser.add_argument("--re-threshold", type=float, required=True,
                        help="Use the existing frozen ReasonEval threshold")
    parser.add_argument("--lambda-h", type=float, default=1.0)
    parser.add_argument("--mu-values", nargs="+", type=float,
                        default=[0.0, 0.05, 0.1, 0.2, 0.5])
    parser.add_argument("--budget", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not (0 <= args.re_threshold <= 1 and 0 < args.budget <= 1):
        parser.error("Threshold must be within [0,1] and budget within (0,1]")
    if args.lambda_h < 0 or any(mu < 0 for mu in args.mu_values):
        parser.error("lambda-h and mu-values must be non-negative")

    train = read_split(args, "train")
    validation = read_split(args, "validation")
    tr_re = [r for _, r in train]
    va_re = [r for _, r in validation]
    tr_labels = np.asarray([g["label"] for g, _ in train], dtype=int)
    va_labels = np.asarray([g["label"] for g, _ in validation], dtype=int)
    tr_scores = np.asarray([r["disprm_score"] for r in tr_re], dtype=float)
    va_scores = np.asarray([r["disprm_score"] for r in va_re], dtype=float)
    tr_base = (tr_scores >= args.re_threshold).astype(int)
    va_base = (va_scores >= args.re_threshold).astype(int)
    tr_gen = np.asarray([g["genprm_prediction"] for g, _ in train], dtype=int)
    va_gen = np.asarray([g["genprm_prediction"] for g, _ in validation], dtype=int)
    tr_gain = (tr_gen == tr_labels).astype(int) - (tr_base == tr_labels).astype(int)
    va_gain = (va_gen == va_labels).astype(int) - (va_base == va_labels).astype(int)
    if set(np.unique(tr_gain)) != set(GAIN_CLASSES):
        raise ValueError("Train split lacks one of the gain classes")

    router = make_expected_gain_router(seed=20260903)
    router.fit(router_features(tr_re, tr_scores, args.re_threshold), tr_gain)
    probs = gain_probabilities(router, router_features(va_re, va_scores, args.re_threshold))
    gain_score = probs[:, 2] - args.lambda_h * probs[:, 0]
    costs = {name: load_predictions(args.predictions, validation, name)
             for name in dict.fromkeys(args.model_columns)}
    train_runtime = np.asarray([g["genprm_runtime_seconds"] for g, _ in train], dtype=float)
    val_runtime = np.asarray([g["genprm_runtime_seconds"] for g, _ in validation], dtype=float)
    re_runtime = np.asarray([r["disprm_runtime_seconds"] for r in va_re], dtype=float)
    median_cost = float(np.median(train_runtime))
    if median_cost <= 0:
        raise ValueError("Non-positive train median runtime")

    gain_selected = top_budget_indices(gain_score, args.budget)
    gain_set = set(map(int, gain_selected))
    rows = []
    selection_flags = {"example_id": [g["example_id"] for g, _ in validation],
                       "label": va_labels, "re_prediction": va_base,
                       "genprm_prediction": va_gen, "measured_genprm_seconds": val_runtime,
                       "gain_probability": probs[:, 2],
                       "harm_probability": probs[:, 0]}
    for model_name, cost in costs.items():
        selection_flags[f"predicted_{model_name}_seconds"] = cost
        for mu in args.mu_values:
            utility = gain_score - mu * cost / median_cost
            selected = top_budget_indices(utility, args.budget)
            final = va_base.copy()
            final[selected] = va_gen[selected]
            called = np.zeros(len(validation), dtype=int)
            called[selected] = 1
            rows.append({
                "cost_model": model_name,
                "mu": mu,
                "called_examples": len(selected),
                "routed_accuracy": float(np.mean(final == va_labels)),
                "correct_examples": int(np.sum(final == va_labels)),
                "measured_genprm_seconds": float(val_runtime[selected].sum()),
                "measured_total_seconds": float(re_runtime.sum() + val_runtime[selected].sum()),
                "predicted_genprm_seconds": float(cost[selected].sum()),
                "beneficial_calls": int(np.sum(va_gain[selected] == 1)),
                "harmful_calls": int(np.sum(va_gain[selected] == -1)),
                "neutral_calls": int(np.sum(va_gain[selected] == 0)),
                "capped_calls": int(sum(not validation[i][0]["genprm_analysis_complete"]
                                        for i in selected)),
                "overlap_with_gain_only": len(gain_set & set(map(int, selected))),
            })
            selection_flags[f"called_{model_name}_mu_{mu:g}"] = called

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "routing_tradeoff.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "routing_predictions.jsonl").open("w", encoding="utf-8") as f:
        for i in range(len(validation)):
            f.write(json.dumps({key: (value[i].item() if hasattr(value[i], "item") else value[i])
                                for key, value in selection_flags.items()}, ensure_ascii=False) + "\n")
    info = {
        "scope": "exploratory_256_analysis_only", "re_threshold": args.re_threshold,
        "lambda_h": args.lambda_h, "budget": args.budget,
        "cost_model_columns": list(costs), "predictions_file": str(args.predictions),
        "train_median_genprm_seconds": median_cost,
        "re_only_accuracy": float(np.mean(va_base == va_labels)),
        "genprm_only_accuracy": float(np.mean(va_gen == va_labels)),
        "validation_examples": len(validation),
        "analysis_capped_examples": int(sum(not g["genprm_analysis_complete"] for g, _ in validation)),
        "note": "Rows vary mu descriptively. Do not choose mu from this repeatedly viewed validation set. "
                "Constant mean cost gives the same ranking as mu=0 for a fixed call count. No test used.",
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"RE-only accuracy={info['re_only_accuracy']:.3f}; GenPRM-only={info['genprm_only_accuracy']:.3f}; "
          f"budget={args.budget:.0%} ({len(gain_selected)} calls)")
    for row in rows:
        print(f"{row['cost_model']:10s} mu={row['mu']:.2f} accuracy={row['routed_accuracy']:.3f} "
              f"GenPRM={row['measured_genprm_seconds']:.1f}s "
              f"beneficial/harmful={row['beneficial_calls']}/{row['harmful_calls']} "
              f"capped={row['capped_calls']} "
              f"overlap={row['overlap_with_gain_only']}/{len(gain_selected)}")
    print(f"Saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
