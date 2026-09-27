"""Post-hoc paired comparison of frozen RE--PF routers against PF-only.

Run from the cl-prm-project root after placing this file in
experiments/formal/prm_router/analysis/:
  python experiments/formal/prm_router/analysis/compute_pf_paired_ci.py \
      --artifacts-dir ../cl-prm-artifacts

This script reads the frozen train/test predictions and development choices.
It does not select a new threshold, budget, or routing parameter, and it does
not overwrite the archived one-time test result.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    artifacts = args.artifacts_dir.resolve()
    sys.path.insert(0, str(repo / "src"))

    from cl_prm.data.records import align_verifier_records, correctness_arrays
    from cl_prm.evaluation.bootstrap import bootstrap_group_metric
    from cl_prm.evaluation.cost import cost_features, make_cost_predictor, selected_runtime
    from cl_prm.evaluation.routing import (
        final_predictions, gain_probabilities, make_expected_gain_router,
        router_features, top_budget_indices,
    )

    config = read_json(repo / "configs/experiments/prm_router_formal.json")
    threshold_info = read_json(artifacts / "evaluation/router/reasoneval_threshold.json")
    development = read_json(artifacts / "evaluation/router/pathfinder_development_selection.json")
    frozen = read_json(artifacts / "evaluation/router/formal_test_results.json")
    threshold = float(threshold_info["selected_threshold"])
    budget = float(config["routing"]["primary_call_budget"])
    seed = int(config["formal_splits"]["seed"])
    assert threshold_info["selection_split"] == "validation"
    assert development["scope"] == "formal_train_validation_only"
    assert frozen["scope"] == "held_out_test_once"
    assert threshold == float(frozen["protocol"]["reason_eval_threshold"])
    assert budget == float(frozen["protocol"]["primary_call_budget"])

    train_re = artifacts / "formal/train/reasoneval.jsonl"
    train_pf = artifacts / "formal/train/pathfinder.jsonl"
    test_re = artifacts / "formal/test/reasoneval.jsonl"
    test_pf = artifacts / "formal/test/pathfinder.jsonl"
    for name, path in [("reasoneval", test_re), ("pathfinder", test_pf)]:
        metadata = read_json(artifacts / f"formal/test/{name}.metadata.json")
        assert canonical_sha256(path) == metadata["output"]["sha256"], name

    train_base, train_second = align_verifier_records(train_re, train_pf)
    test_base, test_second = align_verifier_records(test_re, test_pf)
    train = correctness_arrays(train_base, train_second, threshold)
    test = correctness_arrays(test_base, test_second, threshold)

    model = make_expected_gain_router(seed=seed)
    model.fit(router_features(train_base, train["scores"], threshold), train["gain_labels"])
    probabilities = gain_probabilities(
        model, router_features(test_base, test["scores"], threshold)
    )
    expected_gain_score = probabilities[:, 2] - probabilities[:, 0]

    selected_cost = development["cost_aware_selection"]["selected"]
    ridge = make_cost_predictor(alpha=float(config["routing"]["cost_aware"]["ridge_alpha"]))
    train_runtime = np.asarray(
        [float(row["pathfinder_runtime_seconds"]) for row in train_second]
    )
    ridge.fit(cost_features(train_base), train_runtime)
    predicted_cost = np.maximum(ridge.predict(cost_features(test_base)), 0.0)
    cost_score = (
        probabilities[:, 2]
        - float(selected_cost["lambda_h"]) * probabilities[:, 0]
        - float(selected_cost["mu"]) * predicted_cost / np.median(train_runtime)
    )

    groups = np.asarray([
        str(row.get("original_question_group") or row.get("original_question") or row["example_id"])
        for row in test_base
    ], dtype=object)
    assert len(test_base) == 400 and len(set(groups.tolist())) == 242
    labels = test["labels"]
    pf_correct = test["second_predictions"] == labels
    base_runtime = np.asarray([float(x["disprm_runtime_seconds"]) for x in test_base])
    second_runtime = np.asarray([float(x["pathfinder_runtime_seconds"]) for x in test_second])
    samples = int(frozen["protocol"]["bootstrap_samples"])
    confidence = float(frozen["protocol"]["confidence_level"])

    rows = []
    for offset, (name, score) in enumerate([
        ("expected_gain", expected_gain_score), ("cost_aware", cost_score)
    ]):
        selected = top_budget_indices(score, budget)
        routed = final_predictions(
            test["base_predictions"], test["second_predictions"], selected
        )
        routed_correct = routed == labels
        assert len(selected) == 80
        assert np.isclose(np.mean(routed_correct), frozen["accuracy"][name]["estimate"])
        assert np.isclose(np.mean(pf_correct), frozen["accuracy"]["pathfinder_only"]["estimate"])
        assert np.isclose(
            selected_runtime(base_runtime, second_runtime, selected),
            frozen["runtime"][f"{name}_cascade_total_seconds"], atol=1e-9
        )
        if name == "expected_gain":
            assert (np.sum(test["gain_labels"][selected] == 1),
                    np.sum(test["gain_labels"][selected] == 0),
                    np.sum(test["gain_labels"][selected] == -1)) == (54, 12, 14)

        diff = routed_correct.astype(float) - pf_correct.astype(float)
        bootstrap_seed = seed + 103 + offset
        interval = bootstrap_group_metric(diff, groups, samples, confidence, bootstrap_seed)
        rows.append({
            "comparison": f"{name}_minus_pathfinder_only",
            "estimate": interval["estimate"],
            "ci_low": interval["ci_low"],
            "ci_high": interval["ci_high"],
            "router_only_correct": int(np.sum(diff == 1)),
            "pathfinder_only_correct": int(np.sum(diff == -1)),
            "examples": len(labels),
            "question_groups": len(set(groups.tolist())),
            "bootstrap_samples": samples,
            "confidence_level": confidence,
            "bootstrap_seed": bootstrap_seed,
        })

    output = args.output or repo / "outputs/prm_router/formal/analysis/paired_pf_comparison_additional.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['comparison']}: {100*row['estimate']:+.2f} pp "
              f"[95% CI {100*row['ci_low']:+.2f}, {100*row['ci_high']:+.2f}] "
              f"(discordant: {row['router_only_correct']} vs "
              f"{row['pathfinder_only_correct']})")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
