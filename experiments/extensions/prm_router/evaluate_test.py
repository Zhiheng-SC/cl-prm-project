"""Apply frozen RE -> alternative PRM routing to the exploratory test split.

Run from the project root with the HF artifact checkout as a sibling directory.
This script performs CPU-only analysis; it never runs either 7B verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

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
    top_budget_indices,
)
from evaluate_development import arrays  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("math_prm", "skywork_prm"))
    parser.add_argument(
        "--artifacts-dir", type=Path, default=REPO_ROOT.parent / "cl-prm-artifacts",
        help="HF artifacts checkout; defaults to a sibling of cl-prm-project.",
    )
    parser.add_argument("--output", type=Path, help="Override the default HF-style result path.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_metadata(re_path: Path, ext_path: Path, split: str, count: int) -> str:
    """Check each output against its sidecar and the shared frozen input hash."""
    hashes = []
    for path in (re_path, ext_path):
        metadata_path = path.with_suffix(".metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "completed":
            raise ValueError(f"Incomplete inference: {metadata_path}")
        source = metadata["input"]
        if int(source["selected_examples"]) != count:
            raise ValueError(f"Wrong {split} example count in {metadata_path}")
        if Path(source["path"]).name != f"{split}.jsonl":
            raise ValueError(f"Wrong split in {metadata_path}")
        if sha256(path) != metadata["output"]["sha256"]:
            raise ValueError(f"Output checksum differs from {metadata_path}")
        hashes.append(source["sha256"])
    if hashes[0] != hashes[1]:
        raise ValueError(f"RE and extension {split} input hashes differ")
    return hashes[0]


def accuracy(prediction: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(prediction == labels))


def main() -> None:
    args = parse_args()
    artifact_root = args.artifacts_dir.expanduser().resolve()
    formal = json.loads((REPO_ROOT / "configs/experiments/prm_router_formal.json").read_text())
    threshold_path = artifact_root / "evaluation/router/reasoneval_threshold.json"
    threshold_info = json.loads(threshold_path.read_text(encoding="utf-8"))
    selection_path = (
        artifact_root / "extensions/full/evaluation" / f"{args.model}_development_selection.json"
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    threshold = float(threshold_info["selected_threshold"])
    budget = float(formal["routing"]["primary_call_budget"])
    seed = int(formal["formal_splits"]["seed"])
    if threshold_info.get("selection_split") != "validation":
        raise ValueError("RE threshold must have been selected on validation")
    if selection.get("scope") != "exploratory_train_validation_only" or selection.get("model") != args.model:
        raise ValueError("Wrong extension development selection")
    if not np.isclose(float(selection["reason_eval_threshold"]), threshold, rtol=0, atol=1e-12):
        raise ValueError("Selected RE thresholds differ")
    if not np.isclose(float(selection["call_budget"]), budget, rtol=0, atol=1e-12):
        raise ValueError("Call budgets differ")
    if selection["train_examples"] != 600 or selection["validation_examples"] != 200:
        raise ValueError("Expected the selected 600/200 development experiment")

    records = {}
    inputs_sha256 = {}
    for split, expected in (("train", 600), ("test", 400)):
        re_path = artifact_root / "formal" / split / "reasoneval.jsonl"
        ext_path = artifact_root / "extensions/full" / split / f"{args.model}.jsonl"
        inputs_sha256[split] = checked_metadata(re_path, ext_path, split, expected)
        base, second = align_verifier_records(re_path, ext_path)
        if len(base) != expected:
            raise ValueError(f"Expected {expected} aligned {split} rows, got {len(base)}")
        if {row.get("formal_split") for row in base + second} != {split}:
            raise ValueError(f"Unexpected formal_split field in {split} records")
        if any(row.get("extension_verifier") != args.model for row in second):
            raise ValueError(f"Wrong extension model in {split} records")
        records[split] = (base, second, arrays(base, second, threshold))

    train_base, _, tr = records["train"]
    test_base, test_second, te = records["test"]
    if set(np.unique(tr["gain"])) != set(GAIN_CLASSES):
        raise ValueError("Train split is missing a gain class")

    gain_router = make_expected_gain_router(seed=seed)
    gain_router.fit(router_features(train_base, tr["scores"], threshold), tr["gain"])
    probs = gain_probabilities(gain_router, router_features(test_base, te["scores"], threshold))

    cost_settings = formal["routing"]["cost_aware"]
    cost_router = make_cost_predictor(alpha=float(cost_settings["ridge_alpha"]))
    cost_router.fit(cost_features(train_base), tr["runtime"])
    predicted_runtime = np.maximum(cost_router.predict(cost_features(test_base)), 0.0)

    lambda_h = float(selection["selected"]["lambda_h"])
    mu = float(selection["selected"]["mu"])
    median_runtime = float(np.median(tr["runtime"]))
    utility = probs[:, 2] - lambda_h * probs[:, 0] - mu * predicted_runtime / median_runtime
    selected = top_budget_indices(utility, budget)
    if len(selected) != 80:
        raise ValueError(f"Expected 80 of 400 test calls, got {len(selected)}")
    routed = final_predictions(te["base"], te["second"], selected)
    base_runtime = np.asarray([float(row["disprm_runtime_seconds"]) for row in test_base])
    if not np.all(np.isfinite(base_runtime)) or np.any(base_runtime <= 0):
        raise ValueError("Invalid test RE runtime")

    selected_flag = np.zeros(len(test_base), dtype=bool)
    selected_flag[selected] = True
    group_ids = np.asarray([
        str(row.get("original_question_group") or row.get("original_question") or row["example_id"])
        for row in test_base
    ], dtype=object)
    bootstrap_samples = int(formal["evaluation"]["bootstrap_samples"])
    confidence_level = float(formal["evaluation"]["confidence_level"])
    paired = (routed == te["labels"]).astype(float) - (te["base"] == te["labels"]).astype(float)
    result = {
        "schema_version": 1,
        "scope": "exploratory_posthoc_test",
        "model": args.model,
        "protocol": {
            "reason_eval_threshold": threshold,
            "call_budget": budget,
            "lambda_h": lambda_h,
            "mu": mu,
            "router_fit_split": "train",
            "parameter_selection_split": "validation",
            "evaluation_split": "test",
            "development_selection": selection_path.relative_to(artifact_root).as_posix(),
            "input_sha256": inputs_sha256,
        },
        "counts": {
            "train_examples": len(train_base),
            "test_examples": len(test_base),
            "test_question_groups": len(set(group_ids.tolist())),
            "selected_calls": len(selected),
            "beneficial_selected": int(np.sum(te["gain"][selected] == 1)),
            "neutral_selected": int(np.sum(te["gain"][selected] == 0)),
            "harmful_selected": int(np.sum(te["gain"][selected] == -1)),
            "beneficial_available": int(np.sum(te["gain"] == 1)),
            "harmful_available": int(np.sum(te["gain"] == -1)),
        },
        "accuracy": {
            "reasoneval_only": accuracy(te["base"], te["labels"]),
            "second_stage_only": accuracy(te["second"], te["labels"]),
            "selected_router": accuracy(routed, te["labels"]),
            "pairwise_oracle": float(np.mean((te["base"] == te["labels"]) | (te["second"] == te["labels"]))),
        },
        "paired_difference_vs_re": bootstrap_group_metric(
            paired, group_ids, bootstrap_samples, confidence_level, seed + 201
        ),
        "runtime_seconds": {
            "reasoneval_only": float(np.sum(base_runtime)),
            "second_stage_only": float(np.sum(te["runtime"])),
            "selected_extra": float(np.sum(te["runtime"][selected])),
            "selected_cascade": float(np.sum(base_runtime) + np.sum(te["runtime"][selected])),
        },
    }
    destination = args.output or (
        artifact_root / "extensions/full/evaluation" / f"{args.model}_test_results.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Order is the stable key order returned by align_verifier_records.
    decisions = destination.with_name(destination.stem.replace("_results", "_decisions") + ".jsonl")
    with decisions.open("w", encoding="utf-8") as stream:
        for i, (base, second) in enumerate(zip(test_base, test_second)):
            row = {
                "example_id": base["example_id"],
                "current_step": int(base["current_step"]),
                "label": int(te["labels"][i]),
                "reasoneval_prediction": int(te["base"][i]),
                "extension_prediction": int(te["second"][i]),
                "selected": bool(selected_flag[i]),
                "router_utility": float(utility[i]),
                "routed_prediction": int(routed[i]),
                "gain_if_called": int(te["gain"][i]),
                "extension_runtime_seconds": float(te["runtime"][i]),
            }
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"model": args.model, "counts": result["counts"],
                      "accuracy": result["accuracy"], "runtime_seconds": result["runtime_seconds"]}, indent=2))
    print(f"Wrote {destination} and {decisions}")


if __name__ == "__main__":
    main()
