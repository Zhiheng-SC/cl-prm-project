"""Create post-hoc threshold-sensitivity curves and paper-ready PDF figures.

Run after analyze_threshold_sensitivity.py. This script reads saved inference
artifacts, retrains the gain router at each threshold, selects lambda/mu only
on validation, and reports validation and test curves. The test sweep is
DESCRIPTIVE: never select or advertise its maximum as an independent result.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_threshold_sensitivity import (
    GAIN_CLASSES, VERIFIERS, arrays_for_threshold, load_split,
)
from cl_prm.evaluation.cost import cost_features, make_cost_predictor
from cl_prm.evaluation.routing import (
    gain_probabilities, make_expected_gain_router, routed_accuracy,
    router_features, top_budget_indices,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
NAMES = [("pathfinder", "PathFinder", "#345D7E"),
         ("math_prm", "MathPRM", "#BD583D"),
         ("skywork_prm", "Skywork", "#4F8B70")]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def calculate_curve(artifact_root: Path, sensitivity_root: Path, verifier: str,
                    formal: dict, re_threshold: float) -> list[dict[str, float]]:
    records = {}
    for split in ("train", "validation", "test"):
        base, second, _ = load_split(artifact_root, verifier, split)
        records[split] = (base, second)
    arrays = {
        split: arrays_for_threshold(base, second, verifier, re_threshold, 0.5)
        for split, (base, second) in records.items()
    }
    train, validation, test = (arrays[s] for s in ("train", "validation", "test"))
    train_base, val_base, test_base = (records[s][0] for s in ("train", "validation", "test"))
    cost_cfg = formal["routing"]["cost_aware"]
    budget = float(formal["routing"]["primary_call_budget"])
    if budget != 0.2:
        raise ValueError("This figure is intended for the archived 20% call budget.")
    cost_model = make_cost_predictor(alpha=float(cost_cfg["ridge_alpha"]))
    cost_model.fit(cost_features(train_base), train["runtime"])
    median_cost = float(np.median(train["runtime"]))
    val_cost = np.maximum(cost_model.predict(cost_features(val_base)), 0) / median_cost
    test_cost = np.maximum(cost_model.predict(cost_features(test_base)), 0) / median_cost
    train_x = router_features(train_base, train["re_scores"], re_threshold)
    val_x = router_features(val_base, validation["re_scores"], re_threshold)
    test_x = router_features(test_base, test["re_scores"], re_threshold)
    selected_threshold = float(json.loads(
        (sensitivity_root / verifier / "selected_threshold.json").read_text()
    )["selected_threshold"])
    thresholds = sorted(set(np.linspace(0, 1, 101).tolist() + [0.5, selected_threshold]))
    result = []
    for threshold in thresholds:
        threshold = float(threshold)
        tr_pred = (train["second_scores"] >= threshold).astype(int)
        va_pred = (validation["second_scores"] >= threshold).astype(int)
        te_pred = (test["second_scores"] >= threshold).astype(int)
        gain = ((tr_pred == train["labels"]).astype(int)
                - (train["re_predictions"] == train["labels"]).astype(int))
        if set(np.unique(gain).tolist()) != set(GAIN_CLASSES):
            # At a degenerate threshold, a multinomial router cannot be fitted.
            continue
        router = make_expected_gain_router(seed=int(formal["formal_splits"]["seed"]))
        router.fit(train_x, gain)
        val_prob = gain_probabilities(router, val_x)
        test_prob = gain_probabilities(router, test_x)
        candidates = []
        for lam in cost_cfg["lambda_h_candidates"]:
            for mu in cost_cfg["mu_candidates"]:
                lam, mu = float(lam), float(mu)
                val_utility = val_prob[:, 2] - lam * val_prob[:, 0] - mu * val_cost
                val_calls = top_budget_indices(val_utility, budget)
                val_acc = routed_accuracy(validation["labels"], validation["re_predictions"], va_pred, val_calls)
                val_time = float(validation["runtime"][val_calls].sum())
                candidates.append((val_acc, val_time, lam, mu))
        val_acc, _, lam, mu = sorted(candidates, key=lambda row: (-row[0], row[1], row[2], row[3]))[0]
        test_utility = test_prob[:, 2] - lam * test_prob[:, 0] - mu * test_cost
        test_calls = top_budget_indices(test_utility, budget)
        test_acc = routed_accuracy(test["labels"], test["re_predictions"], te_pred, test_calls)
        result.append({
            "verifier": verifier, "threshold": threshold,
            "validation_standalone_accuracy": float(np.mean(va_pred == validation["labels"])),
            "validation_routed_accuracy": val_acc,
            "test_standalone_accuracy": float(np.mean(te_pred == test["labels"])),
            "test_routed_accuracy": test_acc,
            "lambda_h": lam, "mu": mu,
        })
    # Check all six archived validation/test routed outcomes before plotting.
    archived = read_csv(sensitivity_root / verifier / "router_comparison.csv")
    for saved in archived:
        threshold = float(saved["second_stage_threshold"])
        match = next((row for row in result if abs(row["threshold"] - threshold) < 1e-10), None)
        if match is None:
            raise AssertionError(f"Missing archived threshold {verifier}: {threshold}")
        for key in ("validation_routed_accuracy", "test_routed_accuracy", "lambda_h", "mu"):
            if not np.isclose(match[key], float(saved[key]), atol=1e-10):
                raise AssertionError(f"Archived result mismatch for {verifier} {threshold}: {key}")
    return result


def draw_figure(rows: list[dict[str, float]], sensitivity_root: Path,
                split: str, output_dir: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), sharey=True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.25, top=.83, wspace=.13)
    for ax, (verifier, title, color) in zip(axes, NAMES):
        own = [row for row in rows if row["verifier"] == verifier]
        x = [row["threshold"] for row in own]
        y_route = [100 * row[f"{split}_routed_accuracy"] for row in own]
        y_alone = [100 * row[f"{split}_standalone_accuracy"] for row in own]
        ax.plot(x, y_route, color=color, lw=1.65, label="Routed (20% calls)")
        ax.plot(x, y_alone, color=color, alpha=.55, lw=1.35, ls="--", label="Second verifier alone")
        chosen = float(json.loads((sensitivity_root / verifier / "selected_threshold.json").read_text())["selected_threshold"])
        for threshold, mark_color in ((.5, "#4D4D4D"), (chosen, color)):
            row = next(row for row in own if abs(row["threshold"] - threshold) < 1e-10)
            ax.axvline(threshold, color=mark_color, ls=":", lw=.8, alpha=.7)
            ax.scatter(threshold, 100 * row[f"{split}_routed_accuracy"], color=mark_color,
                       s=18, zorder=3, edgecolor="white", linewidth=.3)
        ax.set_title(title, fontsize=9, pad=5)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .25, .5, .75, 1], labels=["0", ".25", ".5", ".75", "1"])
        ax.grid(axis="y", lw=.5, alpha=.17)
        ax.set_xlabel("Threshold", fontsize=8)
    axes[0].set_ylabel(f"{split.capitalize()} accuracy (%)")
    axes[0].set_ylim(45, 85)
    axes[0].set_yticks([50, 60, 70, 80])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(.53, -.005), fontsize=7.5)
    filename = f"threshold_routing_{split}_posthoc"
    fig.savefig(output_dir / f"{filename}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{filename}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", type=Path, default=REPO_ROOT.parent / "cl_finalproject_hg")
    parser.add_argument("--output-dir", type=Path, help="Defaults to <artifacts>/extensions/threshold_sensitivity/figures")
    args = parser.parse_args()
    artifact_root = args.artifacts_dir.resolve()
    sensitivity_root = artifact_root / "extensions" / "threshold_sensitivity"
    if not (sensitivity_root / "reproduction_gate.json").is_file():
        raise FileNotFoundError("Run Yimin's analyze_threshold_sensitivity.py first (or download its outputs).")
    gate = json.loads((sensitivity_root / "reproduction_gate.json").read_text())
    if not gate.get("passed"):
        raise ValueError("Original 0.5 reproduction gate did not pass.")
    formal = json.loads((REPO_ROOT / "configs/experiments/prm_router_formal.json").read_text())
    re_threshold = float(json.loads((artifact_root / "evaluation/router/reasoneval_threshold.json").read_text())["selected_threshold"])
    output_dir = (args.output_dir or sensitivity_root / "figures").resolve()
    if output_dir == sensitivity_root or sensitivity_root not in output_dir.parents:
        raise ValueError("Write figure outputs beneath extensions/threshold_sensitivity/ only.")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for verifier, _, _ in NAMES:
        rows.extend(calculate_curve(artifact_root, sensitivity_root, verifier, formal, re_threshold))
    csv_path = output_dir / "routed_threshold_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for split in ("validation", "test"):
        draw_figure(rows, sensitivity_root, split, output_dir)
    print(f"Validated archived endpoints and saved {len(rows)} threshold rows to {csv_path}")
    print(f"Paper-ready PDF and PNG previews saved to {output_dir}")
    print("Post-hoc descriptive test sweep: do not select a new test-optimal threshold.")


if __name__ == "__main__":
    main()
