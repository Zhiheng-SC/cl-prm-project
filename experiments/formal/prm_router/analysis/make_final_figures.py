"""Generate final PDF figures from frozen formal-analysis artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Official *ACL formatting uses 7.7 cm columns.  Creating the paper figures at
# their final width keeps the apparent font, marker, and line sizes predictable.
ACL_COLUMN_WIDTH_INCHES = 7.7 / 2.54

plt.rcParams.update(
    {
        "font.size": 8.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_HF_ROOT = REPO_ROOT.parent / "cl-prm-artifacts"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "prm_router" / "formal" / "analysis"

METHODS = [
    "uncertainty",
    "failure_prediction",
    "benefit_only",
    "expected_gain",
    "cost_aware",
]
LABELS = {
    "uncertainty": "Uncertainty",
    "failure_prediction": "Failure pred.",
    "benefit_only": "Benefit-only",
    "expected_gain": "Expected gain",
    "cost_aware": "Cost-aware",
}
CALL_OUTCOME_LABELS = {
    "uncertainty": "Uncert.",
    "failure_prediction": "Failure\npred.",
    "benefit_only": "Benefit-\nonly",
    "expected_gain": "Exp.\ngain",
    "cost_aware": "Cost-\naware",
}
COLORS = {
    "uncertainty": "#0072B2",
    "failure_prediction": "#E69F00",
    "benefit_only": "#009E73",
    "expected_gain": "#CC79A7",
    "cost_aware": "#D55E00",
}
MARKERS = {
    "uncertainty": "o",
    "failure_prediction": "s",
    "benefit_only": "^",
    "expected_gain": "D",
    "cost_aware": "P",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def style_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.grid(
        True,
        axis=grid_axis,
        color="#D9D9D9",
        linewidth=0.5,
        alpha=0.3,
    )
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def save(figure: plt.Figure, path: Path) -> None:
    figure.savefig(path, metadata={"Creator": "CL-PRM analysis"})
    plt.close(figure)


def accuracy_vs_budget(formal: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(
        figsize=(ACL_COLUMN_WIDTH_INCHES, 3.0), constrained_layout=True
    )
    for method in METHODS:
        rows = formal["routing_curves"][method]
        axis.plot(
            [100.0 * row["call_budget"] for row in rows],
            [100.0 * row["accuracy"] for row in rows],
            label=LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.6,
            markersize=4.5,
        )
    re_accuracy = 100.0 * formal["accuracy"]["reasoneval_only"]["estimate"]
    pf_accuracy = 100.0 * formal["accuracy"]["pathfinder_only"]["estimate"]
    axis.axhline(
        re_accuracy,
        color="#555555",
        linewidth=1.2,
        linestyle="--",
        label="ReasonEval only",
    )
    axis.axhline(
        pf_accuracy,
        color="#000000",
        linewidth=1.2,
        linestyle=":",
        label="PathFinder only",
    )
    axis.axvline(20.0, color="#777777", linewidth=0.9, alpha=0.75)
    axis.set_ylim(69.5, 82.5)
    axis.text(
        0.98,
        0.98,
        "Frozen budget: 20%",
        transform=axis.transAxes,
        fontsize=8,
        ha="right",
        va="top",
    )
    axis.set_xlabel("PathFinder call budget (%)")
    axis.set_ylabel("Held-out accuracy (%)")
    axis.set_xticks([10, 20, 30, 40, 50, 75, 100])
    style_axis(axis)
    axis.legend(
        ncol=2,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0.0,
        columnspacing=0.9,
        handlelength=1.5,
        handletextpad=0.4,
        labelspacing=0.3,
    )
    save(figure, path)


def accuracy_vs_runtime(formal: dict[str, Any], path: Path) -> None:
    figure, axis = plt.subplots(
        figsize=(ACL_COLUMN_WIDTH_INCHES, 3.0), constrained_layout=True
    )
    for method in METHODS:
        rows = formal["routing_curves"][method]
        axis.plot(
            [row["cascade_total_seconds"] for row in rows],
            [100.0 * row["accuracy"] for row in rows],
            label=LABELS[method],
            color=COLORS[method],
            marker=MARKERS[method],
            linewidth=1.6,
            markersize=4.5,
        )
    axis.scatter(
        [formal["runtime"]["reasoneval_only_total_seconds"]],
        [100.0 * formal["accuracy"]["reasoneval_only"]["estimate"]],
        color="#555555",
        marker="X",
        s=45,
        label="RE only (standalone)",
        zorder=5,
    )
    axis.scatter(
        [formal["runtime"]["pathfinder_only_total_seconds"]],
        [100.0 * formal["accuracy"]["pathfinder_only"]["estimate"]],
        color="#000000",
        marker="*",
        s=65,
        label="PF only (standalone)",
        zorder=5,
    )
    axis.annotate(
        "100% cascade",
        xy=(
            formal["routing_curves"]["expected_gain"][-1]["cascade_total_seconds"],
            100.0 * formal["routing_curves"]["expected_gain"][-1]["accuracy"],
        ),
        xytext=(-55, -18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        fontsize=8,
    )
    axis.set_xlabel("Measured total runtime (seconds)")
    axis.set_ylabel("Held-out accuracy (%)")
    axis.set_ylim(69.5, 82.5)
    style_axis(axis)
    axis.legend(
        ncol=2,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0.0,
        columnspacing=0.8,
        handlelength=1.4,
        handletextpad=0.4,
        labelspacing=0.3,
    )
    save(figure, path)


def call_outcomes(analysis: dict[str, Any], path: Path) -> None:
    rows = [
        row
        for row in analysis["selection_outcomes_20pct"]
        if row["method"] in METHODS
    ]
    labels = [CALL_OUTCOME_LABELS[row["method"]] for row in rows]
    beneficial = np.asarray([row["beneficial"] for row in rows])
    neutral = np.asarray([row["neutral"] for row in rows])
    harmful = np.asarray([row["harmful"] for row in rows])
    positions = np.arange(len(rows))
    figure, axis = plt.subplots(
        figsize=(ACL_COLUMN_WIDTH_INCHES, 2.55), constrained_layout=True
    )
    axis.bar(positions, beneficial, color="#009E73", label="Beneficial")
    axis.bar(positions, neutral, bottom=beneficial, color="#BDBDBD", label="Neutral")
    axis.bar(
        positions,
        harmful,
        bottom=beneficial + neutral,
        color="#D55E00",
        label="Harmful",
    )
    for index, row in enumerate(rows):
        axis.text(
            index,
            81.5,
            f"Net\n{row['net_gain']:+d}",
            ha="center",
            va="bottom",
            fontsize=8,
            linespacing=0.9,
        )
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 94)
    axis.set_ylabel("Selected examples")
    style_axis(axis)
    axis.legend(
        ncol=3,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        borderaxespad=0.0,
        columnspacing=0.8,
        handlelength=1.4,
        handletextpad=0.4,
    )
    save(figure, path)


def calibration_figure(
    analysis: dict[str, Any],
    target: str,
    path: Path,
) -> None:
    result = analysis["router_diagnostics"][target]
    rows = result["reliability_bins"]
    x = [row["mean_predicted_probability"] for row in rows]
    y = [row["observed_frequency"] for row in rows]
    figure, axis = plt.subplots(
        figsize=(ACL_COLUMN_WIDTH_INCHES, 2.75), constrained_layout=True
    )
    axis.plot([0, 1], [0, 1], color="#666666", linestyle="--", label="Ideal")
    axis.plot(x, y, color="#0072B2", marker="o", linewidth=2.0, label="Observed")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed frequency")
    axis.text(
        0.02,
        0.98,
        f"Prevalence={result['prevalence']:.3f}\nBrier={result['brier_score']:.3f}",
        transform=axis.transAxes,
        va="top",
        fontsize=8,
    )
    style_axis(axis, grid_axis="both")
    axis.legend(frameon=False, loc="lower right", handlelength=1.5)
    save(figure, path)


def main() -> None:
    args = parse_args()
    formal = read_json(
        args.hf_root / "evaluation" / "router" / "formal_test_results.json"
    )
    analysis = read_json(args.output_dir / "final_statistics.json")
    if analysis.get("reproduction_gate", {}).get("status") != "PASS":
        raise ValueError("Refusing to create figures before a PASS reproduction gate.")
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    accuracy_vs_budget(formal, figure_dir / "accuracy_vs_budget.pdf")
    accuracy_vs_runtime(formal, figure_dir / "accuracy_vs_runtime.pdf")
    call_outcomes(analysis, figure_dir / "call_outcomes_20pct.pdf")
    calibration_figure(
        analysis,
        "beneficial_vs_rest",
        figure_dir / "beneficial_calibration.pdf",
    )
    calibration_figure(
        analysis,
        "harmful_vs_rest",
        figure_dir / "harmful_calibration.pdf",
    )
    print(f"Figures written to: {figure_dir}")


if __name__ == "__main__":
    main()
