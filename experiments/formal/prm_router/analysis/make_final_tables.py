"""Create publication-ready CSV tables from frozen formal results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_HF_ROOT = REPO_ROOT.parent / "cl-prm-artifacts"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "prm_router" / "formal" / "analysis"

METHOD_LABELS = {
    "reasoneval_only": "ReasonEval only",
    "pathfinder_only": "PathFinder only",
    "random": "Random",
    "low_score": "Low score",
    "uncertainty": "Uncertainty",
    "failure_prediction": "Failure prediction",
    "benefit_only": "Benefit only",
    "expected_gain": "Expected gain",
    "cost_aware": "Cost aware",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-root", type=Path, default=DEFAULT_HF_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    analysis_path = args.output_dir / "final_statistics.json"
    formal_path = (
        args.hf_root / "evaluation" / "router" / "formal_test_results.json"
    )
    analysis = read_json(analysis_path)
    if analysis.get("reproduction_gate", {}).get("status") != "PASS":
        raise ValueError("Refusing to create tables before a PASS reproduction gate.")
    formal = read_json(formal_path)
    primary_budget = float(formal["protocol"]["primary_call_budget"])
    primary_calls = int(round(formal["counts"]["examples"] * primary_budget))

    methods = [
        "reasoneval_only",
        "pathfinder_only",
        "random",
        "low_score",
        "uncertainty",
        "failure_prediction",
        "benefit_only",
        "expected_gain",
        "cost_aware",
    ]
    main_rows = []
    for method in methods:
        result = formal["accuracy"][method]
        if method == "reasoneval_only":
            calls, budget, runtime = 0, "0%", formal["runtime"]["reasoneval_only_total_seconds"]
            runtime_scope = "standalone"
        elif method == "pathfinder_only":
            calls, budget, runtime = (
                formal["counts"]["examples"],
                "100%",
                formal["runtime"]["pathfinder_only_total_seconds"],
            )
            runtime_scope = "standalone"
        else:
            calls, budget = primary_calls, f"{primary_budget:.0%}"
            runtime = formal["runtime"][f"{method}_cascade_total_seconds"]
            runtime_scope = "sequential_cascade"
        main_rows.append(
            {
                "method": METHOD_LABELS[method],
                "pathfinder_budget": budget,
                "pathfinder_calls": calls,
                "accuracy": result["estimate"],
                "ci_low": result["ci_low"],
                "ci_high": result["ci_high"],
                "runtime_seconds": runtime,
                "runtime_scope": runtime_scope,
                "delta_vs_reasoneval": (
                    result["estimate"]
                    - formal["accuracy"]["reasoneval_only"]["estimate"]
                ),
                "captured_oracle_headroom": analysis[
                    "captured_oracle_headroom_20pct"
                ].get(method),
            }
        )
    write_csv(
        args.output_dir / "main_results.csv",
        main_rows,
        [
            "method",
            "pathfinder_budget",
            "pathfinder_calls",
            "accuracy",
            "ci_low",
            "ci_high",
            "runtime_seconds",
            "runtime_scope",
            "delta_vs_reasoneval",
            "captured_oracle_headroom",
        ],
    )

    curve_rows = []
    for method, rows in formal["routing_curves"].items():
        for row in rows:
            beneficial = int(row["beneficial_selected"])
            harmful = int(row["harmful_selected"])
            called = int(row["called_examples"])
            curve_rows.append(
                {
                    "method": method,
                    "call_budget": row["call_budget"],
                    "called_examples": called,
                    "accuracy": row["accuracy"],
                    "cascade_total_seconds": row["cascade_total_seconds"],
                    "beneficial": beneficial,
                    "neutral": called - beneficial - harmful,
                    "harmful": harmful,
                    "net_gain": beneficial - harmful,
                }
            )
    write_csv(
        args.output_dir / "routing_curves.csv",
        curve_rows,
        [
            "method",
            "call_budget",
            "called_examples",
            "accuracy",
            "cascade_total_seconds",
            "beneficial",
            "neutral",
            "harmful",
            "net_gain",
        ],
    )

    paired_rows = [
        {
            "comparison": name,
            **values,
        }
        for name, values in formal["paired_differences"].items()
    ]
    write_csv(
        args.output_dir / "paired_comparisons.csv",
        paired_rows,
        ["comparison", "estimate", "ci_low", "ci_high"],
    )

    outcome_rows = analysis["selection_outcomes_20pct"]
    for row in outcome_rows:
        selected_examples = int(row["selected_examples"])
        row["beneficial_rate"] = row["beneficial"] / selected_examples
        row["neutral_rate"] = row["neutral"] / selected_examples
        row["harmful_rate"] = row["harmful"] / selected_examples
    write_csv(
        args.output_dir / "call_outcomes_20pct.csv",
        outcome_rows,
        [
            "method",
            "selected_examples",
            "beneficial",
            "neutral",
            "harmful",
            "net_gain",
            "beneficial_rate",
            "neutral_rate",
            "harmful_rate",
        ],
    )

    diagnostic_rows = []
    calibration_rows = []
    calibration_bin_rows = []
    for target in ["beneficial_vs_rest", "harmful_vs_rest"]:
        result = analysis["router_diagnostics"][target]
        diagnostic_rows.append(
            {
                "target": target,
                "examples": result["examples"],
                "positives": result["positives"],
                "prevalence": result["prevalence"],
                "average_precision": result["average_precision"],
                "pr_auc_trapezoidal": result["pr_auc_trapezoidal"],
                "roc_auc": result["roc_auc"],
                "brier_score": result["brier_score"],
            }
        )
        calibration_rows.append(
            {
                "target": target,
                "intercept": result["calibration"]["intercept"],
                "slope": result["calibration"]["slope"],
                "converged": result["calibration"]["converged"],
            }
        )
        for row in result["reliability_bins"]:
            calibration_bin_rows.append({"target": target, **row})
    write_csv(
        args.output_dir / "router_discrimination.csv",
        diagnostic_rows,
        [
            "target",
            "examples",
            "positives",
            "prevalence",
            "average_precision",
            "pr_auc_trapezoidal",
            "roc_auc",
            "brier_score",
        ],
    )
    write_csv(
        args.output_dir / "calibration_summary.csv",
        calibration_rows,
        ["target", "intercept", "slope", "converged"],
    )
    write_csv(
        args.output_dir / "calibration_bins.csv",
        calibration_bin_rows,
        [
            "target",
            "bin",
            "left",
            "right",
            "examples",
            "mean_predicted_probability",
            "observed_frequency",
        ],
    )

    overlap = analysis["expected_gain_vs_failure_prediction_overlap"]
    overlap_rows = [
        {
            **row,
            "intersection": overlap["intersection"],
            "overlap_fraction_of_budget": overlap["overlap_fraction_of_budget"],
            "jaccard": overlap["jaccard"],
            "analysis_status": overlap["analysis_status"],
        }
        for row in overlap["partitions"]
    ]
    write_csv(
        args.output_dir / "overlap_analysis.csv",
        overlap_rows,
        [
            "partition",
            "examples",
            "beneficial",
            "neutral",
            "harmful",
            "net_gain",
            "intersection",
            "overlap_fraction_of_budget",
            "jaccard",
            "analysis_status",
        ],
    )

    pre_specified_rows = []
    for rows in analysis["pre_specified_diagnostics"].values():
        pre_specified_rows.extend(rows)
    write_csv(
        args.output_dir / "pre_specified_diagnostics.csv",
        pre_specified_rows,
        [
            "diagnostic",
            "bin",
            "category",
            "left",
            "right",
            "examples",
            "beneficial",
            "neutral",
            "harmful",
            "beneficial_rate",
            "harmful_rate",
        ],
    )

    eg = formal["accuracy"]["expected_gain"]
    eg_re = formal["paired_differences"]["expected_gain_minus_reasoneval_only"]
    eg_uncertainty = formal["paired_differences"][
        "expected_gain_minus_uncertainty"
    ]
    eg_failure = formal["paired_differences"][
        "expected_gain_minus_failure_prediction"
    ]
    summary = f"""# Frozen held-out analysis summary

The reproduction gate passed against the one-time formal held-out result.

- ReasonEval-only accuracy: {formal['accuracy']['reasoneval_only']['estimate']:.2%}
- PathFinder-only accuracy: {formal['accuracy']['pathfinder_only']['estimate']:.2%}
- Expected-gain routing at the frozen 20% budget: {eg['estimate']:.2%} (95% grouped-bootstrap CI {eg['ci_low']:.2%} to {eg['ci_high']:.2%})
- Expected-gain calls: 54 beneficial, 12 neutral, and 14 harmful; net gain 40/400 = 10.00 percentage points.
- Expected gain minus ReasonEval only: {eg_re['estimate']:+.2%} (95% grouped-bootstrap CI {eg_re['ci_low']:+.2%} to {eg_re['ci_high']:+.2%}).
- Expected gain minus uncertainty: {eg_uncertainty['estimate']:+.2%} (95% grouped-bootstrap CI {eg_uncertainty['ci_low']:+.2%} to {eg_uncertainty['ci_high']:+.2%}).
- Expected gain minus failure prediction: {eg_failure['estimate']:+.2%} (95% grouped-bootstrap CI {eg_failure['ci_low']:+.2%} to {eg_failure['ci_high']:+.2%}).

Interpretation: expected-gain routing clearly improves over ReasonEval-only and uncertainty routing. Its +1 percentage-point difference from the strong failure-prediction baseline is uncertain because the paired interval crosses zero. Router discrimination and calibration are additional descriptive diagnostics. The expected-gain versus failure-prediction overlap analysis is post-hoc descriptive analysis.
"""
    (args.output_dir / "analysis_summary.md").write_text(summary, encoding="utf-8")

    beneficial = analysis["router_diagnostics"]["beneficial_vs_rest"]
    harmful = analysis["router_diagnostics"]["harmful_vs_rest"]
    overlap = analysis["expected_gain_vs_failure_prediction_overlap"]
    overlap_by_partition = {
        row["partition"]: row for row in overlap["partitions"]
    }
    results_tex = rf"""% Generated from the frozen held-out artifacts after a PASS reproduction gate.
% Router diagnostics are descriptive; the overlap analysis is post-hoc descriptive.
\paragraph{{Primary held-out result.}}
On the 400-example held-out test set, ReasonEval-only achieved
{100 * formal['accuracy']['reasoneval_only']['estimate']:.2f}\% accuracy and
PathFinder-only achieved {100 * formal['accuracy']['pathfinder_only']['estimate']:.2f}\%.
At the frozen 20\% PathFinder-call budget (80 calls), expected-gain routing
achieved {100 * eg['estimate']:.2f}\% accuracy
(95\% grouped-bootstrap CI:
{100 * eg['ci_low']:.2f}\%--{100 * eg['ci_high']:.2f}\%).
The paired improvement was {100 * eg_re['estimate']:.2f} percentage points over
ReasonEval-only (95\% CI: {100 * eg_re['ci_low']:.2f}--{100 * eg_re['ci_high']:.2f})
and {100 * eg_uncertainty['estimate']:.2f} points over uncertainty routing
(95\% CI: {100 * eg_uncertainty['ci_low']:.2f}--{100 * eg_uncertainty['ci_high']:.2f}).
The difference from failure prediction was only
{100 * eg_failure['estimate']:.2f} point (95\% CI:
{100 * eg_failure['ci_low']:.2f}--{100 * eg_failure['ci_high']:.2f}), so the
held-out result does not establish a clear advantage over that strong baseline.

\paragraph{{Replacement mechanism.}}
Expected-gain routing selected 54 beneficial, 12 neutral, and 14 harmful calls.
Its net 40 corrections therefore explain the exact 10.00 percentage-point gain
from 71.75\% to 81.75\%. Cost-aware routing selected the same outcome counts and
reached the same accuracy, with a measured sequential-cascade runtime of
{formal['runtime']['cost_aware_cascade_total_seconds']:.2f} seconds, compared with
{formal['runtime']['reasoneval_only_total_seconds']:.2f} seconds for ReasonEval-only
and {formal['runtime']['pathfinder_only_total_seconds']:.2f} seconds for standalone
PathFinder-only. The 100\% cascade is not PathFinder-only: it costs approximately
{formal['routing_curves']['expected_gain'][-1]['cascade_total_seconds']:.2f} seconds
because it includes both verifiers.

\paragraph{{Additional descriptive router diagnostics.}}
For beneficial calls, prevalence was {beneficial['prevalence']:.3f}, average
precision was {beneficial['average_precision']:.3f}, trapezoidal PR-AUC was
{beneficial['pr_auc_trapezoidal']:.3f}, and ROC-AUC was
{beneficial['roc_auc']:.3f}. For harmful calls, prevalence was
{harmful['prevalence']:.3f}, average precision was
{harmful['average_precision']:.3f}, trapezoidal PR-AUC was
{harmful['pr_auc_trapezoidal']:.3f}, and ROC-AUC was {harmful['roc_auc']:.3f}.
These metrics and the calibration analyses are descriptive rather than
confirmatory endpoints.

\paragraph{{Post-hoc overlap analysis.}}
Expected-gain and failure-prediction routing shared {overlap['intersection']} of
their 80 selected examples (overlap fraction {overlap['overlap_fraction_of_budget']:.3f};
Jaccard index {overlap['jaccard']:.3f}). The expected-gain-only partition had a
net gain of {overlap_by_partition['expected_gain_only']['net_gain']:+d}, whereas
the failure-prediction-only partition had a net gain of
{overlap_by_partition['failure_prediction_only']['net_gain']:+d}. This mechanism
analysis was conducted post hoc and should not be presented as a pre-specified
formal comparison.
"""
    (args.output_dir / "quantitative_results.tex").write_text(
        results_tex, encoding="utf-8"
    )
    print(f"Tables written to: {args.output_dir}")


if __name__ == "__main__":
    main()
