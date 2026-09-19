import os
import argparse
from pathlib import Path
from copy import deepcopy

from helpers import read_json, read_jsonl, write_json
from metrics import *


def load(config: dict, manifest: dict, input_dir: Path) -> tuple:
    seeds = [seed for seed in config["inference"]["seeds"]]
    indices = [src_idx for src_idx in manifest["source_indices"]]
    examples = config["dataset"]["pilot_examples"]

    if len(indices) != examples:
        raise ValueError(f"Manifest has {len(indices)} indices but expected are {examples}")

    seed2out, all_seeds_combined = {}, []
    for seed in seeds:
        path = input_dir / f"seed_{seed}.jsonl"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing seed file at {path}")

        rows = read_jsonl(path)

        seed2out[seed] = rows
        all_seeds_combined.extend(rows)

    return seed2out, all_seeds_combined


def metric_values(rows: list) -> dict:
    greedy_acc, n = 0, len(rows)
    sc_acc = 0
    orm_acc, ovm_acc = 0, 0
    ovm_last_expansion_corr, ovm_final_beam_corr = 0, 0
    greedy_runtime, sc_runtime, orm_runtime, ovm_runtime = 0.0, 0.0, 0.0, 0.0
    vanilla_correct_fraction = 0
    max_peak_gb, max_res_gb = float("-inf"), float("-inf")

    if n == 0:
        raise ValueError("Empty input to calc metrics")

    for row in rows:

        greedy_correct = row["greedy"]["correct"]
        sc_correct = row["sampling"]["majority_correct"]
        orm_correct, ovm_correct = row["orm"]["correct"], row["ovm"]["correct"]

        ovm_last_expansion_corr += row["ovm"]["last_expansion_correct_mean"]
        ovm_final_beam_corr += row["ovm"]["final_beam_correct_mean"]

        if greedy_correct: greedy_acc += 1
        if sc_correct: sc_acc += 1
        if orm_correct: orm_acc += 1
        if ovm_correct: ovm_acc += 1

        greedy_runtime += row["greedy"]["runtime_seconds"]
        sc_runtime += row["sampling"]["runtime_seconds"]
        orm_runtime += row["orm"]["runtime_seconds"]
        ovm_runtime += row["ovm"]["runtime_seconds"]

        vanilla_correct_fraction += row["sampling"]["correct_mean"]

        for method in ("greedy", "sampling", "orm", "ovm"):
            allocated_gbs = row[method]["peak_allocated_bytes"] / (2 ** 30)
            reserved_gbs = row[method]["peak_reserved_bytes"] / (2 ** 30)

            if max_peak_gb < allocated_gbs:
                max_peak_gb = allocated_gbs
            if max_res_gb < reserved_gbs:
                max_res_gb = reserved_gbs

    return {
        GREEDY_ACC: greedy_acc / n,
        SELF_CONSISTENCY_ACC: sc_acc / n,
        ORM_ACC: orm_acc / n,
        OVM_ACC: ovm_acc / n,
        OVM_LAST_EXPANSION_CORRECT_FRACTION: ovm_last_expansion_corr / n,
        OVM_FINAL_BEAM_CORRECT_FRACTION: ovm_final_beam_corr / n,
        VANILLA_CORRECT_FRACTION: vanilla_correct_fraction / n,
        GREEDY_RUNTIME_SEC: greedy_runtime / n,
        SAMPLING_RUNTIME_SEC: sc_runtime / n,
        ORM_RUNTIME_SEC: orm_runtime / n,
        OVM_RUNTIME_SEC: ovm_runtime / n,
        MAX_PEAK_ALLOCATED_GIB: max_peak_gb,
        MAX_PEAK_RESERVED_GIB: max_res_gb
    }


def add_result(metrics: dict) -> dict:
    result = deepcopy(metrics)

    result.update(
        {
            OVM_MINUS_SELF_CONSISTENCY: metrics[OVM_ACC] - metrics[SELF_CONSISTENCY_ACC],
            OVM_MINUS_GREEDY: metrics[OVM_ACC] - metrics[GREEDY_ACC],
            OVM_MINUS_ORM: metrics[OVM_ACC] - metrics[ORM_ACC],
            EXPANSION_MINUS_VANILLA_CORRECT: metrics[OVM_LAST_EXPANSION_CORRECT_FRACTION] - metrics[
                VANILLA_CORRECT_FRACTION],
            OVM_SAMPLING_RUNTIME_RATIO: metrics[OVM_RUNTIME_SEC] / metrics[SAMPLING_RUNTIME_SEC]
        }
    )

    return result


def evaluate_study(config: dict, per_seed: dict, aggregate: dict) -> list:
    thresholds, hardware = config["feasibility_gates"], config["hardware"]
    minimum_completed_seeds = thresholds["minimum_completed_seeds"]

    # check per-requirement and assign pass/fail label
    ## hardware requirements are also feasibility gates
    feasibility_gates = [
        {
            "name": "minimum_completed_seeds",
            "value": len(per_seed),
            "threshold": minimum_completed_seeds,
            "passed": len(per_seed) >= minimum_completed_seeds,
        },
        {
            "name": "ovm_minus_self_consistency_accuracy",
            "value": aggregate[OVM_MINUS_SELF_CONSISTENCY],
            "threshold": thresholds["minimum_ovm_minus_self_consistency_accuracy"],
            "passed": aggregate[OVM_MINUS_SELF_CONSISTENCY] >= thresholds["minimum_ovm_minus_self_consistency_accuracy"]
        },
        {
            "name": "no_seed_worse_than_self_consistency",
            "value": min(value[OVM_MINUS_SELF_CONSISTENCY] for value in per_seed.values()),
            "threshold": 0.0,
            "passed": min(value[OVM_MINUS_SELF_CONSISTENCY] for value in per_seed.values()) >= 0.0,
        },
        {
            "name": "ovm_minus_greedy_accuracy",
            "value": aggregate[OVM_MINUS_GREEDY],
            "threshold": thresholds["minimum_ovm_minus_greedy_accuracy"],
            "passed": aggregate[OVM_MINUS_GREEDY] >= thresholds["minimum_ovm_minus_greedy_accuracy"]
        },
        {
            "name": "last_expansion_minus_vanilla_correct_fraction",
            "value": aggregate[EXPANSION_MINUS_VANILLA_CORRECT],
            "threshold": thresholds["minimum_last_expansion_minus_vanilla_correct_fraction"],
            "passed": aggregate[EXPANSION_MINUS_VANILLA_CORRECT] >= thresholds[
                "minimum_last_expansion_minus_vanilla_correct_fraction"],
        },
        {
            "name": "ovm_minus_orm_accuracy",
            "value": aggregate[OVM_MINUS_ORM],
            "threshold": thresholds["minimum_ovm_minus_orm_accuracy"],
            "passed": aggregate[OVM_MINUS_ORM] >= thresholds["minimum_ovm_minus_orm_accuracy"],
        },
        {
            "name": "ovm_to_sampling_runtime_ratio",
            "value": aggregate[OVM_SAMPLING_RUNTIME_RATIO],
            "threshold": thresholds["maximum_ovm_to_sampling_runtime_ratio"],
            "passed": aggregate[OVM_SAMPLING_RUNTIME_RATIO] <= thresholds["maximum_ovm_to_sampling_runtime_ratio"],
        },
        {
            "name": "maximum_peak_allocated_gib",
            "value": aggregate["maximum_peak_allocated_gib"],
            "threshold": hardware["max_peak_memory_gib"],
            "passed": aggregate["maximum_peak_allocated_gib"] <= hardware["max_peak_memory_gib"]
        },
    ]
    # remove no seed worse than SC requirement (OPTIONAL from config)
    if not thresholds["require_no_seed_worse_than_self_consistency"]:
        feasibility_gates.pop(2)

    return feasibility_gates


def create_summary(config: dict, manifest: dict, input_dir: Path, ) -> dict:
    seed2rows, rows = load(config, manifest, input_dir)

    per_seed_combined = {
        seed: add_result(metric_values(seed_rows))
        for seed, seed_rows in seed2rows.items()
    }
    aggregate = add_result(metric_values(rows))

    per_question = {}
    for row in rows:
        per_question.setdefault(int(row["source_index"]), []).append(row)

    feasibility = evaluate_study(config, per_seed_combined, aggregate)

    signature = {
        "schema_version": 1,
        "study": config["study"]["name"],
        "status": "PASS" if all(requirement["passed"] for requirement in feasibility) else "FAIL",
        "examples_per_seed": config["dataset"]["pilot_examples"],
        "seeds": [seed for seed in config["inference"]["seeds"]],
        "aggregate": aggregate,
        "per_seed": {str(seed): value for seed, value in per_seed_combined.items()},
        "feasibility": feasibility
    }

    return signature


def _to_percent(value) -> str:
    return f"{value:.2%}"


def render_markdown(summary: dict) -> str:
    # prepare Markdown analysis
    aggregate = summary["aggregate"]
    lines = [
        "# OVM Feasibility Results\n",
        f"**Decision: {summary['status'].upper()}**\n",
        f"Pilot Study: {summary['examples_per_seed']} GSM8K questions x {len(summary['seeds'])} seeds\n"
        "## Average Accuracy\n\n",
        "| Method | Accuracy |",
        "|---:|---:|",
        f"| Greedy | {_to_percent(aggregate['greedy_accuracy'])} |",
        f"| Self-consistency (K=20) | {_to_percent(aggregate['self_consistency_accuracy'])} |",
        f"| ORM post-selection (K=20) | {_to_percent(aggregate['orm_accuracy'])} |",
        f"| OVM guided (K=20, b=10) | {_to_percent(aggregate['ovm_accuracy'])} |",
        "## Seed accuracy\n\n",
        "| Seed | Greedy | SC | ORM | OVM |",
        "|---:|---:|---:|---:|---:|",
    ]

    for seed, values in summary["per_seed"].items():
        lines.append(
            f"| {seed} "
            f"| {_to_percent(values[GREEDY_ACC])} "
            f"| {_to_percent(values[SELF_CONSISTENCY_ACC])} "
            f"| {_to_percent(values[ORM_ACC])} "
            f"| {_to_percent(values[OVM_ACC])} |"
        )

    lines.extend(
        [
            "## Feasibility gates\n\n",
            "| Requirement | Observed | Threshold | Result |",
            "|---|---:|---:|:---:|",
        ]
    )

    for requirement in summary["feasibility"]:
        lines.append(
            f"| `{requirement['name']}` | {requirement['value']:.4g} | {requirement['threshold']:.4g} | {'PASS' if requirement['passed'] else 'FAIL'} |")

    lines.extend(
        [
            "",
            "## Observability",
            "",
            (
                "- Vanilla sample correct fraction: "
                f"{_to_percent(aggregate[VANILLA_CORRECT_FRACTION])}"
            ),
            (
                "- OVM last-expansion correct fraction: "
                f"{_to_percent(aggregate[OVM_LAST_EXPANSION_CORRECT_FRACTION])}"
            ),
            (
                "- OVM final-beam correct fraction: "
                f"{_to_percent(aggregate[OVM_FINAL_BEAM_CORRECT_FRACTION])}"
            ),
            (
                "- OVM / sampling runtime ratio: "
                f"{aggregate[OVM_SAMPLING_RUNTIME_RATIO]:.2f}"
            ),
            (
                "- Maximum peak allocated CUDA memory: "
                f"{aggregate[MAX_PEAK_ALLOCATED_GIB]:.2f} GiB"
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True, help="Path to output JSON")
    parser.add_argument("--output-markdown", type=Path, required=True, help="Path to output Markdown")

    args = parser.parse_args()

    # prepare analysis
    summary = create_summary(
        read_json(args.config),
        read_json(args.pilot_manifest),
        args.input_dir,
    )

    # save to JSON
    write_json(args.output_json, summary)

    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    # save to Markdown
    with open(args.output_markdown, "w") as f:
        f.write(render_markdown(summary))

    print(f"Feasibility Study Check: {summary['status'].upper()}")


if __name__ == "__main__":
    main()