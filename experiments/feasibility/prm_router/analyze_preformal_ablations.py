"""CPU-only pre-formal ablations on saved ReasonEval/PathFinder outputs.

Development-only analysis for the already inspected expanded-200 training-pool
subset. This script does not run either verifier and does not access held-out
test data.

It compares:
1. current linear cost Ridge vs minimal nonlinear basis expansions;
2. current router features vs small information-preserving additions;
3. multinomial expected gain vs direct Ridge regression on g in {-1, 0, +1}.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from datasets import load_dataset
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cl_prm.data.records import align_verifier_records, correctness_arrays
from cl_prm.evaluation.cost import cost_features
from cl_prm.evaluation.routing import (
    GAIN_CLASSES,
    gain_probabilities,
    make_expected_gain_router,
    routed_accuracy,
    router_features,
    top_budget_indices,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = "hitsmy/PRMBench_Preview"
DEFAULT_REVISION = "5cc7683d0ae5797f84d7aeac0607966f277c39e1"
DEFAULT_SEEDS = [7, 42, 2026, 20260903, 20260915]
DEFAULT_BUDGETS = [20, 40]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reasoneval", type=Path, required=True)
    parser.add_argument("--pathfinder", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--dataset-revision", default=DEFAULT_REVISION)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "outputs/prm_router/robustness/preformal_ablations.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=REPO_ROOT / "outputs/prm_router/robustness/preformal_ablations.md",
    )
    return parser.parse_args()


def normalize_question(text: Any) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_groups(args: argparse.Namespace, example_ids: list[str]) -> np.ndarray:
    dataset = load_dataset(args.dataset, revision=args.dataset_revision)
    rows = dataset[args.dataset_split]
    question_by_id = {}
    for row in rows:
        original = (
            row.get("original_question")
            or row.get("question")
            or row.get("modified_question")
        )
        question_by_id[str(row["idx"])] = normalize_question(original)
    missing = sorted(set(example_ids) - set(question_by_id))
    if missing:
        raise ValueError("Missing dataset IDs: " + ", ".join(missing[:10]))
    return np.asarray([question_by_id[x] for x in example_ids], dtype=object)


def field(records: list[dict[str, Any]], name: str) -> np.ndarray:
    if any(name not in row for row in records):
        raise ValueError(f"Required field missing from saved outputs: {name}")
    return np.asarray([float(row[name]) for row in records], dtype=np.float64)


def ridge(alpha: float) -> Pipeline:
    return Pipeline(
        [("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))]
    )


def router_sets(
    records: list[dict[str, Any]],
    scores: np.ndarray,
    threshold: float,
) -> dict[str, np.ndarray]:
    f0 = router_features(records, scores, threshold)
    p_positive = field(records, "disprm_probability_positive")[:, None]
    input_tokens = field(records, "disprm_input_tokens")[:, None]
    f1 = np.c_[f0, p_positive]
    f2 = np.c_[f1, input_tokens]
    f3 = np.c_[f2, scores * f0[:, 3]]
    return {
        "F0_current_9": f0,
        "F1_plus_positive_probability": f1,
        "F2_plus_input_tokens": f2,
        "F3_plus_score_x_step_position": f3,
    }


def cost_sets(records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    c0 = cost_features(records)
    tokens = c0[:, 0]
    position = c0[:, 3]
    return {
        "C0_linear_current": c0,
        "C1_plus_input_tokens_squared": np.c_[c0, tokens**2],
        "C2_plus_tokens_squared_and_tokens_x_position": np.c_[
            c0, tokens**2, tokens * position
        ],
    }


def route_summary(
    ranking: np.ndarray,
    budgets: list[int],
    labels: np.ndarray,
    base_predictions: np.ndarray,
    second_predictions: np.ndarray,
    gains: np.ndarray,
) -> dict[str, Any]:
    result = {}
    for budget in budgets:
        selected = top_budget_indices(ranking, budget / 100.0)
        selected_gains = gains[selected]
        result[str(budget)] = {
            "accuracy": routed_accuracy(
                labels, base_predictions, second_predictions, selected
            ),
            "beneficial": int(np.sum(selected_gains == 1)),
            "harmful": int(np.sum(selected_gains == -1)),
        }
    return result


def safe_spearman(y: np.ndarray, pred: np.ndarray) -> float | None:
    value = float(spearmanr(y, pred).statistic)
    return value if np.isfinite(value) else None


def evaluate_seed(
    seed: int,
    folds: int,
    groups: np.ndarray,
    gains: np.ndarray,
    labels: np.ndarray,
    base_predictions: np.ndarray,
    second_predictions: np.ndarray,
    router_features_by_name: dict[str, np.ndarray],
    cost_features_by_name: dict[str, np.ndarray],
    runtime: np.ndarray,
    budgets: list[int],
    alpha: float,
) -> dict[str, Any]:
    splitter = StratifiedGroupKFold(
        n_splits=folds, shuffle=True, random_state=seed
    )
    reference = next(iter(router_features_by_name.values()))
    splits = list(splitter.split(reference, gains, groups))

    for fold_id, (train_idx, _) in enumerate(splits, start=1):
        missing = set(GAIN_CLASSES) - set(np.unique(gains[train_idx]).tolist())
        if missing:
            raise ValueError(
                f"Seed {seed} fold {fold_id} lacks gain classes {sorted(missing)}"
            )

    benefit = (gains == 1).astype(int)
    harm = (gains == -1).astype(int)

    cost_results = {}
    for name, x in cost_features_by_name.items():
        pred = np.zeros(len(x))
        for train_idx, test_idx in splits:
            model = ridge(alpha)
            model.fit(x[train_idx], runtime[train_idx])
            pred[test_idx] = np.maximum(model.predict(x[test_idx]), 0.0)
        cost_results[name] = {
            "mae": float(mean_absolute_error(runtime, pred)),
            "rmse": float(mean_squared_error(runtime, pred) ** 0.5),
            "spearman": safe_spearman(runtime, pred),
        }

    multinomial = {}
    direct = {}
    class_to_col = {label: i for i, label in enumerate(GAIN_CLASSES)}

    for name, x in router_features_by_name.items():
        probs = np.zeros((len(x), 3))
        direct_score = np.zeros(len(x))
        for fold_id, (train_idx, test_idx) in enumerate(splits, start=1):
            model = make_expected_gain_router(seed + fold_id)
            model.fit(x[train_idx], gains[train_idx])
            probs[test_idx] = gain_probabilities(model, x[test_idx])

            direct_model = ridge(alpha)
            direct_model.fit(x[train_idx], gains[train_idx].astype(float))
            direct_score[test_idx] = direct_model.predict(x[test_idx])

        p_harm = probs[:, class_to_col[-1]]
        p_benefit = probs[:, class_to_col[1]]
        expected_gain = p_benefit - p_harm

        multinomial[name] = {
            "benefit_ap": float(average_precision_score(benefit, p_benefit)),
            "harm_ap": float(average_precision_score(harm, p_harm)),
            "budgets": route_summary(
                expected_gain,
                budgets,
                labels,
                base_predictions,
                second_predictions,
                gains,
            ),
        }
        direct[name] = {
            "benefit_ap": float(average_precision_score(benefit, direct_score)),
            "harm_ap": float(average_precision_score(harm, -direct_score)),
            "budgets": route_summary(
                direct_score,
                budgets,
                labels,
                base_predictions,
                second_predictions,
                gains,
            ),
        }

    return {
        "seed": seed,
        "cost": cost_results,
        "multinomial": multinomial,
        "direct_utility_ridge": direct,
    }


def summarize(values: list[float | int | None]) -> dict[str, float | None]:
    clean = [float(x) for x in values if x is not None and np.isfinite(x)]
    if not clean:
        return {"mean": None, "sd": None}
    return {
        "mean": float(mean(clean)),
        "sd": float(pstdev(clean)) if len(clean) > 1 else 0.0,
    }


def aggregate(seed_results: list[dict[str, Any]], budgets: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {"cost": {}, "multinomial": {}, "direct_utility_ridge": {}}

    for name in seed_results[0]["cost"]:
        out["cost"][name] = {
            metric: summarize([row["cost"][name][metric] for row in seed_results])
            for metric in ("mae", "rmse", "spearman")
        }

    for estimator in ("multinomial", "direct_utility_ridge"):
        for feature_name in seed_results[0][estimator]:
            rows = [row[estimator][feature_name] for row in seed_results]
            summary = {
                "benefit_ap": summarize([row["benefit_ap"] for row in rows]),
                "harm_ap": summarize([row["harm_ap"] for row in rows]),
                "budgets": {},
            }
            for budget in budgets:
                key = str(budget)
                summary["budgets"][key] = {
                    metric: summarize(
                        [row["budgets"][key][metric] for row in rows]
                    )
                    for metric in ("accuracy", "beneficial", "harmful")
                }
            out[estimator][feature_name] = summary
    return out


def fmt(item: dict[str, float | None], digits: int = 4) -> str:
    if item["mean"] is None:
        return "n/a"
    return f"{item['mean']:.{digits}f} +/- {item['sd']:.{digits}f}"


def markdown(payload: dict[str, Any]) -> str:
    agg = payload["aggregate"]
    lines = [
        "# Pre-formal CPU Ablation Summary",
        "",
        "Development-only analysis on previously inspected expanded-200 training-pool data.",
        "This is not a formal validation or held-out test result.",
        "",
        f"Examples: {payload['examples']}",
        f"Original-question groups: {payload['groups']}",
        (
            "Beneficial / neutral / harmful: "
            f"{payload['counts']['beneficial']} / "
            f"{payload['counts']['neutral']} / "
            f"{payload['counts']['harmful']}"
        ),
        "",
        "## Cost predictor",
        "",
        "| Variant | MAE (s) | RMSE (s) | Spearman |",
        "|---|---:|---:|---:|",
    ]
    for name, row in agg["cost"].items():
        lines.append(
            f"| {name} | {fmt(row['mae'], 6)} | "
            f"{fmt(row['rmse'], 6)} | {fmt(row['spearman'])} |"
        )

    for estimator, title in (
        ("multinomial", "Multinomial expected gain"),
        ("direct_utility_ridge", "Direct utility Ridge"),
    ):
        lines += [
            "",
            f"## {title}",
            "",
            "| Features | Benefit AP | Harm AP | Acc@20 | B/H@20 | Acc@40 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, row in agg[estimator].items():
            b20 = row["budgets"]["20"]
            b40 = row["budgets"]["40"]
            lines.append(
                f"| {name} | {fmt(row['benefit_ap'])} | {fmt(row['harm_ap'])} | "
                f"{fmt(b20['accuracy'])} | "
                f"{fmt(b20['beneficial'], 2)} / {fmt(b20['harmful'], 2)} | "
                f"{fmt(b40['accuracy'])} |"
            )

    lines += [
        "",
        "## Interpretation rule",
        "",
        "Retain the simpler current specification unless a more complex variant shows",
        "a clear and consistent grouped-CV improvement. Harm AP is diagnostic only",
        "because harmful calls are rare; prioritize routing accuracy, seed stability,",
        "and beneficial/harmful selections at the fixed budgets.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1].")

    base, second = align_verifier_records(args.reasoneval, args.pathfinder)
    arrays = correctness_arrays(base, second, args.threshold)
    groups = load_groups(args, [str(row["example_id"]) for row in base])
    gains = arrays["gain_labels"]

    counts = {
        "beneficial": int(np.sum(gains == 1)),
        "neutral": int(np.sum(gains == 0)),
        "harmful": int(np.sum(gains == -1)),
    }
    if min(counts.values()) < args.folds:
        raise ValueError(f"Insufficient gain-class counts: {counts}")

    runtime = field(second, "pathfinder_runtime_seconds")
    rsets = router_sets(base, arrays["scores"], args.threshold)
    csets = cost_sets(base)

    seed_results = [
        evaluate_seed(
            seed,
            args.folds,
            groups,
            gains,
            arrays["labels"],
            arrays["base_predictions"],
            arrays["second_predictions"],
            rsets,
            csets,
            runtime,
            args.budgets,
            args.ridge_alpha,
        )
        for seed in args.seeds
    ]

    payload = {
        "scope": "development_only_preformal_ablation",
        "examples": len(base),
        "groups": len(set(groups.tolist())),
        "threshold": args.threshold,
        "counts": counts,
        "folds": args.folds,
        "seeds": args.seeds,
        "budgets": args.budgets,
        "ridge_alpha": args.ridge_alpha,
        "seed_results": seed_results,
        "aggregate": aggregate(seed_results, args.budgets),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(markdown(payload), encoding="utf-8")

    print("Pre-formal CPU ablation complete")
    print(f"Examples: {len(base)}")
    print(f"Groups:   {len(set(groups.tolist()))}")
    print(
        f"B/N/H:    {counts['beneficial']}/{counts['neutral']}/{counts['harmful']}"
    )
    print(f"JSON:     {args.output_json}")
    print(f"Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
