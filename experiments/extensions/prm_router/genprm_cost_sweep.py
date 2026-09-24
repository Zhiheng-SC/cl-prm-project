#!/usr/bin/env python3
"""Exploratory, pre-call GenPRM cost prediction on the 600/200 development split.

Requires: numpy, scipy, scikit-learn. No GPU or model checkpoints are needed.
The 400-example formal test split is deliberately unsupported.
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def read_jsonl(data: bytes, source: str) -> list[dict]:
    try:
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid JSONL in {source}: {exc}") from exc
    if not rows:
        raise ValueError(f"Empty JSONL: {source}")
    return rows


def read_split(args: argparse.Namespace, split: str) -> list[tuple[dict, dict]]:
    if args.artifacts_zip:
        base = f"cl-prm-artifacts/formal/{split}/"
        with zipfile.ZipFile(args.artifacts_zip) as archive:
            gen = read_jsonl(archive.read(base + "genprm.jsonl"), base + "genprm.jsonl")
            re = read_jsonl(archive.read(base + "reasoneval.jsonl"), base + "reasoneval.jsonl")
    else:
        root = args.artifacts_dir / "formal" / split
        gen = read_jsonl((root / "genprm.jsonl").read_bytes(), str(root / "genprm.jsonl"))
        re = read_jsonl((root / "reasoneval.jsonl").read_bytes(), str(root / "reasoneval.jsonl"))

    ids = [r["example_id"] for r in gen]
    re_ids = [r["example_id"] for r in re]
    if len(ids) != len(set(ids)) or len(re_ids) != len(set(re_ids)):
        raise ValueError(f"Duplicate example_id in {split}")
    if set(ids) != set(re_ids):
        raise ValueError(f"RE/GenPRM IDs do not match in {split}")
    re_by_id = {r["example_id"]: r for r in re}
    pairs = [(g, re_by_id[g["example_id"]]) for g in gen]
    for g, r in pairs:
        if g["model_input"] != r["model_input"] or g["label"] != r["label"]:
            raise ValueError(f"RE/GenPRM input or label mismatch: {g['example_id']}")
        if g.get("formal_split") != split or r.get("formal_split") != split:
            raise ValueError(f"Unexpected split: {g['example_id']}")
    return pairs


def features(pairs: list[tuple[dict, dict]], threshold: float) -> np.ndarray:
    rows = []
    for g, r in pairs:
        probs = np.array(
            [r["disprm_probability_negative"], r["disprm_probability_neutral"],
             r["disprm_probability_positive"]], dtype=float
        )
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-15, 1.0))) / np.log(3.0)
        score = float(r["disprm_score"])
        rows.append([
            g["genprm_input_tokens"], g["current_step"], g["total_steps"],
            g["step_position"], len(g["question"]), len(g["current_step_text"]),
            score, abs(score - threshold), entropy,
            g["question"] + " " + g["current_step_text"], g["model_input"],
        ])
    return np.array(rows, dtype=object)


def estimator(numeric_columns: list[int], text: tuple[str, int] | None = None):
    transforms = [("numeric", StandardScaler(), numeric_columns)]
    if text:
        kind, column = text
        if kind == "char":
            vectorizer = TfidfVectorizer(
                analyzer="char", ngram_range=(3, 5), min_df=3,
                max_features=10000, sublinear_tf=True,
            )
        else:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2), min_df=2, max_features=6000,
                sublinear_tf=True,
            )
        transforms.append(("text", vectorizer, column))
    return ColumnTransformer(transforms, sparse_threshold=1.0)


def candidate_models(preset: str) -> dict[str, object]:
    def ridge(columns, text=None):
        return Pipeline([
            ("features", estimator(columns, text)),
            ("regressor", Ridge(alpha=10.0, solver="lsqr")),
        ])

    def tree(model):
        return Pipeline([
            ("features", estimator(list(range(9)))),
            ("regressor", model),
        ])

    models = {
        "mean": DummyRegressor(strategy="mean"),
        "ridge_basic": ridge(list(range(6))),
        "ridge_re": ridge(list(range(9))),
        "random_forest": tree(RandomForestRegressor(
            n_estimators=200, min_samples_leaf=10, max_features=0.8,
            n_jobs=-1, random_state=42,
        )),
        "extra_trees": tree(ExtraTreesRegressor(
            n_estimators=200, min_samples_leaf=10, max_features=0.8,
            n_jobs=-1, random_state=42,
        )),
        "hist_gradient_boosting": tree(HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=7, min_samples_leaf=20,
            l2_regularization=10.0, random_state=42,
        )),
        "ridge_char_step": ridge(list(range(9)), ("char", 9)),
    }
    if preset == "full":
        models.update({
            "gradient_boosting": tree(GradientBoostingRegressor(
                n_estimators=100, max_depth=2, learning_rate=0.05,
                min_samples_leaf=15, random_state=42,
            )),
            "ridge_word_step": ridge(list(range(9)), ("word", 9)),
            "ridge_char_prefix": ridge(list(range(9)), ("char", 10)),
            "ridge_word_prefix": ridge(list(range(9)), ("word", 10)),
        })
    return models


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    rank_correlation = (
        float("nan") if np.ptp(predicted) < 1e-12
        else float(spearmanr(actual, predicted).statistic)
    )
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
        "spearman": rank_correlation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifacts-zip", type=Path, help="Downloaded cl-prm-artifacts.zip")
    source.add_argument("--artifacts-dir", type=Path, help="Folder containing formal/train and formal/validation")
    parser.add_argument("--output-dir", type=Path, default=Path("cost_sweep_output"))
    parser.add_argument("--preset", choices=("quick", "full"), default="quick")
    parser.add_argument("--target", choices=("runtime", "tokens"), default="runtime")
    parser.add_argument("--re-threshold", type=float, default=0.5,
                        help="Fixed reference threshold for distance; not fitted on validation")
    args = parser.parse_args()
    if not (0.0 <= args.re_threshold <= 1.0):
        parser.error("--re-threshold must be between 0 and 1")

    train = read_split(args, "train")
    validation = read_split(args, "validation")
    train_groups = {g["original_question_group"] for g, _ in train}
    val_groups = {g["original_question_group"] for g, _ in validation}
    if train_groups & val_groups:
        raise ValueError("Original questions overlap across train and validation")
    if len(train) != 600 or len(validation) != 200:
        raise ValueError(f"Expected 600/200 examples; found {len(train)}/{len(validation)}")

    x_train = features(train, args.re_threshold)
    x_val = features(validation, args.re_threshold)
    target_field = "genprm_runtime_seconds" if args.target == "runtime" else "genprm_generated_tokens"
    y_train = np.array([g[target_field] for g, _ in train], dtype=float)
    y_val = np.array([g[target_field] for g, _ in validation], dtype=float)
    groups = np.array([g["original_question_group"] for g, _ in train])
    cap_val = np.array([not g["genprm_analysis_complete"] for g, _ in validation])
    if not (np.isfinite(y_train).all() and np.isfinite(y_val).all()):
        raise ValueError("Cost target contains non-finite values")

    candidates = candidate_models(args.preset)
    summary = []
    predictions = [
        {"example_id": g["example_id"], "actual": float(y_val[i]),
         "analysis_capped": bool(cap_val[i])}
        for i, (g, _) in enumerate(validation)
    ]
    for name, prototype in candidates.items():
        oof = np.empty(len(train), dtype=float)
        for train_idx, held_idx in GroupKFold(n_splits=4).split(x_train, y_train, groups):
            model = clone(prototype)
            model.fit(x_train[train_idx], y_train[train_idx])
            oof[held_idx] = model.predict(x_train[held_idx])
        model = clone(prototype)
        model.fit(x_train, y_train)
        pred = np.asarray(model.predict(x_val), dtype=float)
        for row, estimate in zip(predictions, pred):
            row[name] = float(estimate)
        row = {
            "model": name,
            **{f"train_group_cv_{key}": value for key, value in metrics(y_train, oof).items()},
            **{f"validation_{key}": value for key, value in metrics(y_val, pred).items()},
            "validation_complete_mae": float(mean_absolute_error(y_val[~cap_val], pred[~cap_val])),
            "validation_capped_mae": float(mean_absolute_error(y_val[cap_val], pred[cap_val])),
        }
        summary.append(row)

    summary.sort(key=lambda row: row["train_group_cv_mae"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    with (args.output_dir / "validation_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    info = {
        "target": target_field, "preset": args.preset, "re_distance_threshold": args.re_threshold,
        "train_examples": len(train), "validation_examples": len(validation),
        "validation_capped_examples": int(cap_val.sum()),
        "selection_rule": "lowest grouped-CV MAE on train only",
        "selected_by_train_cv": summary[0]["model"],
        "note": "200-example validation is exploratory; earlier analyses have inspected it. No test data used.",
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Target: {target_field}; train {len(train)}, validation {len(validation)}")
    print(f"Selected by train grouped CV: {summary[0]['model']}")
    for row in summary:
        print(f"{row['model']:24s} CV MAE={row['train_group_cv_mae']:.3f}  "
              f"validation MAE={row['validation_mae']:.3f}  "
              f"R2={row['validation_r2']:.3f}  "
              f"capped MAE={row['validation_capped_mae']:.3f}")
    print(f"Saved: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
