#!/usr/bin/env python3
"""Tune 256-token GenPRM cost models using train-only grouped CV.

Place beside genprm_cost_sweep.py. Requires numpy, scipy, scikit-learn;
CatBoost and XGBoost are optional. Validation is evaluated only for the
train-CV-selected model and two prespecified baselines. Formal test is unused.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from genprm_cost_sweep import features, read_split


def versions() -> dict[str, str | None]:
    result = {}
    for name in ("numpy", "scipy", "scikit-learn", "catboost", "xgboost"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def numeric(model):
    return Pipeline([("features", ColumnTransformer([
        ("numeric", StandardScaler(), list(range(9))),
    ])), ("regressor", model)])


def ridge_text(alpha: float):
    return Pipeline([("features", ColumnTransformer([
        ("numeric", StandardScaler(), list(range(9))),
        ("text", TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                 min_df=3, max_features=10000, sublinear_tf=True), 10),
    ], sparse_threshold=1.0)), ("regressor", Ridge(alpha=alpha, solver="lsqr"))])


def ridge_re():
    return Pipeline([("features", ColumnTransformer([
        ("numeric", StandardScaler(), list(range(9))),
    ])), ("regressor", Ridge(alpha=10.0, solver="lsqr"))])


def candidates(selected: set[str], quick: bool):
    trials = []
    if "rf" in selected:
        for leaf in ((5, 10) if quick else (5, 10, 20)):
            for fraction in ((0.8,) if quick else (0.6, 1.0)):
                params = {"min_samples_leaf": leaf, "max_features": fraction}
                model = RandomForestRegressor(n_estimators=200, n_jobs=-1,
                                              random_state=42, **params)
                trials.append(("rf", params, numeric(model)))
    if "histgb" in selected:
        for leaf in ((10, 25) if quick else (10, 20, 40)):
            for nodes in ((7,) if quick else (5, 10)):
                params = {"min_samples_leaf": leaf, "max_leaf_nodes": nodes}
                model = HistGradientBoostingRegressor(max_iter=150,
                    learning_rate=0.05, l2_regularization=10.0,
                    random_state=42, **params)
                trials.append(("histgb", params, numeric(model)))
    if "ridge_text" in selected:
        for alpha in ((1.0, 10.0) if quick else (1.0, 3.0, 10.0, 30.0)):
            trials.append(("ridge_text", {"alpha": alpha}, ridge_text(alpha)))
    if "catboost" in selected:
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise SystemExit("CatBoost requested: python -m pip install catboost") from exc
        for depth in ((4,) if quick else (3, 4, 6)):
            for reg in ((10.0,) if quick else (5.0, 20.0)):
                params = {"depth": depth, "l2_leaf_reg": reg}
                model = CatBoostRegressor(iterations=300, learning_rate=0.04,
                    loss_function="MAE", random_seed=42, verbose=False,
                    thread_count=4, **params)
                trials.append(("catboost", params, numeric(model)))
    if "xgboost" in selected:
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise SystemExit("XGBoost requested: python -m pip install xgboost") from exc
        for depth in ((2,) if quick else (2, 3)):
            for child in ((10.0,) if quick else (5.0, 15.0)):
                params = {"max_depth": depth, "min_child_weight": child}
                model = XGBRegressor(n_estimators=300, learning_rate=0.04,
                    subsample=0.9, colsample_bytree=0.9, reg_lambda=10.0,
                    objective="reg:squarederror", tree_method="hist",
                    n_jobs=4, random_state=42, **params)
                trials.append(("xgboost", params, numeric(model)))
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifacts-dir", type=Path)
    source.add_argument("--artifacts-zip", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=["rf", "histgb", "ridge_text"],
                        choices=["rf", "histgb", "ridge_text", "catboost", "xgboost"])
    parser.add_argument("--quick", action="store_true", help="Smaller search grid")
    parser.add_argument("--target", choices=["runtime", "tokens"], default="runtime")
    args = parser.parse_args()

    train = read_split(args, "train")
    validation = read_split(args, "validation")
    groups = np.asarray([g["original_question_group"] for g, _ in train])
    val_groups = {g["original_question_group"] for g, _ in validation}
    if set(groups) & val_groups:
        raise ValueError("Question-group overlap between train and validation")
    xtr = features(train, threshold=0.5)
    xval = features(validation, threshold=0.5)
    target = ("genprm_runtime_seconds" if args.target == "runtime"
              else "genprm_generated_tokens")
    ytr = np.asarray([g[target] for g, _ in train], dtype=float)
    yval = np.asarray([g[target] for g, _ in validation], dtype=float)
    trials = candidates(set(args.families), args.quick)
    if not trials:
        raise ValueError("No model families selected")

    results = []
    for family, params, prototype in trials:
        oof = np.empty(len(ytr), dtype=float)
        for fit_idx, held_idx in GroupKFold(4).split(xtr, ytr, groups):
            model = clone(prototype)
            model.fit(xtr[fit_idx], ytr[fit_idx])
            oof[held_idx] = model.predict(xtr[held_idx])
        result = {
            "family": family, "parameters": json.dumps(params, sort_keys=True),
            "train_group_cv_mae": float(mean_absolute_error(ytr, oof)),
            "train_group_cv_r2": float(r2_score(ytr, oof)),
        }
        results.append((result, prototype))
        print(f"{family:12s} {result['parameters']:48s} CV MAE={result['train_group_cv_mae']:.3f}", flush=True)

    results.sort(key=lambda item: item[0]["train_group_cv_mae"])
    best, prototype = results[0]
    eval_models = {"selected": prototype, "mean": DummyRegressor(strategy="mean"),
                   "ridge_re": ridge_re()}
    capped = np.asarray([not g["genprm_analysis_complete"] for g, _ in validation])
    validation_scores = []
    predictions = [{"example_id": g["example_id"], "actual": float(yval[i]),
                    "analysis_capped": bool(capped[i])}
                   for i, (g, _) in enumerate(validation)]
    for name, base_model in eval_models.items():
        model = clone(base_model).fit(xtr, ytr)
        pred = np.asarray(model.predict(xval), dtype=float)
        validation_scores.append({
            "model": name, "validation_mae": float(mean_absolute_error(yval, pred)),
            "validation_r2": float(r2_score(yval, pred)),
            "validation_capped_mae": float(mean_absolute_error(yval[capped], pred[capped])),
        })
        for row, value in zip(predictions, pred):
            row[name] = float(value)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "train_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0][0]))
        writer.writeheader()
        writer.writerows(item[0] for item in results)
    with (args.output_dir / "validation_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(validation_scores[0]))
        writer.writeheader()
        writer.writerows(validation_scores)
    with (args.output_dir / "validation_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "target": target, "selection_rule": "minimum 4-fold question-group CV MAE on train",
        "best_model": best, "families": args.families, "quick": args.quick,
        "packages": versions(), "train_n": len(train), "validation_n": len(validation),
        "note": "Validation inspected only for train-selected candidate and fixed baselines. "
                "Previously viewed development validation remains exploratory; formal test untouched.",
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Selected by train CV: {best['family']} {best['parameters']}")
    for row in validation_scores:
        print(f"{row['model']:12s} validation MAE={row['validation_mae']:.3f} "
              f"R2={row['validation_r2']:.3f} capped MAE={row['validation_capped_mae']:.3f}")
    print(f"Saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
