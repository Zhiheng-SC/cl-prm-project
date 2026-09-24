#!/usr/bin/env python3
"""Exploratory 256-token GenPRM token-to-runtime prediction.

Place beside genprm_cost_sweep.py. The runtime model sees only out-of-fold
predicted generated tokens during training, and predicted tokens at validation.
Question groups are kept intact at both cross-fitting levels. No test is read.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from genprm_cost_sweep import features, read_split


def make_model(family: str, rf_trees: int, numeric_features: int):
    if family == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise SystemExit("Install CatBoost: python -m pip install catboost") from exc
        model = CatBoostRegressor(
            iterations=300, learning_rate=0.04, depth=3, l2_leaf_reg=20.0,
            loss_function="MAE", random_seed=42, verbose=False, thread_count=4,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=rf_trees, min_samples_leaf=10, max_features=0.8,
            n_jobs=-1, random_state=42,
        )
    return Pipeline([
        ("features", ColumnTransformer([
            ("numeric", StandardScaler(), list(range(numeric_features))),
        ])),
        ("regressor", model),
    ])


def clip_tokens(prediction: np.ndarray, limit: int) -> np.ndarray:
    return np.clip(np.asarray(prediction, dtype=float), 0.0, float(limit))


def append_tokens(x: np.ndarray, predicted_tokens: np.ndarray) -> np.ndarray:
    # Existing nine numeric pre-call features, then a predicted token count.
    return np.column_stack((x[:, :9].astype(float), predicted_tokens))


def token_oof(x: np.ndarray, y_tokens: np.ndarray, groups: np.ndarray,
              model, folds: int, limit: int) -> np.ndarray:
    oof = np.empty(len(x), dtype=float)
    splitter = GroupKFold(folds)
    for train_idx, held_idx in splitter.split(x, y_tokens, groups):
        fitted = clone(model).fit(x[train_idx], y_tokens[train_idx])
        oof[held_idx] = clip_tokens(fitted.predict(x[held_idx]), limit)
    return oof


def fit_predict(x_train: np.ndarray, tokens_train: np.ndarray,
                runtime_train: np.ndarray, groups_train: np.ndarray,
                x_held: np.ndarray, token_model, runtime_model,
                inner_folds: int, limit: int) -> tuple[np.ndarray, np.ndarray]:
    train_token_oof = token_oof(x_train, tokens_train, groups_train,
                                token_model, inner_folds, limit)
    fitted_runtime = clone(runtime_model).fit(
        append_tokens(x_train, train_token_oof), runtime_train)
    if fitted_runtime.named_steps["features"].transform(
            append_tokens(x_train[:1], train_token_oof[:1])).shape[1] != 10:
        raise AssertionError("Runtime model must receive all 9 input features and predicted tokens")
    fitted_token = clone(token_model).fit(x_train, tokens_train)
    held_tokens = clip_tokens(fitted_token.predict(x_held), limit)
    held_runtime = np.maximum(
        fitted_runtime.predict(append_tokens(x_held, held_tokens)), 0.0)
    return held_tokens, held_runtime


def load_direct(path: Path, pairs: list[tuple[dict, dict]],
                runtime: np.ndarray) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    by_id = {row["example_id"]: row for row in rows}
    ids = [gen["example_id"] for gen, _ in pairs]
    if len(rows) != len(by_id) or set(ids) != set(by_id):
        raise ValueError("Direct predictions must match validation IDs exactly")
    result = []
    for i, example_id in enumerate(ids):
        row = by_id[example_id]
        if abs(float(row["actual"]) - runtime[i]) > 1e-5:
            raise ValueError(f"Direct prediction target is not runtime: {example_id}")
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifacts-dir", type=Path)
    source.add_argument("--artifacts-zip", type=Path)
    parser.add_argument("--direct-predictions", type=Path, required=True,
                        help="Runtime cost_tune_256/full/validation_predictions.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--family", choices=["catboost", "rf"], default="catboost")
    parser.add_argument("--rf-trees", type=int, default=200)
    parser.add_argument("--token-prediction-cap", type=int,
                        help="Optional cap for predicted TOTAL output tokens; default: train maximum")
    args = parser.parse_args()
    if (args.token_prediction_cap is not None and args.token_prediction_cap <= 0) or args.rf_trees <= 0:
        parser.error("--token-prediction-cap and --rf-trees must be positive")

    train = read_split(args, "train")
    validation = read_split(args, "validation")
    if (len(train), len(validation)) != (600, 200):
        raise ValueError("Expected 600 train and 200 validation examples")
    groups = np.asarray([gen["original_question_group"] for gen, _ in train])
    val_groups = {gen["original_question_group"] for gen, _ in validation}
    if set(groups) & val_groups:
        raise ValueError("Original questions overlap between train and validation")

    # Match genprm_cost_tune.py's direct model feature definition.
    x_train = features(train, threshold=0.5)
    x_val = features(validation, threshold=0.5)
    token_train = np.asarray([gen["genprm_generated_tokens"] for gen, _ in train], dtype=float)
    token_val = np.asarray([gen["genprm_generated_tokens"] for gen, _ in validation], dtype=float)
    runtime_train = np.asarray([gen["genprm_runtime_seconds"] for gen, _ in train], dtype=float)
    runtime_val = np.asarray([gen["genprm_runtime_seconds"] for gen, _ in validation], dtype=float)
    if not all(np.isfinite(a).all() for a in
               (token_train, token_val, runtime_train, runtime_val)):
        raise ValueError("Non-finite token count or runtime")
    token_cap = args.token_prediction_cap or int(np.max(token_train))
    if token_cap < np.max(token_train):
        raise ValueError("Prediction cap must cover all observed training tokens")

    token_model = make_model(args.family, args.rf_trees, numeric_features=9)
    runtime_model = make_model(args.family, args.rf_trees, numeric_features=10)
    # Outer CV evaluates the entire two-stage pipeline without fitting on
    # held-out question groups. Each outer-train fold cross-fits stage one.
    outer_tokens = np.empty(len(train), dtype=float)
    outer_runtime = np.empty(len(train), dtype=float)
    for fold, (fit_idx, held_idx) in enumerate(
            GroupKFold(4).split(x_train, runtime_train, groups), start=1):
        inner_groups = groups[fit_idx]
        predicted_tokens, predicted_runtime = fit_predict(
            x_train[fit_idx], token_train[fit_idx], runtime_train[fit_idx],
            inner_groups, x_train[held_idx], token_model, runtime_model,
            inner_folds=3, limit=token_cap)
        outer_tokens[held_idx] = predicted_tokens
        outer_runtime[held_idx] = predicted_runtime
        print(f"Outer group fold {fold}/4 complete", flush=True)

    val_tokens, val_runtime = fit_predict(
        x_train, token_train, runtime_train, groups, x_val,
        token_model, runtime_model, inner_folds=4, limit=token_cap)
    direct_rows = load_direct(args.direct_predictions, validation, runtime_val)
    capped = np.asarray([not gen["genprm_analysis_complete"] for gen, _ in validation])
    merged = []
    for i, ((gen, _), direct) in enumerate(zip(validation, direct_rows)):
        merged.append({
            "example_id": gen["example_id"],
            "actual": float(runtime_val[i]),
            "analysis_capped": bool(capped[i]),
            "actual_generated_tokens": float(token_val[i]),
            "predicted_generated_tokens": float(val_tokens[i]),
            "two_stage": float(val_runtime[i]),
            **{name: float(direct[name]) for name in ("selected", "mean", "ridge_re")},
        })
    metric_rows = [
        {"model": "two_stage", "validation_mae_seconds": mean_absolute_error(runtime_val, val_runtime),
         "validation_r2": r2_score(runtime_val, val_runtime),
         "validation_capped_mae_seconds": mean_absolute_error(runtime_val[capped], val_runtime[capped])},
    ]
    for name in ("selected", "mean", "ridge_re"):
        direct_pred = np.asarray([row[name] for row in merged], dtype=float)
        metric_rows.append({
            "model": name,
            "validation_mae_seconds": mean_absolute_error(runtime_val, direct_pred),
            "validation_r2": r2_score(runtime_val, direct_pred),
            "validation_capped_mae_seconds": mean_absolute_error(runtime_val[capped], direct_pred[capped]),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "validation_predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.output_dir / "validation_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    info = {
        "scope": "exploratory_256_analysis_only", "family": args.family,
        "rf_trees": args.rf_trees if args.family == "rf" else None,
        "total_output_token_prediction_cap_from_train": token_cap,
        "stage_one_train_target": "genprm_generated_tokens",
        "stage_two_train_feature": "stage_one_question_group_out_of_fold_prediction",
        "train_outer_group_cv_token_mae": float(mean_absolute_error(token_train, outer_tokens)),
        "train_outer_group_cv_runtime_mae_seconds": float(mean_absolute_error(runtime_train, outer_runtime)),
        "validation_token_mae": float(mean_absolute_error(token_val, val_tokens)),
        "validation_token_r2": float(r2_score(token_val, val_tokens)),
        "validation_capped_examples": int(capped.sum()),
        "direct_predictions_file": str(args.direct_predictions),
        "note": "Validation was previously inspected; results are exploratory. "
                "Realized tokens are used as stage-one training targets and for diagnostics only; "
                "the runtime prediction never sees realized validation tokens. No test data used.",
    }
    (args.output_dir / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Tokens: train group-CV MAE={info['train_outer_group_cv_token_mae']:.2f}, "
          f"validation MAE={info['validation_token_mae']:.2f}, "
          f"validation R2={info['validation_token_r2']:.3f}")
    print(f"Two-stage runtime: train group-CV MAE="
          f"{info['train_outer_group_cv_runtime_mae_seconds']:.3f}s")
    for row in metric_rows:
        print(f"{row['model']:10s} validation MAE={row['validation_mae_seconds']:.3f}s "
              f"R2={row['validation_r2']:.3f} "
              f"capped MAE={row['validation_capped_mae_seconds']:.3f}s")
    print(f"Saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
