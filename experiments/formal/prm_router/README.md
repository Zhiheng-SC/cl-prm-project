# PRM Router Formal Evaluation

This directory contains the guarded command-line entry points for the formal
PRM Router study. Reusable implementation lives under `src/cl_prm/`; completed
exploratory and feasibility workflows remain under
`experiments/feasibility/prm_router/`.

## Entry points

- `prepare_formal_splits.py`: reconstruct the pinned 600/200/400
  train/validation/test split and verify the tracked manifest.
- `run_formal_inference.py`: plan or execute ReasonEval/PathFinder inference.
  Development mode runs train + validation; test execution is explicitly
  locked until the protocol is frozen.
- `select_reasoneval_threshold.py`: select the ReasonEval threshold using the
  validation split only.
- `evaluate_development.py`: fit train-only routers/runtime predictor and use
  validation only for the pre-specified utility-weight selection.
- `evaluate_test.py`: evaluate the frozen protocol once on held-out test data.

The verifier-specific inference runners are currently retained under
`experiments/feasibility/prm_router/` for compatibility with the completed
feasibility workflow. The formal orchestrator calls those runners, while shared
resume/metadata logic is provided by `cl_prm.utils.inference_io`.

## Development sequence

From the repository root, after installing the package with
`python -m pip install -e . --no-deps`:

```bash
python experiments/formal/prm_router/prepare_formal_splits.py \
  --config configs/experiments/prm_router_formal.json \
  --output-dir data/prm_router/formal

python experiments/formal/prm_router/run_formal_inference.py \
  --phase development \
  --verifiers reasoneval pathfinder
```

The inference command is plan-only unless `--execute` is supplied.

After train and validation inference:

1. select the ReasonEval threshold on validation;
2. run `evaluate_development.py`;
3. inspect and freeze all validation-selected choices;
4. only then unlock test inference and run `evaluate_test.py`.

Do not change the method in response to held-out test results.
