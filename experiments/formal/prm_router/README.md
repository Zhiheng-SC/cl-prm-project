# PRM Router Formal Evaluation

This directory is the operational entry point for the frozen formal PRM Router
study. Reusable implementation lives under `src/cl_prm/`; completed
exploratory and feasibility workflows remain under
`experiments/feasibility/prm_router/`.

The formal JSON configuration is treated as a protocol contract. Formal entry
points validate the configuration before running so that unsupported changes
to grouping, features, budgets, runtime policy, or test locking cannot be
silently ignored.

## Entry Points

- `prepare_formal_splits.py`: reconstruct the pinned 600/200/400
  train/validation/test splits and write a local split manifest.
- `run_formal_inference.py`: validate split hashes against the tracked
  reference manifest and plan or execute verifier inference.
- `select_reasoneval_threshold.py`: select the ReasonEval threshold using the
  validation split only.
- `evaluate_development.py`: fit train-only routers/runtime predictor and use
  validation only for the pre-specified utility-weight selection.
- `evaluate_test.py`: evaluate the frozen protocol once on held-out test data.

The verifier-specific ReasonEval, PathFinder, and GenPRM runners remain under
`experiments/feasibility/prm_router/` because those implementations were
already validated during feasibility. The formal orchestrator calls those
runners, while shared routing, data, cost, bootstrap, and metadata logic lives
under `src/cl_prm/`.

## Environment

Run all commands from the repository root.

Install the project dependencies first, then the editable package:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --no-build-isolation
```

On the verified RunPod environment, use `bash scripts/bootstrap_runpod.sh`
instead of `requirements.txt`; see `docs/runpod.md`.

Formal ReasonEval and PathFinder runtime measurements must be collected on the
same GPU type and software stack.

## 1. Reconstruct and Verify the Formal Splits

```bash
python experiments/formal/prm_router/prepare_formal_splits.py \
  --config configs/experiments/prm_router_formal.json \
  --output-dir data/prm_router/formal
```

This writes:

```text
data/prm_router/formal/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── split_manifest.json
```

The generated manifest is a local audit record. The guarded inference planner
uses `configs/experiments/prm_router_formal_split_manifest.json` as the
tracked reference and rejects split files whose SHA256 or example counts do
not match.

## 2. Preview Development Inference

Before spending GPU time, run the plan-only command:

```bash
python experiments/formal/prm_router/run_formal_inference.py \
  --phase development \
  --verifiers reasoneval pathfinder
```

It must end with:

```text
Plan only; no model inference was started.
```

Check the selected models, revisions, split paths, output paths, and common
hardware before adding `--execute`.

## 3. Run Train + Validation Inference

Only after the team has agreed on the common GPU/environment:

```bash
python experiments/formal/prm_router/run_formal_inference.py \
  --phase development \
  --verifiers reasoneval pathfinder \
  --execute
```

Expected primary artifacts:

```text
outputs/prm_router/formal/
├── train/
│   ├── reasoneval.jsonl
│   ├── reasoneval.metadata.json
│   ├── pathfinder.jsonl
│   └── pathfinder.metadata.json
└── validation/
    ├── reasoneval.jsonl
    ├── reasoneval.metadata.json
    ├── pathfinder.jsonl
    └── pathfinder.metadata.json
```

After a completed run, verify metadata status, row counts, and output hashes,
then archive the generated artifacts in the private shared repository:

```bash
hf upload cl-prm-team/cl-prm-artifacts \
  outputs/prm_router/formal/train \
  formal/train \
  --repo-type dataset

hf upload cl-prm-team/cl-prm-artifacts \
  outputs/prm_router/formal/validation \
  formal/validation \
  --repo-type dataset
```

## 4. Select the ReasonEval Threshold

```bash
python experiments/formal/prm_router/select_reasoneval_threshold.py \
  --validation-reasoneval outputs/prm_router/formal/validation/reasoneval.jsonl
```

This writes:

```text
outputs/prm_router/formal/validation/reasoneval_threshold.json
```

Record the numeric `selected_threshold` value for the next commands.

## 5. Run Development Selection

Replace `<SELECTED_THRESHOLD>` with the value from the previous step:

```bash
python experiments/formal/prm_router/evaluate_development.py \
  --train-reasoneval outputs/prm_router/formal/train/reasoneval.jsonl \
  --train-pathfinder outputs/prm_router/formal/train/pathfinder.jsonl \
  --validation-reasoneval outputs/prm_router/formal/validation/reasoneval.jsonl \
  --validation-pathfinder outputs/prm_router/formal/validation/pathfinder.jsonl \
  --reason-eval-threshold <SELECTED_THRESHOLD>
```

This writes:

```text
outputs/prm_router/formal/development_selection.json
```

The development stage fits the train-only routing and runtime models and uses
validation only to select the pre-specified `lambda_h` and `mu` values.

Archive the two development-selection artifacts after checking them:

```bash
hf upload cl-prm-team/cl-prm-artifacts \
  outputs/prm_router/formal/validation/reasoneval_threshold.json \
  evaluation/router/reasoneval_threshold.json \
  --repo-type dataset

hf upload cl-prm-team/cl-prm-artifacts \
  outputs/prm_router/formal/development_selection.json \
  evaluation/router/development_selection.json \
  --repo-type dataset
```

## 6. Freeze the Protocol

Before touching the held-out test split, record and freeze:

- the selected ReasonEval threshold;
- the selected `lambda_h` and `mu`;
- the pre-call feature set and cost feature set;
- the primary 20% call budget and secondary budget curve;
- the primary comparisons and bootstrap settings;
- any secondary post-call arbitration rule that will be reported;
- the common hardware/runtime protocol.

Do not change the method in response to held-out test results.

## 7. Preview and Run Held-Out Test Inference

First preview the locked test plan:

```bash
python experiments/formal/prm_router/run_formal_inference.py \
  --phase test \
  --verifiers reasoneval pathfinder \
  --reason-eval-threshold <SELECTED_THRESHOLD> \
  --confirm-test-protocol-frozen
```

Only after checking that plan, execute it:

```bash
python experiments/formal/prm_router/run_formal_inference.py \
  --phase test \
  --verifiers reasoneval pathfinder \
  --reason-eval-threshold <SELECTED_THRESHOLD> \
  --confirm-test-protocol-frozen \
  --execute
```

Archive the completed test artifacts only after the protocol is frozen:

```bash
hf upload cl-prm-team/cl-prm-artifacts \
  outputs/prm_router/formal/test \
  formal/test \
  --repo-type dataset
```

## 8. Evaluate the Frozen Test Protocol

```bash
python experiments/formal/prm_router/evaluate_test.py \
  --train-reasoneval outputs/prm_router/formal/train/reasoneval.jsonl \
  --train-pathfinder outputs/prm_router/formal/train/pathfinder.jsonl \
  --test-reasoneval outputs/prm_router/formal/test/reasoneval.jsonl \
  --test-pathfinder outputs/prm_router/formal/test/pathfinder.jsonl \
  --confirm-protocol-frozen
```

The default output is:

```text
outputs/prm_router/formal/test/formal_test_results.json
```

The held-out result must be interpreted under the protocol that was frozen
before test inference. Test performance must not be used to select new
features, thresholds, utility weights, budgets, or comparison methods.
