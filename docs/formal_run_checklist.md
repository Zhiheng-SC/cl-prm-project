# Formal PRM Router Run Checklist

This checklist is the team-facing operational procedure for the frozen primary
ReasonEval -> Router -> PathFinder study.

It is intentionally separate from the scientific protocol. The protocol lives
in `configs/experiments/prm_router_formal.json` and
`experiments/formal/prm_router/README.md`.

## 0. Preconditions

Before any formal GPU inference:

- main branch is up to date;
- repository tests pass;
- formal splits reconstruct to 600 train / 200 validation / 400 test;
- grouped leakage audit passes;
- guarded development plan-only check passes;
- team agrees on one common GPU type and software stack for formal runtime;
- Hugging Face artifact access is confirmed for all team members.

Do not use the held-out test split for method selection.

## 1. Prepare the common GPU environment

From the repository root on the agreed RunPod machine:

```bash
git pull
bash scripts/bootstrap_runpod.sh
```

The bootstrap must finish with:

```text
RUNPOD ENVIRONMENT READY
No model inference was started.
```

Record:

- GPU model;
- Python version;
- PyTorch version;
- CUDA version;
- FlashAttention version;
- repository commit SHA.

## 2. Reconstruct and verify formal splits

```bash
python experiments/formal/prm_router/prepare_formal_splits.py \
  --config configs/experiments/prm_router_formal.json \
  --output-dir data/prm_router/formal
```

Expected:

- train: 600 examples, 300/300 labels;
- validation: 200 examples, 100/100 labels;
- test: 400 examples, 200/200 labels;
- leakage audit: passed.

Do not inspect test examples for model-development decisions.

## 3. Run the guarded development plan

```bash
python experiments/formal/prm_router/run_formal_inference.py \
  --phase development \
  --verifiers reasoneval pathfinder
```

Confirm:

- split hashes match the tracked reference manifest;
- ReasonEval revision is frozen;
- PathFinder revision is frozen;
- PathFinder uses BF16 + Flash Attention 2;
- output paths are under `outputs/prm_router/formal/`;
- final line says `Plan only; no model inference was started.`

## 4. Execute train + validation verifier inference

Only after the common environment has been agreed:

```bash
python experiments/formal/prm_router/run_formal_inference.py \
  --phase development \
  --verifiers reasoneval pathfinder \
  --execute
```

Expected primary outputs:

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

## 5. Verify outputs before analysis

For every verifier/split pair confirm:

- metadata status is complete;
- row count matches the split size;
- output SHA256 is recorded;
- input SHA256 matches the formal split;
- model name and revision match the frozen config;
- runtime metadata was collected on the common GPU;
- no interrupted/truncated JSONL remains.

Do not regenerate successful outputs unless a documented bug invalidates them.

## 6. Archive development artifacts

Upload train and validation artifacts to the private shared dataset repository:

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

## 7. Select the ReasonEval threshold

```bash
python experiments/formal/prm_router/select_reasoneval_threshold.py \
  --validation-reasoneval outputs/prm_router/formal/validation/reasoneval.jsonl
```

Record the selected threshold and archive the generated JSON.

## 8. Run development selection

```bash
python experiments/formal/prm_router/evaluate_development.py \
  --train-reasoneval outputs/prm_router/formal/train/reasoneval.jsonl \
  --train-pathfinder outputs/prm_router/formal/train/pathfinder.jsonl \
  --validation-reasoneval outputs/prm_router/formal/validation/reasoneval.jsonl \
  --validation-pathfinder outputs/prm_router/formal/validation/pathfinder.jsonl \
  --reason-eval-threshold <SELECTED_THRESHOLD>
```

This stage may select only the pre-specified validation choices:

- ReasonEval threshold;
- `lambda_h`;
- `mu`;
- secondary arbitration threshold/rule if reported.

Do not add new Router features or model families in response to validation
performance.

## 9. Freeze before held-out test

Record in the project notes / report:

- selected threshold;
- selected `lambda_h`;
- selected `mu`;
- frozen Router feature set;
- frozen cost-predictor feature set;
- primary 20% call budget;
- secondary budget curve;
- primary comparisons;
- bootstrap settings;
- common hardware/runtime protocol;
- secondary arbitration rule, if used.

Only after this freeze may held-out test inference be unlocked.

## 10. Held-out test

Use the guarded test commands documented in
`experiments/formal/prm_router/README.md`.

The test is evaluated once under the frozen protocol. Test results must not be
used to introduce new features, hyperparameters, thresholds, or baselines.

## Team handoff rule

Whoever runs a formal GPU stage should leave the next person with:

1. the exact git commit SHA;
2. the command that was run;
3. the output paths;
4. the metadata files;
5. the artifact upload location;
6. any failure/restart notes.
