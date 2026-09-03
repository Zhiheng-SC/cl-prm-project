# Data

Datasets generated for this project are stored locally under `data/` and are excluded from Git.

## Reconstruct the Feasibility Pilot

From the repository root:

```bash
python experiments/feasibility/prm_router/prepare_subset.py --n-correct 50 --n-error 50 --seed 2026 --output data/prm_router/feasibility_100.jsonl
```

## Reconstruct the Formal Splits

The formal configuration pins the PRMBench revision and excludes all original-question groups represented in the feasibility pilot.

```bash
python experiments/feasibility/prm_router/prepare_formal_splits.py --config configs/experiments/prm_router_formal.json --output-dir data/prm_router/formal
```

This produces:

- `data/prm_router/formal/train.jsonl` with 600 examples;
- `data/prm_router/formal/validation.jsonl` with 200 examples;
- `data/prm_router/formal/test.jsonl` with 400 examples;
- `data/prm_router/formal/split_manifest.json`.

Compare the generated manifest with `configs/experiments/prm_router_formal_split_manifest.json`. Dataset fingerprint, counts, and all three SHA256 hashes must match before formal inference begins.

Do not commit generated JSONL files, model checkpoints, Hugging Face caches, or private access tokens.
