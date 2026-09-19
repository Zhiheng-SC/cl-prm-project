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
python experiments/formal/prm_router/prepare_formal_splits.py --config configs/experiments/prm_router_formal.json --output-dir data/prm_router/formal
```

This produces:

- `data/prm_router/formal/train.jsonl` with 600 examples;
- `data/prm_router/formal/validation.jsonl` with 200 examples;
- `data/prm_router/formal/test.jsonl` with 400 examples;
- `data/prm_router/formal/split_manifest.json`.

The generated `split_manifest.json` is a local audit record. Before formal
inference begins, run the guarded plan in
`experiments/formal/prm_router/run_formal_inference.py`; it validates the
generated split files against the tracked reference manifest at
`configs/experiments/prm_router_formal_split_manifest.json`, including
example counts and SHA256 hashes.

Do not commit generated JSONL files, model checkpoints, Hugging Face caches, or private access tokens.
