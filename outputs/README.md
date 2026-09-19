# Outputs

Generated predictions, logs, metrics, figures, and checkpoints are stored locally under `outputs/` and are excluded from Git.

## Recommended Layout

```text
outputs/prm_router/
├── feasibility/               # Optional local feasibility artifacts
├── formal/
│   ├── train/
│   │   ├── reasoneval.jsonl
│   │   ├── reasoneval.metadata.json
│   │   ├── pathfinder.jsonl
│   │   └── pathfinder.metadata.json
│   ├── validation/
│   │   ├── reasoneval.jsonl
│   │   ├── reasoneval.metadata.json
│   │   ├── pathfinder.jsonl
│   │   ├── pathfinder.metadata.json
│   │   └── reasoneval_threshold.json
│   ├── test/
│   │   ├── reasoneval.jsonl
│   │   ├── reasoneval.metadata.json
│   │   ├── pathfinder.jsonl
│   │   ├── pathfinder.metadata.json
│   │   └── formal_test_results.json
│   └── development_selection.json
└── robustness/                # Diagnostic analyses
```

Formal verifier outputs should be generated once, cached, and reused for lightweight router experiments.

## Team Sharing

The team uses the private Hugging Face Dataset repository `cl-prm-team/cl-prm-artifacts` for generated feasibility/formal JSONL predictions, metadata, and downstream analysis artifacts. The GitHub repository stores only code, configuration, documentation, and compact reference manifests.

Do not share model checkpoints when they can be downloaded from the pinned upstream revision. Never commit or upload authentication tokens.

Compact machine-readable summaries (for example, `ovm-eval.json`) may remain in GitHub when they are small, human-auditable references rather than raw per-example outputs.

## Shared Formal Artifact Layout

The private Hugging Face artifact repository mirrors the formal stages:

```text
formal/
├── train/
├── validation/
└── test/
```

Upload train and validation artifacts after each completed development run and
hash verification. Upload held-out test artifacts only after the protocol has
been frozen and test inference has been explicitly unlocked. The operational
commands are documented in `experiments/formal/prm_router/README.md`.

