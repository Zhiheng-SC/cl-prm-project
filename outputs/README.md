# Outputs

Generated predictions, logs, metrics, figures, and checkpoints are stored locally under `outputs/` and are excluded from Git.

## Recommended Layout

```text
outputs/prm_router/
├── feasibility/               # Optional local feasibility artifacts
├── formal/
│   ├── reasoneval/
│   ├── pathfinder/
│   ├── genprm/
│   ├── router/
│   ├── metrics/
│   └── run_metadata/
└── robustness/                # Diagnostic analyses
```

Formal verifier outputs should be generated once, cached, and reused for lightweight router experiments.

## Team Sharing

The team should use a private shared artifact repository, such as a private Hugging Face Dataset repository, for formal JSONL predictions and run metadata. The GitHub repository stores only code, configuration, documentation, and compact reference manifests.

Do not share model checkpoints when they can be downloaded from the pinned upstream revision. Never commit or upload authentication tokens.
