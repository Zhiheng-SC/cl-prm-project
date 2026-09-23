# CL-PRM Project

This repository contains code, documentation, and experiments for a course project on process reward modeling in large language models.

## Project Status

The primary ReasonEval -> PathFinder selective-verification study is complete.

The frozen primary system is:

`ReasonEval -> utility router -> optional PathFinder`

The formal experiment used original-question-grouped 600/200/400 train, validation, and held-out test splits. Thresholds and routing hyperparameters were selected using development data before the test set was opened.

Held-out results on 400 test examples:

| System | Accuracy |
|---|---:|
| ReasonEval only | 71.75% |
| PathFinder only | 79.00% |
| Expected-gain routing @ 20% PathFinder calls | 81.75% |
| Cost-aware routing @ 20% PathFinder calls | 81.75% |

The pairwise ReasonEval--PathFinder oracle reaches 92.25%. The primary protocol is now frozen and closed to further test-based tuning. Current work focuses on final descriptive diagnostics, visualization, reproducibility checks, and report writing.

GenPRM remains an optional post-primary verifier-pair extension and does not affect the completed ReasonEval--PathFinder primary result.

See:

- [PRM Router proposal and protocol history](docs/proposals/prm_router.md)
- [Formal PRM Router runbook](experiments/formal/prm_router/README.md)
- [PRM Router feasibility study](experiments/feasibility/prm_router/README.md)
- [Expanded 200-example feasibility results](docs/results/prm_router_expanded_feasibility_200.md)

## Project Resources

- **Code, protocol, and documentation:** this GitHub repository.
- **Generated experiment artifacts:** [`cl-prm-team/cl-prm-artifacts`](https://huggingface.co/datasets/cl-prm-team/cl-prm-artifacts) (private during development).
- **Internal writing workspace:** shared Overleaf project (team access only; no editable share link is stored in Git).

## Primary Study

The project asks whether pair-specific correction utility can allocate limited process-verification compute more effectively than confidence- or difficulty-based escalation.

The final primary study combines:

- ReasonEval as the first-stage verifier;
- a lightweight pre-call utility router;
- PathFinder-PRM as the optional second-stage verifier;
- beneficial / neutral / harmful replacement targets;
- expected-gain and cost-aware routing;
- random, low-score, uncertainty, failure-prediction, and benefit-only baselines;
- grouped bootstrap confidence intervals;
- held-out call-budget and same-hardware runtime comparisons.

Post-call arbitration was explored during feasibility but is not part of the primary held-out result.

## Alternative / Archived Directions

OVM completed a separate feasibility study and is retained as an archived alternative direction. Its compact evaluation summaries remain in GitHub, while raw generated seed outputs are stored in the team's private Hugging Face artifact repository.

- [OVM feasibility record](docs/proposals/ovm.md)

GenPRM is retained as an optional post-primary extension. Preliminary simplified GenPRM development runs are not treated as final held-out results.

## Repository Structure

```text
cl-prm-project/
├── configs/                    # Version-pinned experiment configurations and manifests
├── docs/                       # Proposals, literature notes, and meeting notes
├── experiments/
│   ├── feasibility/            # Completed exploratory/feasibility workflows
│   └── formal/prm_router/      # Frozen formal workflow + reserved final-analysis area
├── src/
│   └── cl_prm/                 # Shared data, routing, evaluation, and IO modules
├── data/                       # Generated local datasets, excluded from Git
├── outputs/                    # Generated local artifacts, excluded from Git
├── pyproject.toml              # Editable src-package configuration
├── requirements-common.txt     # Shared Python dependencies
├── requirements.txt            # Local/workstation dependency entry point
├── requirements-runpod.txt     # RunPod-safe dependency entry point
└── README.md
```

`src/cl_prm/` contains the reusable implementation shared by feasibility and formal workflows. Formal-only CLI entry points live under `experiments/formal/prm_router/`; completed exploratory scripts remain under `experiments/feasibility/`.

## Environment

The local ReasonEval and GenPRM feasibility experiments used Python 3.10 with an NVIDIA RTX 4060 Laptop GPU. PathFinder feasibility inference used Linux, BF16, Flash Attention 2, and an NVIDIA A40.

For the local Windows environment, install the PyTorch build appropriate
for the target CUDA version before installing `requirements.txt`, then install
the project package in editable mode with `python -m pip install -e . --no-deps --no-build-isolation`.

The completed formal run was executed on an NVIDIA A100-SXM4-80GB with the pinned software stack recorded in `experiments/formal/prm_router/run_gpu_environment.ipynb`. The notebook preserves the executed development and held-out workflow.

The optional [alternative PRM extension](experiments/extensions/prm_router/README.md) has independent official GenPRM, Qwen Math PRM, and Skywork PRM runners. It reuses the frozen splits and RE outputs while leaving the completed RE -> PathFinder notebook and formal results intact.

For compatible cloud environments, `bash scripts/bootstrap_runpod.sh` remains the environment bootstrap entry point; see [RunPod environment](docs/runpod.md) for the pinned stack and storage/secrets policy.

## Data and Outputs

Raw and generated data are not committed to Git.

- `data/README.md` documents deterministic dataset reconstruction.
- `configs/experiments/prm_router_formal.json` pins dataset and model revisions.
- `configs/experiments/prm_router_formal_split_manifest.json` records formal split hashes and statistics.
- `outputs/README.md` defines the local and shared artifact policy.
- [`cl-prm-team/cl-prm-artifacts`](https://huggingface.co/datasets/cl-prm-team/cl-prm-artifacts) is the private shared Hugging Face repository for generated inference outputs and analysis artifacts.

Model checkpoints should be downloaded from their pinned upstream revisions and must not be committed.

## Development Convention

- Completed feasibility code remains in `experiments/feasibility/prm_router/`.
- Formal-only experiment entry points live in `experiments/formal/prm_router/`.
- Reusable implementation belongs in `src/cl_prm/`.
- Generated datasets and outputs remain outside Git.
- Formal model selection uses training and validation data only.
- The held-out test set is evaluated only after thresholds and routing rules are frozen.
