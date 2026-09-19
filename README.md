# CL-PRM Project

This repository contains code, documentation, and experiments for a course project on process reward modeling in large language models.

## Project Status

The PRM Router feasibility study is complete and has passed the feasibility check. The recommended primary system is:

`ReasonEval -> utility router -> PathFinder -> optional risk-controlled arbitration`

GenPRM is retained as an alternative second-stage comparison baseline.

The PRM Router has been selected as the primary project direction. Formal held-out evaluation has not started. Preparation for that evaluation is complete:

- dataset and model revisions are pinned;
- the 100-example feasibility pilot is excluded from formal data;
- deterministic grouped train, validation, and test splits are defined;
- a tracked manifest records dataset provenance, split statistics, and SHA256 hashes.

The original 100-example pilot has now been confirmed by a 200-example
expanded feasibility study with five grouped-CV seeds. The reported
feasibility results remain exploratory and are not final held-out benchmark
results.

See:

- [PRM Router proposal](docs/proposals/prm_router.md)
- [PRM Router feasibility study](experiments/feasibility/prm_router/README.md)
- [Expanded 200-example feasibility results](docs/results/prm_router_expanded_feasibility_200.md)

## Primary Candidate

The PRM Router project studies whether pair-specific correction utility can allocate limited verification compute more effectively than confidence-based escalation.

The completed feasibility study includes:

- balanced PRMBench subset preparation;
- ReasonEval, GenPRM, and PathFinder inference;
- verifier complementarity analysis;
- random and uncertainty-routing baselines;
- grouped out-of-fold benefit and expected-gain routers;
- DisPRM threshold sensitivity and grouped leakage checks;
- post-call trust arbitration;
- explicit pilot limitations and a held-out evaluation plan.

## Alternative / Archived Directions

OVM completed a separate feasibility study and is retained as an archived alternative direction. Its compact evaluation summaries remain in GitHub, while raw generated seed outputs are stored in the team's private Hugging Face artifact repository. StepBADGE remains an unselected proposal only.

- [OVM feasibility record](docs/proposals/ovm.md)
- [StepBADGE proposal](docs/proposals/stepbadge.md)

## Repository Structure

```text
cl-prm-project/
├── configs/                    # Version-pinned experiment configurations and manifests
├── docs/                       # Proposals, literature notes, and meeting notes
├── experiments/
│   ├── feasibility/            # Completed exploratory/feasibility workflows
│   └── formal/prm_router/      # Guarded formal train/validation/test entry points
├── src/
│   └── cl_prm/                 # Shared data, routing, evaluation, and IO modules
├── data/                       # Generated local datasets, excluded from Git
├── outputs/                    # Generated local artifacts, excluded from Git
├── pyproject.toml              # Editable src-package configuration
├── requirements.txt            # Common Python dependencies
└── README.md
```

`src/cl_prm/` contains the reusable implementation shared by feasibility and formal workflows. Formal-only CLI entry points live under `experiments/formal/prm_router/`; completed exploratory scripts remain under `experiments/feasibility/`.

## Environment

The local ReasonEval and GenPRM feasibility experiments used Python 3.10 with an NVIDIA RTX 4060 Laptop GPU. PathFinder feasibility inference used Linux, BF16, Flash Attention 2, and an NVIDIA A40.

For the local Windows environment, install the PyTorch build appropriate
for the target CUDA version before installing `requirements.txt`, then install
the project package in editable mode with `python -m pip install -e . --no-deps`.

The verified RunPod image already provides a compatible PyTorch,
CUDA, and FlashAttention binary stack. Do not replace that stack with the
local requirements file. Use:

```bash
bash scripts/bootstrap_runpod.sh
```

See [RunPod environment](docs/runpod.md) for the verified versions, secrets,
storage policy, data reconstruction, and startup workflow.

The formal experiment must record the exact Python, PyTorch, Transformers, CUDA, attention-backend, and GPU versions used for inference.

## Data and Outputs

Raw and generated data are not committed to Git.

- `data/README.md` documents deterministic dataset reconstruction.
- `configs/experiments/prm_router_formal.json` pins dataset and model revisions.
- `configs/experiments/prm_router_formal_split_manifest.json` records formal split hashes and statistics.
- `outputs/README.md` defines the local and shared artifact policy.
- `cl-prm-team/cl-prm-artifacts` is the private shared Hugging Face repository for generated inference outputs and analysis artifacts.

Model checkpoints should be downloaded from their pinned upstream revisions and must not be committed.

## Development Convention

- Completed feasibility code remains in `experiments/feasibility/prm_router/`.
- Formal-only experiment entry points live in `experiments/formal/prm_router/`.
- Reusable implementation belongs in `src/cl_prm/`.
- Generated datasets and outputs remain outside Git.
- Formal model selection uses training and validation data only.
- The held-out test set is evaluated only after thresholds and routing rules are frozen.
