# CL-PRM Project

This repository contains code, documentation, and experiments for a course project on process reward modeling in large language models.

## Project Status

The PRM Router feasibility study is complete and has passed the feasibility check. The recommended primary system is:

`ReasonEval -> utility router -> PathFinder -> optional risk-controlled arbitration`

GenPRM is retained as an alternative second-stage comparison baseline.

The project is ready for a final team decision. Formal held-out evaluation has not started. Preparation for that evaluation is complete:

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

## Alternative Candidates

OVM and StepBADGE are retained as early candidate placeholders while the final team decision is pending. No completed feasibility results for those directions are currently recorded in this repository.

- [OVM placeholder](docs/proposals/ovm.md)
- [StepBADGE placeholder](docs/proposals/stepbadge.md)

## Repository Structure

```text
cl-prm-project/
├── configs/                    # Version-pinned experiment configurations and manifests
├── docs/                       # Proposals, literature notes, and meeting notes
├── experiments/
│   └── feasibility/            # Completed feasibility code and candidate placeholders
├── src/
│   └── cl_prm/                 # Reserved for reusable formal-project modules
├── data/                       # Generated local datasets, excluded from Git
├── outputs/                    # Generated local artifacts, excluded from Git
├── requirements.txt            # Common Python dependencies
└── README.md
```

The empty `src/cl_prm/` package is reserved for reusable formal router, IO, and evaluation code. Formal shared entry points will be added only after the team confirms the project direction.

## Environment

The local ReasonEval and GenPRM feasibility experiments used Python 3.10 with an NVIDIA RTX 4060 Laptop GPU. PathFinder feasibility inference used Linux, BF16, Flash Attention 2, and an NVIDIA A40.

For the local Windows environment, install the PyTorch build appropriate
for the target CUDA version before installing `requirements.txt`.

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

Model checkpoints should be downloaded from their pinned upstream revisions and must not be committed.

## Development Convention

- Completed feasibility code remains in `experiments/feasibility/prm_router/`.
- Reusable formal-project modules belong in `src/cl_prm/`.
- Generated datasets and outputs remain outside Git.
- Formal model selection uses training and validation data only.
- The held-out test set is evaluated only after thresholds and routing rules are frozen.
