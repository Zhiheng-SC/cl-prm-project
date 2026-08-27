# CL-PRM Project

This repository contains the code, documentation, and feasibility experiments for our course project on process reward modeling in large language models.

## Project Status

The final project direction has not yet been selected. We are currently comparing three candidate directions:

1. OVM
2. PRM Router
3. StepBADGE

The preliminary PRM Router feasibility study has been completed and passed the feasibility check. Its results are exploratory and are not final held-out benchmark results.

See the detailed report:

* [PRM Router Feasibility Study](experiments/feasibility/prm_router/README.md)

## Repository Structure

```text
cl-prm-project/
├── docs/                       # Proposals, literature notes, and meeting notes
├── experiments/
│   └── feasibility/            # Feasibility studies for candidate directions
├── src/
│   └── cl_prm/                 # Reusable and finalized project code
├── scripts/                    # Shared command-line entry points
├── data/                       # Local datasets, excluded from Git
├── outputs/                    # Local experiment outputs, excluded from Git
├── requirements.txt            # Python dependencies
└── README.md
```

## Candidate Directions

### OVM

Proposal and feasibility notes:

* [OVM Proposal](docs/proposals/ovm.md)
* [OVM Feasibility](experiments/feasibility/ovm/README.md)

### PRM Router

The PRM Router direction studies whether a fast discriminative PRM can be combined with a slower generative PRM through budget-aware routing.

The current feasibility experiment includes:

* balanced PRMBench subset preparation;
* ReasonEval inference;
* GenPRM inference;
* verifier complementarity analysis;
* heuristic routing baselines;
* a lightweight logistic-regression benefit router.

See:

* [PRM Router Proposal](docs/proposals/prm_router.md)
* [PRM Router Feasibility](experiments/feasibility/prm_router/README.md)

### StepBADGE

Proposal and feasibility notes:

* [StepBADGE Proposal](docs/proposals/stepbadge.md)
* [StepBADGE Feasibility](experiments/feasibility/stepbadge/README.md)

## Environment

The current PRM Router feasibility experiment uses Python 3.10.

Install the appropriate PyTorch build for the local CUDA environment first. For CUDA 12.4:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Detailed reproduction commands are provided in the README for each feasibility experiment.

## Data and Outputs

Large datasets, model checkpoints, generated predictions, and experiment outputs must not be committed to Git.

* Local datasets are stored under `data/`.
* Generated results are stored under `outputs/`.
* Model weights and checkpoints are stored locally.

Only the corresponding documentation files are tracked by Git.

## Development Convention

* Exploratory or candidate-specific code belongs in `experiments/feasibility/`.
* Reusable and finalized code belongs in `src/cl_prm/`.
* Shared command-line entry points belong in `scripts/`.
* Experimental results and decisions should be documented in the corresponding feasibility README.
