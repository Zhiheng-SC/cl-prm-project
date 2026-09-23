# AI-Assisted Development

AI coding assistants were used frequently during the implementation of this project, especially for drafting or revising parts of scripts, debugging, refactoring, writing tests and utilities, and improving documentation.

The repository is not entirely AI-generated. Team members also wrote and modified code directly, integrated components, performed substantial debugging, ran the experiments, inspected failures and intermediate outputs, and revised implementations when the generated or existing code did not behave as intended.

The team remained responsible for the research direction, experimental design, verifier and baseline selection, routing formulation, formal train/validation/test protocol, model execution, protocol freezing, evaluation choices, validation of the resulting code, and scientific interpretation.

AI-assisted code was reviewed in the context of the actual experiments rather than accepted automatically. The team executed the pipelines, checked intermediate and final artifacts, investigated errors, verified the frozen formal protocol, and compared reported results against the generated experiment outputs.

## Main areas of AI assistance

AI assistance was used to varying degrees across several implementation areas, including:

- formal split construction and protocol-validation utilities;
- ReasonEval, PathFinder, and GenPRM inference wrappers;
- routing, cost-prediction, bootstrap, and evaluation utilities;
- development-selection and held-out evaluation scripts;
- tests, repository utilities, experiment runbooks, and documentation;
- statistical-analysis and visualization scripts where AI assistance is used.

Representative scripts and modules include:

- `experiments/formal/prm_router/prepare_formal_splits.py`
- `experiments/formal/prm_router/run_formal_inference.py`
- `experiments/formal/prm_router/evaluate_development.py`
- `experiments/formal/prm_router/evaluate_test.py`
- `experiments/feasibility/prm_router/run_disprm.py`
- `experiments/feasibility/prm_router/run_pathfinder.py`
- `experiments/feasibility/prm_router/run_genprm.py`
- shared routing/evaluation utilities under `src/cl_prm/`

This list is representative rather than exhaustive. The amount and type of AI assistance differ across files: some code was drafted with AI support, some was manually written and later revised or debugged with AI assistance, and some was primarily written or modified by team members.

## Human contribution and verification

The team independently determined and/or carried out:

- the project research questions and scope;
- the ReasonEval -> router -> optional PathFinder primary design;
- beneficial, neutral, and harmful call definitions;
- the grouped 600/200/400 formal split protocol;
- the pre-call feature set and primary 20% call budget;
- the validation-only threshold and utility-selection procedure;
- the held-out test lock and one-time evaluation procedure;
- implementation integration and substantial manual debugging;
- the actual GPU runs and artifact handling;
- inspection and validation of generated outputs;
- interpretation of the held-out results and final scientific claims.

All final code, experiments, and report claims remain the responsibility of the team.
