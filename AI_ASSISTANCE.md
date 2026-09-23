# AI-Assisted Development

AI coding assistants were used extensively in the software implementation of this project. A substantial portion of the repository code was initially drafted, revised, refactored, debugged, or documented with AI assistance.

The team remained responsible for the research direction, experimental design, verifier and baseline selection, routing formulation, formal train/validation/test protocol, model execution, protocol freezing, validation of generated code, evaluation, and scientific interpretation.

AI-assisted code was not accepted without review. The team executed the resulting pipelines, inspected intermediate and final artifacts, investigated failures, validated the frozen formal protocol, and checked reported results against the generated experiment outputs.

## Main AI-assisted components

The main AI-assisted implementation areas include:

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

This list is representative rather than exhaustive: unless otherwise stated, most newly developed Python code in the PRM-routing pipeline involved some level of AI-assisted implementation.

## Human responsibility

The team independently determined and/or verified:

- the project research questions and scope;
- the ReasonEval -> router -> optional PathFinder primary design;
- beneficial, neutral, and harmful call definitions;
- the grouped 600/200/400 formal split protocol;
- the pre-call feature set and primary 20% call budget;
- the validation-only threshold and utility-selection procedure;
- the held-out test lock and one-time evaluation procedure;
- the actual GPU runs and artifact handling;
- interpretation of the held-out results and final scientific claims.

All final code and report claims remain the responsibility of the team.
