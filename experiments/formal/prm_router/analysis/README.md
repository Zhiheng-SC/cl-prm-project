# Formal Held-Out Analysis

This directory is reserved for the final descriptive analysis and visualization
of the frozen ReasonEval--PathFinder formal experiment.

The primary protocol has already been frozen and evaluated on the 400-example
held-out test split. Scripts added here should analyze the existing artifacts
without changing the frozen model, features, threshold, utility weights, or
primary call budget.

## Suggested analysis directions

The following are useful directions for the final report. They are suggestions,
not a required file structure, and can be reorganized as the analysis develops.

### Main result presentation

- summarize the primary 20% results for ReasonEval-only, PathFinder-only,
  heuristic baselines, learned routers, expected-gain routing, and cost-aware
  routing;
- include grouped-bootstrap confidence intervals where available;
- summarize measured sequential-cascade runtime alongside accuracy.

### Budget and compute trade-offs

- plot held-out accuracy across the pre-specified PathFinder call budgets;
- compare accuracy against measured cascade runtime;
- highlight the ReasonEval-only and PathFinder-only reference points.

### Beneficial / neutral / harmful call analysis

- report how many beneficial, neutral, and harmful calls each routing method
  selects at the primary 20% budget;
- report beneficial-capture and harmful-selection rates;
- inspect how these quantities change across the pre-specified budget curve.

### Expected gain versus failure prediction

Failure prediction is a strong held-out baseline. A useful diagnostic is to
compare which examples are selected by failure prediction and expected-gain
routing:

- selected by both;
- selected only by expected gain;
- selected only by failure prediction;
- selected by neither.

For the disagreement subsets, summarize beneficial, neutral, and harmful
outcomes. This can help explain where the small difference between the two
learned routers comes from.

### Router diagnostics

Possible descriptive diagnostics include:

- beneficial one-vs-rest PR-AUC and ROC-AUC;
- harmful one-vs-rest PR-AUC and ROC-AUC;
- Brier scores for predicted beneficial / harmful probabilities;
- reliability or calibration plots;
- score distributions for beneficial, neutral, and harmful examples.

Because beneficial and harmful outcomes are imbalanced, PR-AUC should be
interpreted together with the corresponding class prevalence.

## Protocol guardrail

Held-out analysis is descriptive only. Do not use the test split to:

- add or remove router features;
- change the router model family;
- retune the ReasonEval threshold;
- retune `lambda_h` or `mu`;
- choose a new primary call budget;
- redefine the primary comparison after seeing test performance.

Additional descriptive views are welcome as long as they do not alter the
frozen primary result.

## Inputs and outputs

The main frozen artifacts are stored outside Git under the formal output layout
and in the shared Hugging Face artifact repository.

Analysis scripts may write generated tables, JSON/CSV summaries, and figures
under:

`outputs/prm_router/formal/analysis/`

Generated outputs should remain out of Git unless a small, human-auditable
summary is intentionally committed. Final figures selected for the paper can be
copied into the writing workspace separately.

## Implementation notes

There is no required script naming convention. A reasonable organization could
separate:

- statistical diagnostics;
- plotting;
- report-table generation.

The analysis code can be revised as needed while keeping the frozen-protocol
guardrail above intact.

## Implemented workflow

Run the analysis from the repository root with the pinned analysis environment:

```powershell
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-analysis.txt
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe experiments/formal/prm_router/analysis/analyze_test_statistics.py
.venv\Scripts\python.exe experiments/formal/prm_router/analysis/make_final_tables.py
.venv\Scripts\python.exe experiments/formal/prm_router/analysis/make_final_figures.py
```

The current scripts default to the local sibling path `../cl_finalproject_hg`, which is only a convenience used for the completed analysis run. The artifact repository may be cloned anywhere; pass its path explicitly with `--hf-root` on other machines. Generated files are written under `outputs/prm_router/formal/analysis/`, which remains excluded from Git.

`analyze_test_statistics.py` stops before all descriptive analyses unless its
reproduction gate matches the frozen formal evaluator. Average Precision and
trapezoidal PR-AUC are reported under distinct names. Router discrimination and held-out calibration are additional descriptive diagnostics only; they are not used to recalibrate the router or alter the frozen routing policy. The expected-gain versus failure-prediction overlap is explicitly labeled post-hoc descriptive analysis.


## Archiving generated analysis artifacts

After a successful run and a PASS reproduction gate, archive the generated
analysis outputs in the shared artifact repository rather than committing them
to Git:

```bash
hf upload cl-prm-team/cl-prm-artifacts \
  outputs/prm_router/formal/analysis \
  evaluation/router/final_analysis \
  --repo-type dataset
```

This keeps GitHub focused on reproducible source code while the generated
tables, statistics, and figures remain versioned with the experiment artifacts.
Only figures actually used in the final report need to be copied separately
into the Overleaf writing workspace.
