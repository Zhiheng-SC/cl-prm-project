# Expanded PRM Router Feasibility Study

## Status

**Decision: passed on 2026-09-15.**

This experiment expands the earlier 100-example pilot to 200 examples from
the frozen formal training split. It is a development-stage feasibility check,
not a validation or held-out test result.

The experiment supports continuing with the proposed system:

```text
ReasonEval -> expected-gain router -> PathFinder
           -> optional post-call arbitration
```

The held-out test split was not accessed.

## Scope and provenance

| Item | Value |
|---|---|
| Examples | 200 |
| Original-question groups | 170 |
| Source | First 200 records of the frozen formal train split |
| Input SHA256 | `3a179b26a8af41e6cbb0ccb65e41298721655e2adbfe5194a2b825116ebf44e7` |
| Repository commit used for inference | `5b56e2e714621fd5deb925b03759ef5f4e9d0b4f` |
| Archived bundle SHA256 | `8588012d906e73f732dcbf249de3f34f4fd0b3ccdc2e326bfd46a5f5d42c5d1a` |

The archived inference files are not committed because generated outputs remain
excluded from Git. Their sidecar metadata records the commands, model revisions,
environment, input hash, output hash, timing protocol, and completion status.

| Output | SHA256 |
|---|---|
| `reasoneval_200.jsonl` | `224eca7ee2ff7c7e480c183cef6b8d3c56dc94aec558ac94ef7f568d8543b95b` |
| `pathfinder_200.jsonl` | `2ecca8ed495a061005c607e521e9cb113695bd262d69cea5b3869bef4c2bdac1` |

## Verifiers and environment

| Component | Frozen value |
|---|---|
| ReasonEval | `GAIR/ReasonEval-7B` at `0a6556ef5c937bb17d265ba681b501fd60056cfe` |
| PathFinder | `declare-lab/PathFinder-PRM-7B` at `84a7412511836cb4ed74377d9c703eb5638d814c` |
| GPU | NVIDIA A40, 48 GB |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA reported by PyTorch | 12.8 |
| Transformers | 4.48.2 |
| Accelerate | 1.2.0 |
| FlashAttention | 2.8.3.post1 |
| Precision | BF16 |
| Warmup | 3 untimed examples |

Both verifiers were measured on the same GPU. Model loading was excluded and
CUDA was synchronized around the recorded inference regions.

## Verifier accuracy and complementarity

| Outcome | Count | Fraction |
|---|---:|---:|
| Both correct | 126 | 63.0% |
| ReasonEval only correct | 10 | 5.0% |
| PathFinder only correct | 37 | 18.5% |
| Both wrong | 27 | 13.5% |

| Metric | Result |
|---|---:|
| ReasonEval accuracy at threshold 0.50 | 0.680 |
| PathFinder accuracy | 0.815 |
| Prediction disagreement | 0.235 |
| Full oracle upper bound | 0.865 |
| McNemar exact p-value | `9.84888e-05` |

PathFinder corrects 37 ReasonEval errors while damaging 10 initially correct
predictions. The asymmetry strongly favors PathFinder, but the harmful cases
show why selective replacement remains useful.

## Runtime

| Verifier | Mean seconds per example |
|---|---:|
| ReasonEval | 0.060819 |
| PathFinder | 0.134722 |

PathFinder inference is approximately 2.22 times the ReasonEval inference time.
For a routing fraction (b), the measured sequential cascade cost is estimated
as

[
T(b) = 0.060819 + b 	imes 0.134722.
]

This approximation excludes model loading and does not claim deployment-level
throughput.

## Grouped out-of-fold router evaluation

The pre-call router uses only ReasonEval outputs and input-derived features.
It does not use the ground-truth label, PathFinder prediction, PathFinder
scores, or PRMBench error type when deciding whether to call PathFinder.

Five `StratifiedGroupKFold` evaluations were run with seeds:

```text
7, 42, 2026, 20260903, 20260915
```

All variants of one normalized original question remain in the same fold.

### Per-seed results

| Seed | Benefit AP | Harm AP | Accuracy at 20% | Accuracy at 40% | Accuracy at 50% |
|---:|---:|---:|---:|---:|---:|
| 7 | 0.5882 | 0.1264 | 0.765 | 0.830 | 0.830 |
| 42 | 0.6450 | 0.0968 | 0.775 | 0.825 | 0.840 |
| 2026 | 0.6137 | 0.0897 | 0.770 | 0.825 | 0.825 |
| 20260903 | 0.6640 | 0.0825 | 0.780 | 0.825 | 0.830 |
| 20260915 | 0.6349 | 0.1326 | 0.780 | 0.830 | 0.830 |

### Stability summary

| Metric | Mean | SD | Range |
|---|---:|---:|---:|
| Benefit AP | 0.6292 | 0.0261 | 0.5882--0.6640 |
| Harm AP | 0.1056 | 0.0201 | 0.0825--0.1326 |
| Accuracy at 10% | 0.7410 | 0.0049 | 0.735--0.745 |
| Accuracy at 20% | 0.7740 | 0.0058 | 0.765--0.780 |
| Accuracy at 40% | 0.8270 | 0.0024 | 0.825--0.830 |
| Accuracy at 50% | 0.8310 | 0.0049 | 0.825--0.840 |

| Budget | Mean router accuracy | Mean random accuracy | Mean advantage |
|---:|---:|---:|---:|
| 10% | 0.7410 | 0.6936 | +0.0474 |
| 20% | 0.7740 | 0.7069 | +0.0671 |
| 40% | 0.8270 | 0.7342 | +0.0928 |
| 50% | 0.8310 | 0.7476 | +0.0834 |

At the primary 20% budget, all five seeds outperform random routing. The
standard deviation is 0.0058 and the worst seed still reaches 0.765. At 40%,
the router consistently reaches 0.825--0.830, slightly above PathFinder-only
accuracy while using PathFinder on only 80 of 200 examples.

The benefit prevalence is (37/200=0.185). Mean benefit average precision
of 0.6292 is therefore substantially above the random ranking baseline.

Harm average precision is much weaker. Only 10 harmful examples are available,
so harm prediction should not be treated as a stable standalone result.

## Post-call arbitration

For seed 42, post-call PathFinder features produced the following grouped OOF
benefit average precision:

| Feature set | Benefit AP | Harm AP |
|---|---:|---:|
| Overall | 0.863 | 0.132 |
| Fine-grained | 0.875 | 0.113 |
| All signals | 0.874 | 0.142 |

Arbitration does not improve accuracy at 10%--50% call budgets because the
pre-call router already ranks the useful calls effectively. It provides small
improvements only at high call budgets. It should therefore remain an optional
secondary component or ablation rather than the main contribution.

## Predefined feasibility gate

Before the five-seed run, the expanded study was defined to pass if:

1. every seed beats its corresponding random baseline at the 20% budget;
2. mean 20% accuracy is at least 0.75;
3. mean 20% advantage over random is at least 0.04;
4. mean 40% accuracy is at least 0.81;
5. mean benefit AP is at least 0.50;
6. no seed falls below 0.72 at the 20% budget.

All six conditions pass.

## Interpretation

The expanded experiment supports three development-stage conclusions:

1. ReasonEval and PathFinder have meaningful, asymmetric complementarity.
2. Cheap pre-call features contain a stable signal for ranking beneficial
   PathFinder calls.
3. Budgeted expected-gain routing provides a substantially better
   accuracy-compute trade-off than random or uncertainty routing on this
   sample.

These conclusions justify selecting the direction for a course project. They
do not establish final generalization performance.

## Limitations

- The 200 examples come from the training split.
- Only 170 independent original-question groups are represented.
- There are only 37 beneficial and 10 harmful calls.
- Multiple seeds test fold-assignment stability, not independent-dataset
  generalization.
- Hyperparameters, feature sets, and budgets were inspected on development
  data.
- The ReasonEval threshold has not been selected on independent validation
  data.
- No group-bootstrap confidence interval has yet been reported.
- The held-out test set has not been accessed.

## Decision and next stage

**Expanded feasibility is complete. No additional feasibility-scale GPU
inference is required.**

If the team selects this project, the next stage is:

1. run complete train and validation inference on one common GPU;
2. choose the ReasonEval threshold, router target, features, call budget, and
   optional arbitration using train and validation only;
3. freeze the complete protocol;
4. run the held-out test exactly once;
5. report group-bootstrap confidence intervals and paired statistical tests.
