# PRM Router Proposal

## Status

**Preliminary feasibility check passed. Final project selection is still pending.**

## Research Question

Can process verification achieve a better accuracy–compute trade-off by first applying a fast discriminative process reward model and selectively invoking a slower generative process reward model only when the second verifier is expected to improve the decision?

## Motivation

Discriminative PRMs are relatively efficient but may make systematic or highly confident errors. Generative PRMs can perform more detailed verification by generating an explicit analysis, but invoking them for every reasoning step is computationally expensive.

This project studies whether limited GenPRM inference-time compute can be allocated to the reasoning steps where it provides the greatest marginal benefit.

## Proposed Method

The proposed cascade contains three components:

1. Run a discriminative PRM on every reasoning-step prefix.
2. Use a lightweight router to estimate the benefit of invoking GenPRM.
3. Route only the highest-ranked examples to GenPRM under a fixed computational budget.

The router may use only information available before the GenPRM call, including:

* the discriminative PRM score;
* distance from its decision threshold;
* its binary prediction;
* step position and solution length;
* question, prefix, and current-step lengths.

The router must not use the ground-truth label, GenPRM output, or PRMBench error-type classification at test time.

The supervised benefit target is initially defined as:

```text
positive = DisPRM is wrong and GenPRM is correct
negative = all other cases
```

## Difference from Existing GenPRM Work

The GenPRM paper studies generative process verification and test-time scaling through multiple verification paths.

This proposal does not introduce a new GenPRM. Instead, it studies how to allocate a limited number of calls to an existing GenPRM across different reasoning steps. The central question is whether the marginal benefit of calling GenPRM can be predicted more effectively than with confidence or uncertainty heuristics.

## Minimum Experiment

The minimum feasibility experiment consists of:

1. preparing a balanced subset of PRMBench step prefixes;
2. running ReasonEval and GenPRM on the same examples;
3. measuring their individual accuracy and runtime;
4. constructing the four verifier-outcome groups;
5. evaluating random, low-score, and uncertainty routing;
6. training a lightweight logistic-regression benefit router;
7. comparing accuracy under equal GenPRM call budgets.

## Models and Datasets

### Dataset

* `hitsmy/PRMBench_Preview`
* Pilot subset: 10 correct and 10 erroneous steps
* Feasibility subset: 50 correct and 50 erroneous steps

### Discriminative PRM

* `GAIR/ReasonEval-7B`
* Scalar step-correctness score
* Diagnostic threshold selected on the separate pilot subset

### Generative PRM

* `GenPRM/GenPRM-1.5B`
* One generated analysis and judgement per example
* Current local implementation does not execute model-generated verification code

## Expected Compute

The feasibility experiment can run locally on an NVIDIA RTX 4060 Laptop GPU with 8 GB VRAM.

Measured mean inference times on the 100-example feasibility subset were:

* ReasonEval: approximately 2.02 seconds per example;
* GenPRM: approximately 3.09 seconds per example.

A full project may require access to an A100 or another Linux GPU environment to reproduce the official GenPRM code-execution and majority-voting pipeline.

## Main Risks

1. The two verifiers may not exhibit enough complementary errors.
2. GenPRM calls may be harmful when the discriminative PRM is already correct.
3. Simple router features may not generalize to held-out examples.
4. The balanced feasibility subset may not represent the natural benchmark distribution.
5. Similar variants of the same mathematical problem may cause leakage unless data are split by original problem.
6. The simplified local GenPRM implementation differs from the complete official pipeline.
7. Existing dynamic-verification work may limit the novelty of simple confidence-based routing.

## Current Blockers

There are no blockers for the feasibility implementation.

If this direction is selected as the final project, the main requirements are:

* a separate held-out evaluation set;
* grouped splitting by original problem;
* a frozen feature and routing protocol;
* statistical uncertainty estimates;
* stronger and compute-matched baselines;
* evaluation of the official GenPRM pipeline if compute permits.

## Feasibility Result

The 100-example feasibility experiment produced:

* ReasonEval accuracy: `0.66`;
* GenPRM accuracy: `0.74`;
* beneficial GenPRM calls: `26`;
* harmful GenPRM calls: `18`;
* oracle routing upper bound: `0.92`.

The five-fold out-of-fold logistic-regression router achieved:

* ROC-AUC: `0.9033`;
* average precision: `0.8141`;
* accuracy at a 20% GenPRM budget: `0.79`;
* uncertainty-routing accuracy at the same budget: `0.67`.

These results indicate that verifier complementarity exists and that simple observable features contain a useful benefit-prediction signal. They remain exploratory and are not final held-out benchmark results.

## Decision

**Passed as a candidate direction.**

The PRM Router direction is technically feasible and empirically promising enough to be considered during final project selection. A held-out experiment is required before making claims about generalization.

Detailed implementation and results are available in the [PRM Router Feasibility Study](../../experiments/feasibility/prm_router/README.md).
