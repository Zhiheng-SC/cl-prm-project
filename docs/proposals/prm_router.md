# PRM Router Proposal

## Status

**Revised candidate proposal. Preliminary feasibility passed; final team selection is pending.**

The original proposal used GenPRM as the second-stage verifier. Feasibility experiments subsequently identified PathFinder-PRM as the stronger primary candidate. GenPRM is retained as an alternative second-stage baseline.

## Research Question

Can process verification achieve a better accuracy-compute trade-off by predicting when a stronger second-stage verifier will improve a base PRM, and by deciding whether to trust the second-stage judgement after the call?

The project studies two related decisions:

1. **Pre-call routing:** should the stronger verifier be invoked?
2. **Post-call arbitration:** if invoked, should its judgement replace the base verifier?

## Motivation

A single discriminative PRM is relatively efficient but can make systematic and highly confident errors. Stronger process verifiers can improve individual judgements, but applying them to every reasoning step increases inference cost and may also introduce harmful replacements.

Confidence and uncertainty are not equivalent to expected benefit. A low-confidence base prediction does not guarantee that the second verifier will be correct, while a high-confidence prediction may still contain an error that another verifier can detect.

This motivates predicting the marginal value of an additional verifier call rather than routing solely by confidence.

## Proposed Method

### Stage 1: Base verification

Run `GAIR/ReasonEval-7B` on every reasoning-step prefix and record:

* its validity score;
* its binary prediction;
* step position and solution length;
* question, prefix, and current-step lengths.

### Stage 2: Pre-call expected-gain routing

Train a lightweight router using only information available before the second-stage call.

Each training example receives one of three outcomes:

* beneficial: ReasonEval is wrong and the second-stage verifier is correct;
* neutral: both verifiers have the same correctness;
* harmful: ReasonEval is correct and the second-stage verifier is wrong.

The routing score is:

`P(beneficial) - P(harmful)`

Under a fixed call budget, only examples with the highest predicted expected gain are sent to the second-stage verifier.

### Stage 3: PathFinder verification

Use `declare-lab/PathFinder-PRM-7B` as the primary second-stage candidate.

PathFinder performs a hierarchical two-pass evaluation:

1. predict mathematical correctness and logical consistency;
2. condition the final correctness judgement on those intermediate signals.

The official Flash Attention 2 inference path is used for formal model output.

### Stage 4: Post-call trust arbitration

After PathFinder has been called, a second lightweight model decides whether its prediction should replace ReasonEval.

The arbitration study compares:

* overall PathFinder score, gate, and prediction;
* fine-grained mathematical and consistency signals;
* all available PathFinder signals.

PathFinder features are used only after the call and therefore do not leak into the pre-call routing decision.

## Models

### Base verifier

* `GAIR/ReasonEval-7B`
* One discriminative forward pass
* Pilot-selected validity threshold: `0.96`

### Primary second-stage verifier

* `declare-lab/PathFinder-PRM-7B`
* Hierarchical discriminative PRM
* Two-pass gated scoring
* BF16 with Flash Attention 2

### Alternative second-stage baseline

* `GenPRM/GenPRM-1.5B`
* Generative analysis followed by a Yes/No judgement
* One sampled verification path in the current feasibility implementation
* Retained as a comparison rather than the primary cascade

## Dataset

Dataset: `hitsmy/PRMBench_Preview`

Completed development subsets:

* threshold-selection pilot: 10 correct and 10 erroneous steps;
* feasibility subset: 50 correct and 50 erroneous steps;
* feasibility random seed: `2026`.

If selected as the final project, the current 100 examples will remain a frozen development pilot. New data will be split approximately into:

* training: 600 examples;
* validation: 200 examples;
* held-out test: 400 examples.

All splits must be grouped by normalized `original_question` so that variants of the same mathematical problem cannot cross dataset boundaries.

## Difference from Existing Work

The GenPRM paper studies generative process verification and test-time scaling through multiple verification paths.

The PathFinder-PRM paper introduces hierarchical error-aware supervision and separate mathematical-correctness and logical-consistency signals.

This project does not introduce a new PRM architecture. Its proposed contribution is the selective composition of existing complementary verifiers:

* predicting the marginal benefit of an additional verifier call;
* allocating calls under a fixed budget;
* distinguishing beneficial and harmful replacements;
* arbitrating whether to trust the second-stage output after the call.

The novelty claim must be checked against related dynamic-verification and routing work before the final report.

## Completed Feasibility Evidence

The 100-example pilot produced:

| Verifier | Accuracy |
|---|---:|
| ReasonEval | 0.66 |
| GenPRM | 0.74 |
| PathFinder | 0.81 |

ReasonEval-to-PathFinder complementarity:

| Outcome | Examples |
|---|---:|
| Both correct | 55 |
| ReasonEval correct, PathFinder wrong | 11 |
| ReasonEval wrong, PathFinder correct | 26 |
| Both wrong | 8 |

The ReasonEval-PathFinder oracle accuracy is `0.92`.

By comparison, GenPRM corrects only two PathFinder errors while damaging nine correct PathFinder predictions. The PathFinder-GenPRM oracle is `0.83`, compared with PathFinder-only accuracy of `0.81`.

### Pre-call routing

The grouped out-of-fold expected-gain router achieved:

* benefit average precision: `0.8210`;
* harm average precision: `0.2990`;
* 20% PathFinder budget accuracy: `0.82`;
* 20% uncertainty-routing accuracy: `0.71`;
* 40% PathFinder budget accuracy: `0.84`.

At the 20% budget, the router selected 17 beneficial and one harmful call.

### Post-call arbitration

The overall-signal arbitration model achieved:

* benefit average precision: `0.798`;
* harm average precision: `0.720`;
* accuracy of `0.85` at 75% and 100% PathFinder-call budgets.

Fine-grained signals did not outperform the simpler overall feature set in this pilot. They are therefore retained as a predefined ablation rather than a primary positive claim.

## Formal Experiment Plan

If this direction is selected:

1. freeze the current 100 examples and all current pilot conclusions;
2. prepare grouped train, validation, and held-out test sets;
3. run ReasonEval and PathFinder on the same GPU;
4. tune thresholds, features, and budgets using only training and validation data;
5. freeze the protocol before opening held-out test results;
6. evaluate the held-out test set once;
7. report accuracy-budget and accuracy-latency curves;
8. compute bootstrap confidence intervals and paired significance tests;
9. compare overall and fine-grained arbitration signals;
10. retain GenPRM as an alternative baseline if compute permits.

## Baselines

The final comparison should include:

* ReasonEval only;
* PathFinder only;
* random routing;
* low-score routing;
* uncertainty routing;
* benefit-only routing;
* expected-gain routing;
* always-accept PathFinder cascade;
* post-call trust arbitration;
* oracle routing as an unattainable upper bound.

## Success Criteria

The strongest positive result would show that expected-gain routing:

* outperforms random and uncertainty routing at equal call budgets;
* approaches PathFinder-only accuracy with substantially fewer PathFinder calls;
* provides a better same-hardware accuracy-latency frontier;
* reduces harmful replacements through post-call arbitration.

Raw accuracy does not need to exceed PathFinder-only performance if comparable accuracy is achieved with materially fewer calls.

## Expected Compute

The feasibility experiment ran locally and on a rented NVIDIA A40 with 48 GB VRAM.

The formal experiment should run ReasonEval and PathFinder on the same Linux GPU so that runtime and compute comparisons are valid. PathFinder uses two model passes, making selective invocation potentially valuable even though ReasonEval and PathFinder have similar parameter counts.

## Main Risks

1. The 100-example pilot may overestimate router performance.
2. Benefit and harm classes may be too rare for stable training.
3. Router features may not generalize to held-out problems.
4. ReasonEval threshold selection may influence the apparent routing opportunity.
5. The balanced pilot distribution may not represent natural benchmark data.
6. PathFinder may be insufficiently more expensive than ReasonEval to justify routing.
7. Post-call arbitration may help only at high call budgets.
8. Fine-grained PathFinder signals may add no predictive value.
9. Existing dynamic-verification work may limit methodological novelty.
10. Same-hardware latency results may differ from the current mixed-device measurements.

## Current Blockers

There are no technical blockers for the candidate proposal.

Before starting the formal experiment, the team must:

* select the final project direction;
* agree on the primary research claim;
* confirm access to a Linux GPU;
* freeze the dataset split and evaluation protocol;
* assign model inference, routing, evaluation, and report responsibilities.

## Decision

**Passed as a candidate direction.**

The feasibility evidence supports ReasonEval-to-PathFinder routing as the primary candidate cascade. GenPRM remains a useful alternative baseline, but the current pilot does not support PathFinder-to-GenPRM routing as a core method.

The next decisive step, if selected, is a frozen held-out evaluation.

## Relevant Resources

* [PathFinder-PRM paper](https://arxiv.org/abs/2505.19706)
* [PathFinder-PRM repository](https://github.com/declare-lab/PathFinder-PRM)
* [PathFinder-PRM-7B](https://huggingface.co/declare-lab/PathFinder-PRM-7B)
* [GenPRM paper](https://arxiv.org/abs/2504.00891)
* [ReasonEval-7B](https://huggingface.co/GAIR/ReasonEval-7B)
* [PRMBench Preview](https://huggingface.co/datasets/hitsmy/PRMBench_Preview)
