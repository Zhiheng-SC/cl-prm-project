# PRM Router Proposal

## Status

**Selected primary project direction. The formal ReasonEval--PathFinder held-out evaluation is complete.**

The frozen primary system is:

`ReasonEval -> utility router -> optional PathFinder`

The formal study used grouped 600/200/400 train/validation/test splits and a pre-specified 20% primary PathFinder call budget. Validation selected the exact ReasonEval threshold and the cost-aware utility weights before the held-out test was opened.

Held-out accuracy on 400 test examples is 0.7175 for ReasonEval-only, 0.7900 for PathFinder-only, and 0.8175 for both expected-gain and cost-aware routing at the 20% call budget. The pairwise oracle is 0.9225.

Post-call arbitration was explored during feasibility but is not part of the primary held-out result. GenPRM remains an optional post-primary verifier-pair extension.

## Research Question

Can pair-specific correction utility allocate expensive process-verification compute more effectively than confidence- or difficulty-based routing, when correction benefit, harmful replacement risk, and inference cost are considered?

The project studies two related decisions:

1. **Pre-call routing:** is the expected correction benefit worth the additional verification cost?
2. **Post-call arbitration:** after observing the second verifier, is replacing the base judgement sufficiently reliable?

The formal study has three primary research questions plus one pre-specified secondary question:

1. **RQ1 — Complementarity:** Do ReasonEval and PathFinder exhibit enough complementary errors to create exploitable routing headroom?
2. **RQ2 — Pair-specific utility:** Does pair-specific expected correction utility outperform uncertainty and base-verifier failure prediction on held-out data?
3. **RQ3 — Compute-aware routing:** Does incorporating predicted additional inference cost improve the accuracy-compute trade-off relative to call-budget-only routing?
4. **Secondary RQ4 — Risk-controlled arbitration:** Can post-call evidence reduce harmful replacements without sacrificing too many beneficial corrections?

Cross-dataset transfer to ProcessBench and a formal ReasonEval-to-GenPRM comparison are optional extensions after the primary PRMBench study is complete.

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
* question and current-step lengths;
* exact ReasonEval input-token count.

### Stage 2: Pre-call utility routing

Train a lightweight router using only information available before the second-stage call.

Each training example receives one of three outcomes:

* beneficial: ReasonEval is wrong and the second-stage verifier is correct;
* neutral: both verifiers have the same correctness;
* harmful: ReasonEval is correct and the second-stage verifier is wrong.

The completed feasibility baseline uses:

`expected gain = P(beneficial) - P(harmful)`

The formal extension will additionally study a cost- and risk-aware utility:

`utility = P(beneficial) - lambda_h * P(harmful) - mu * predicted additional cost`

The additional cost term must be available before PathFinder is called. A lightweight cost predictor will therefore be trained on the formal training split to predict PathFinder runtime from pre-call features such as token/character length and step position. Measured PathFinder runtime is used as the prediction target and for final compute accounting, but never as an input to the routing decision for the same example. Predicted cost is normalized by the median measured PathFinder runtime on the training split. The Ridge cost predictor uses a pre-specified regularization value (`alpha = 1.0`). The utility weights are selected on validation only; realized same-example PathFinder runtime is never used as a routing input.

Under each fixed call budget, the router ranks examples by predicted utility and calls PathFinder for the top-ranked fraction. The primary confirmatory budget is fixed at 20%; the other pre-specified budgets are secondary curve points.

### Stage 3: PathFinder verification

Use `declare-lab/PathFinder-PRM-7B` as the primary second-stage candidate.

PathFinder performs a hierarchical two-pass evaluation:

1. predict mathematical correctness and logical consistency;
2. condition the final correctness judgement on those intermediate signals.

The official Flash Attention 2 inference path is used for formal model output.

### Stage 4: Risk-controlled post-call arbitration (secondary)

After PathFinder has been called, a second lightweight model decides whether its prediction should replace ReasonEval.

The arbitration study compares:

* unconditional acceptance of PathFinder;
* overall PathFinder score, gate, and prediction;
* fine-grained mathematical and consistency signals;
* all available PathFinder signals.

A validation-calibrated override threshold will be used to study the trade-off between coverage and harmful replacement risk. The final evaluation will report both routing accuracy and the proportion of incorrect overrides.

PathFinder features are used only after the call and therefore do not leak into the pre-call routing decision.

## Models

### Base verifier

* `GAIR/ReasonEval-7B`
* One discriminative forward pass
* Feasibility threshold: `0.96`, selected on the separate 20-example pilot
* Formal threshold: selected using the validation split only

### Primary second-stage verifier

* `declare-lab/PathFinder-PRM-7B`
* Hierarchical discriminative PRM
* Two-pass gated scoring
* BF16 with Flash Attention 2

### Alternative second-stage baseline / optional extension

* `GenPRM/GenPRM-1.5B` was used in the completed feasibility study.
* The current feasibility implementation uses a simplified single-path
  verification wrapper and does not reproduce the full official GenPRM
  inference procedure.
* `GenPRM/GenPRM-7B` is the preferred candidate for an optional formal
  ReasonEval-to-GenPRM verifier-pair extension if time and compute permit.
* An official-path GenPRM-7B deployment smoke test has passed on a single
  NVIDIA A40; see
  [`docs/results/genprm_7b_official_smoke.md`](../results/genprm_7b_official_smoke.md).
* Before any formal GenPRM-7B comparison, the Hugging Face model revision must
  be pinned and the official inference semantics must be adapted and validated
  on PRMBench examples.
* GenPRM remains an optional comparison rather than the primary cascade.

## Dataset

Dataset: `hitsmy/PRMBench_Preview`

Completed development subsets:

* threshold-selection pilot: 10 correct and 10 erroneous steps;
* feasibility subset: 50 correct and 50 erroneous steps;
* feasibility random seed: `2026`.

The current 100 examples remain a frozen development pilot. These records and all other records derived from the same normalized `original_question` groups are excluded from every formal split.

New data will be split approximately into:

* training: 600 examples;
* validation: 200 examples;
* held-out test: 400 examples.

The primary formal sample will contain approximately equal numbers of correct and erroneous steps while preserving error-type coverage as closely as possible.

All splits must be grouped by normalized `original_question` so that variants of the same mathematical problem cannot cross dataset boundaries.

## Difference from Existing Work

Dynamic verification and model cascading are established research directions. Dyve combines fast and slow process verification, FlexiVe dynamically allocates generative verification compute, and CAMEL uses confidence-gated reflection. General LLM cascades similarly escalate selected examples from weaker to stronger models.

This project therefore does not claim to be the first dynamic verifier, confidence gate, expected-gain router, or model cascade.

Instead, it studies whether the realized correction utility between independently trained, heterogeneous process verifiers is predictable. The compared verifiers include scalar discriminative, structured discriminative, and generative PRMs.

The intended contributions are:

* a controlled analysis of beneficial, neutral, and harmful replacements between independently trained PRMs;
* comparison of base-verifier failure prediction with pair-specific benefit and expected-utility routing;
* cost-aware routing using a leakage-free pre-call cost predictor trained against same-hardware PathFinder runtimes;
* mechanism analysis showing where beneficial and harmful calls arise across ReasonEval confidence, reasoning-step position, error type, and input length;
* an oracle-versus-learnability decomposition that quantifies how much available correction headroom the learned router captures;
* risk-controlled post-call arbitration for verifier disagreement as a secondary component;
* grouped held-out evaluation and, if feasible, verifier-pair or cross-dataset extensions.

RouteGuard provides closely related motivation for distinguishing oracle complementarity from learnable routing gain and for guarding against small-sample overestimation. This project applies those concerns specifically to step-level process verification and additionally studies post-call evidence from PathFinder.

The contribution is an empirical and system-level study rather than a new PRM architecture.

## Completed Feasibility Evidence

The 100-example pilot produced the following exploratory results. Verifier accuracies are measured on the frozen pilot, while router and arbitration metrics are grouped out-of-fold estimates computed on the same pilot. They are not results from a separate held-out test set.

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

### Pilot post-call arbitration

The overall-signal arbitration model achieved:

* benefit average precision: `0.798`;
* harm average precision: `0.720`;
* accuracy of `0.85` at 75% and 100% PathFinder-call budgets.

Fine-grained signals did not outperform the simpler overall feature set in this pilot. They are therefore retained as a predefined ablation rather than a primary positive claim.

### Expanded 200-example feasibility

A subsequent development-only feasibility study used the first 200 records of the frozen formal training split (170 original-question groups) and left validation and test untouched. ReasonEval accuracy at the development threshold of 0.50 was 0.680, PathFinder accuracy was 0.815, and the full pairwise oracle upper bound was 0.865. Across five grouped-CV seeds, mean expected-gain-router accuracy was 0.774 at the primary 20% PathFinder budget and 0.827 at 40%, with mean benefit average precision of 0.629. These results are development evidence only and are not final held-out claims.

The expanded study is considered complete. No additional feasibility-scale GPU inference is required before formal train-and-validation inference.

A CPU-only pre-formal ablation on the same inspected 200-example development subset was used to finalize the lightweight Router specification before formal protocol freeze. Minimal nonlinear runtime terms did not improve grouped-CV PathFinder runtime prediction, so the linear Ridge cost predictor was retained. Direct and factorized utility alternatives also did not improve final routing. The only adopted refinement was to replace raw `prefix_characters` with exact `disprm_input_tokens` in the nine-feature Router; see [`docs/results/prm_router_preformal_ablations.md`](../results/prm_router_preformal_ablations.md). No further feature or model-family tuning will be performed on this 200-example subset.

## Formal Experiment Plan

The formal evaluation will be separated from the completed 100-example feasibility study.

### 1. Freeze the evaluation protocol

Before running the formal experiment, fix:

* the verifier models and prompting procedures;
* the candidate pre-call feature set;
* the routing budgets and utility definition;
* the post-call arbitration feature sets;
* the evaluation metrics and random seeds.

PRMBench error-type annotations will be used only for stratification and analysis, not as router inputs.

### 2. Construct grouped data splits

Create approximately:

* 600 training examples;
* 200 validation examples;
* 400 held-out test examples.

All variants derived from the same normalized `original_question` must remain in the same split. The splits should preserve step-label and error-type distributions as closely as possible.

The existing 100-example feasibility set will not be included in the final held-out test set.

### 3. Run verifier inference

Run ReasonEval and PathFinder on the same formal examples. Formal runtime measurements for the primary study must be collected on the same GPU using the same measurement protocol.

If an optional GenPRM verifier-pair extension is activated, its inference is run separately under the pre-specified extension protocol and does not alter the primary ReasonEval--PathFinder evaluation.

For every example, save:

* verifier scores and predictions;
* PathFinder fine-grained signals;
* input-token counts;
* inference runtime;
* the ground-truth step label.

### 4. Train and select models

Use the training split to fit:

* base-verifier failure prediction;
* benefit-only routing;
* harm-aware expected-gain routing;
* a lightweight PathFinder runtime predictor using pre-call features only;
* post-call arbitration for the secondary analysis.

Use the validation split only to select:

* the ReasonEval decision threshold;
* utility weights `lambda_h` and `mu`;
* any post-call arbitration threshold used in the secondary analysis.

Router and cost-predictor model hyperparameters are pre-specified before formal inference; the Ridge cost predictor uses `alpha = 1.0`.

After these choices are frozen, evaluate once on the held-out test split.

### 5. Report held-out results

Report accuracy and compute trade-offs across fixed budgets, together with group-bootstrap confidence intervals. Results will be compared both at equal PathFinder call budgets and using measured same-hardware sequential cascade runtime.

Pre-specified diagnostic analyses will report beneficial and harmful replacement rates by ReasonEval-confidence bin, reasoning-step-position bin, PRMBench error type, and input-length bin. Error-type annotations are analysis-only and are never router inputs. The report will also include an oracle-learnability decomposition, including the fraction of available pairwise oracle headroom captured by the learned router.

An additional ProcessBench evaluation and a formal ReasonEval-to-GenPRM pair comparison are optional and will be attempted only after the primary PRMBench evaluation is complete.

## Baselines

All routing methods will use the same verifier outputs, grouped data splits, and evaluation budgets.

### Single-verifier baselines

* **ReasonEval only:** use the base verifier for every example.
* **PathFinder only:** use PathFinder for every example.
* **GenPRM only:** use GenPRM for every example when the comparison run is available.

### Pre-call routing baselines

* **Random routing:** select examples uniformly at random.
* **Low-score routing:** call the second verifier for the lowest ReasonEval scores.
* **Uncertainty routing:** prioritize examples closest to the calibrated ReasonEval threshold.
* **Failure prediction:** predict whether ReasonEval is wrong without modeling whether the second verifier can correct it.
* **Benefit-only routing:** predict `P(beneficial)`.
* **Expected-gain routing:** rank by `P(beneficial) - P(harmful)`.
* **Cost-aware utility routing:** rank by the proposed risk- and cost-adjusted utility using predicted pre-call PathFinder cost; realized PathFinder runtime is used only for evaluation.

Comparing failure prediction with benefit and utility routing tests whether pair-specific correction information is more useful than merely detecting difficult examples.

### Post-call arbitration baselines

* **Always replace:** always accept the second-stage prediction after it is called.
* **Score-based arbitration:** accept the replacement using a validation-selected PathFinder score threshold.
* **Learned arbitration:** predict whether replacing ReasonEval is beneficial using the available post-call signals.
* **Risk-controlled arbitration:** accept a replacement only when its estimated harmful-replacement risk is below a validation-selected level.

### Upper bounds

* **Oracle pre-call routing:** prioritize all beneficial calls before neutral or harmful calls.
* **Oracle arbitration:** retain whichever verifier is correct for each example.

Oracle results are diagnostic upper bounds and are not deployable methods.

## Success Criteria

### Primary evaluation

The primary comparison will use the held-out PRMBench test set at a fixed 20% PathFinder call budget.

The main confirmatory question is whether pair-specific expected-gain routing achieves higher test accuracy than:

* ReasonEval-only verification;
* uncertainty routing;
* base-verifier failure prediction.

The paired differences and group-bootstrap confidence intervals will be reported. Cost-aware utility, measured accuracy-runtime Pareto curves, other routing budgets, post-call arbitration, and mechanism analyses are pre-specified secondary analyses rather than opportunities to redefine the primary test result after seeing the held-out data.

### Formal post-call arbitration criterion

Post-call arbitration will be considered useful if it reduces harmful replacements relative to always accepting PathFinder while retaining a comparable level of beneficial replacements.

The evaluation will report:

* accepted replacement coverage;
* beneficial replacement rate;
* harmful replacement rate;
* final cascade accuracy;
* risk-coverage curves.

### Minimum technical success

The project must provide:

* reproducible grouped train, validation, and test splits;
* leakage-free router training and model selection;
* same-hardware runtime measurements;
* call-budget and compute-aware comparisons;
* runnable scripts and documented configurations.

### Interpretation of negative results

A positive result would show that pair-specific utility routing improves the accuracy-compute trade-off beyond confidence and failure-prediction baselines.

A negative result would still be informative if the evaluation demonstrates that verifier complementarity exists but cannot be predicted reliably on held-out data, or that post-call arbitration cannot safely identify harmful replacements. In that case, the project will quantify the gap between oracle complementarity and learnable routing and analyze the conditions under which routing fails.

Therefore, project completion does not depend on reproducing the positive gains observed in the 100-example feasibility study.

## Expected Compute

The primary formal experiment requires inference from ReasonEval-7B and PathFinder-PRM-7B on approximately 1,200 PRMBench examples.

A single NVIDIA A40, A100, or comparable GPU with 40-80 GB VRAM should be sufficient. The models will be loaded and evaluated sequentially rather than kept in memory simultaneously.

The compute plan is:

* run ReasonEval once on all fixed examples;
* run PathFinder once on the same examples using its official two-pass scoring procedure;
* cache all verifier outputs and reuse them for router experiments;
* train lightweight routing and arbitration models locally on CPU;
* use the GPU again only if the dataset, verifier configuration, or inference procedure changes.

Router training, cross-validation, bootstrap evaluation, and threshold selection require negligible compute compared with verifier inference.

GenPRM will be retained as an alternative verifier baseline. A larger official multi-generation and code-execution evaluation will be performed only if time and compute permit; it is not required for completing the primary ReasonEval-PathFinder study.

Formal runtime comparisons will be measured on the same GPU. Runtime measurements from the earlier RTX 4060 and A40 pilot runs will not be combined into a single cost comparison.

Model checkpoints are expected to require tens of gigabytes of persistent storage. Generated JSONL outputs are comparatively small and will remain excluded from Git.

## Main Risks

### 1. Small numbers of beneficial and harmful calls

Even with 1,200 examples, the number of router-positive cases may remain limited.

Mitigation:

* use lightweight regularized models;
* avoid unnecessarily large feature sets;
* report group-bootstrap confidence intervals;
* report the number of beneficial and harmful examples in every split.

### 2. Leakage between related problems

PRMBench may contain multiple modified processes derived from the same original mathematical question.

Mitigation:

* split data by normalized `original_question`;
* keep every related variant in one split;
* verify automatically that no group crosses split boundaries.

### 3. Validation overfitting

Repeatedly changing features, thresholds, or utility weights after observing test results would invalidate the held-out evaluation.

Mitigation:

* freeze the protocol before running the test evaluation;
* select thresholds and hyperparameters using training and validation data only;
* evaluate the held-out test set once after model selection.

### 4. Threshold sensitivity

The ReasonEval threshold changes which calls are labeled beneficial or harmful. The pilot threshold of `0.96` may not transfer to the formal data.

Mitigation:

* recalibrate the threshold using the formal validation split;
* report a validation-based sensitivity analysis;
* never select the threshold using held-out test performance.

### 5. Dataset artifacts and limited generalization

The balanced PRMBench subset may not reflect a natural error distribution, and some modified reasoning steps may contain lexical clues related to correctness.

Mitigation:

* exclude error-type labels from router inputs;
* inspect performance by error type and relevant lexical patterns;
* report both balanced and naturally sampled results if time permits;
* use ProcessBench as an optional external evaluation.

### 6. Unreliable compute comparison

Runtime measured on different GPUs or with different attention implementations is not directly comparable.

Mitigation:

* rerun the primary verifiers on the same GPU;
* use the same warm-up and timing protocol;
* report call budgets separately from measured runtime or compute budgets.

### 7. A stronger verifier may dominate routing

If PathFinder is sufficiently accurate and inexpensive, always using it may be preferable to a cascade.

Mitigation:

* compare against PathFinder-only verification;
* report the full accuracy-cost Pareto curve;
* treat this outcome as evidence about when routing is or is not worthwhile.

### 8. Rare harmful replacements

Post-call arbitration may be difficult to learn if very few harmful replacements occur.

Mitigation:

* keep the arbitration model lightweight;
* report risk-coverage curves rather than only classification accuracy;
* treat post-call arbitration as a secondary contribution if the sample size is insufficient.

### 9. Simplified GenPRM implementation

The current GenPRM feasibility wrapper does not reproduce the complete official multi-generation and code-execution pipeline.

Mitigation:

* use PathFinder as the primary second-stage verifier;
* describe GenPRM as an alternative baseline;
* avoid making conclusions about the full official GenPRM system unless it is reproduced.

## Current Finalization Status

The primary formal experiment is complete and the held-out protocol is closed to further tuning.

Completed milestones include:

1. deterministic grouped 600/200/400 formal splits;
2. same-environment ReasonEval and PathFinder inference;
3. validation-only ReasonEval threshold selection;
4. validation-only cost-aware utility selection;
5. protocol freeze before test access;
6. one-time held-out evaluation on 400 examples;
7. grouped-bootstrap primary comparisons;
8. archival of formal artifacts in `cl-prm-team/cl-prm-artifacts`;
9. preservation of the executed A100 workflow in the formal run notebook.

Remaining project work is descriptive analysis, visualization, reproducibility cleanup, and final report writing. Optional GenPRM work must remain separate from the completed primary result and must not trigger retuning of the ReasonEval--PathFinder protocol.

## Decision

**ReasonEval--PathFinder selective verification is the completed primary project direction.**

The held-out study confirms that:

* ReasonEval and PathFinder retain substantial complementary error structure;
* beneficial and harmful replacements both occur on unseen data;
* selective expected-gain routing improves over ReasonEval-only and uncertainty routing at the pre-specified 20% budget;
* the 20% selective cascade also exceeds the PathFinder-only point estimate while using PathFinder on only one fifth of examples;
* base-verifier failure prediction is a strong learned baseline, so the incremental advantage of pair-specific expected gain over that baseline should be interpreted cautiously;
* cost-aware routing is most useful as a secondary accuracy--runtime analysis rather than as a separate primary accuracy win.

No further feature, threshold, utility-weight, or primary-budget selection will use the held-out test set.

GenPRM remains an optional post-primary extension. Preliminary simplified GenPRM runs are not treated as final results unless a corrected extension protocol is frozen and evaluated separately.

## Relevant Resources

### Primary models and benchmarks

* [ReasonEval paper](https://arxiv.org/abs/2404.05692)
* [ReasonEval-7B model](https://huggingface.co/GAIR/ReasonEval-7B)
* [PathFinder-PRM paper](https://arxiv.org/abs/2505.19706)
* [PathFinder-PRM official code](https://github.com/declare-lab/PathFinder-PRM)
* [PathFinder-PRM-7B model](https://huggingface.co/declare-lab/PathFinder-PRM-7B)
* [GenPRM paper](https://arxiv.org/abs/2504.00891)
* [GenPRM official code](https://github.com/RyanLiu112/GenPRM)
* [PRMBench paper](https://arxiv.org/abs/2501.03124)
* [PRMBench official code](https://github.com/ssmisya/PRMBench)
* [PRMBench Preview dataset](https://huggingface.co/datasets/hitsmy/PRMBench_Preview)
* [ProcessBench paper](https://arxiv.org/abs/2412.06559)

### Related dynamic verification and routing work

* [Dyve: Thinking Fast and Slow for Dynamic Process Verification](https://arxiv.org/abs/2502.11157)
* [Solve-Detect-Verify: Inference-Time Scaling with Flexible Generative Verifier](https://arxiv.org/abs/2505.11966)
* [CAMEL: Confidence-Gated Reflection for Reward Modeling](https://arxiv.org/abs/2602.20670)
* [RouteGuard: Certifying Routing Gain When Complementarity Is Not Enough](https://arxiv.org/abs/2608.07583)
