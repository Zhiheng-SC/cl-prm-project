# PRM Router Feasibility Study

## Status

**Feasibility decision: passed. Final team selection is pending.**

This preliminary study confirms that:

1. ReasonEval, GenPRM, and PathFinder inference pipelines are runnable.
2. ReasonEval and PathFinder exhibit meaningful complementary errors.
3. PathFinder is the strongest second-stage candidate in the current pilot.
4. A lightweight expected-gain router can identify beneficial calls.
5. Post-call trust arbitration can detect many harmful replacements.
6. Random and uncertainty routing leave substantial room for improvement.

These results support project selection but are exploratory and must not be interpreted as final held-out benchmark results.

## Research Question

Can process verification achieve a better accuracy-compute trade-off by first applying a base discriminative PRM and selectively invoking a stronger second-stage verifier only when it is expected to improve the decision?

The feasibility study examines GenPRM and PathFinder-PRM as possible second-stage verifiers.

The proposed pipeline is:

1. Run ReasonEval on every reasoning step.
2. Extract features available before calling the second-stage verifier.
3. Use a lightweight router to estimate the expected benefit of the call.
4. Invoke the second-stage verifier only for selected examples under a fixed budget.
5. Optionally arbitrate whether to accept its judgement after observing its output.
6. Compare accuracy with single-model, random, and uncertainty-routing baselines.

## Models

### Discriminative PRM

* Model: `GAIR/ReasonEval-7B`
* Frozen revision: `0a6556ef5c937bb17d265ba681b501fd60056cfe`
* Output: scalar score indicating whether the current reasoning step is correct
* Inference type: discriminative forward pass

The default threshold of `0.50` was poorly calibrated on the pilot set. A diagnostic threshold of `0.96` was selected using the 20-example pilot experiment and then fixed for the 100-example feasibility experiment.

### Generative PRM

* Final prediction: normalized `Yes/No` probability with threshold `0.50`
* The sampled judgement text is stored separately from the probability-based prediction
* Model: `GenPRM/GenPRM-1.5B`
* Frozen revision: `a0fa69768f4524257e1730fec639aa7781c7fa82`
* Output: generated analysis followed by a `Yes/No` judgement
* Number of generations: one per example
* Local mode: analysis without model-generated Python code execution

The current Windows-compatible implementation is a simplified feasibility version. It does not yet reproduce the complete official GenPRM pipeline with vLLM, iterative code execution, and majority voting.

### Structured PathFinder PRM

* Model: `declare-lab/PathFinder-PRM-7B`
* Frozen revision: `84a7412511836cb4ed74377d9c703eb5638d814c`
* Output: mathematical-reasoning, consistency, and final-correctness signals
* Inference type: official two-pass gated scoring procedure
* Attention implementation: Flash Attention 2
* Precision: BF16
* Evaluation hardware: NVIDIA A40 with 48 GB VRAM

PathFinder was added as an alternative second-stage verifier because its structured signals may support both pre-call routing and post-call trust arbitration.

## Dataset

Dataset: `hitsmy/PRMBench_Preview`

Two balanced subsets were used:

* Pilot set: 10 correct and 10 erroneous step prefixes
* Feasibility set: 50 correct and 50 erroneous step prefixes
* Feasibility-set random seed: `2026`

Each example contains:

* the mathematical question;
* the reasoning prefix up to the evaluated step;
* the current step;
* the current-step ground-truth label;
* step-position and length metadata.

The balanced subsets are intended only for feasibility analysis. Their class distribution does not represent the natural distribution of PRMBench.

Generated datasets are stored locally under `data/prm_router/` and are excluded from Git.

## Repository Files

```text
experiments/feasibility/prm_router/
├── README.md
├── prepare_subset.py
├── run_disprm.py
├── run_genprm.py
├── run_pathfinder.py
├── inference_io.py
├── run_formal_inference.py
├── analyze_complementarity.py
├── evaluate_routing.py
├── train_benefit_router.py
├── analyze_threshold_sensitivity.py
├── analyze_grouped_cv.py
├── train_expected_gain_router.py
└── evaluate_pathfinder_cascade.py
```

| File                               | Purpose                                                                                |
| ---------------------------------- | -------------------------------------------------------------------------------------- |
| `prepare_subset.py`                | Create balanced PRMBench step-prefix subsets                                           |
| `run_disprm.py`                    | Run ReasonEval and save its scores and predictions                                     |
| `run_genprm.py`                    | Run the Windows-compatible GenPRM feasibility inference                                |
| `run_pathfinder.py`                | Run the official two-pass PathFinder scoring procedure                                 |
| `inference_io.py`                  | Validate resume state and record reproducibility metadata                              |
| `run_formal_inference.py`          | Validate formal splits and plan or execute guarded verifier inference                  |
| `analyze_complementarity.py`       | Identify beneficial, harmful, and shared verifier outcomes                             |
| `evaluate_routing.py`              | Evaluate random, low-score, uncertainty, and oracle routing                            |
| `train_benefit_router.py`          | Train and evaluate a logistic-regression benefit router with out-of-fold predictions   |
| `analyze_threshold_sensitivity.py` | Repeat out-of-fold routing evaluation across multiple DisPRM thresholds                |
| `analyze_grouped_cv.py`            | Compare standard and original-question-grouped out-of-fold router evaluation           |
| `train_expected_gain_router.py`    | Compare benefit-only and harm-aware expected-gain routing with grouped OOF predictions |
| `evaluate_pathfinder_cascade.py`   | Evaluate pre-call PathFinder routing and post-call trust arbitration                   |

## Experimental Pipeline

```text
PRMBench step prefix
        |
        v
ReasonEval score and prediction
        |
        v
Pre-call expected-utility router
        |
        v
PathFinder structured verification
        |
        v
Optional post-call trust arbitration
```

GenPRM is retained as an alternative second-stage comparison baseline.

The second-stage replacement target is defined as:

```text
beneficial = ReasonEval is wrong and the second-stage verifier is correct
neutral    = both verifiers have the same correctness
harmful    = ReasonEval is correct and the second-stage verifier is wrong
```

The pre-call router never includes the ground-truth label, second-stage prediction, second-stage score, post-call PathFinder signals, or PRMBench error-type classification.

## Router Features

The initial lightweight router uses:

* ReasonEval score;
* distance from the ReasonEval decision threshold;
* ReasonEval binary prediction;
* normalized step position;
* current step number;
* total number of steps;
* question length;
* reasoning-prefix length;
* current-step length.

The model is logistic regression with feature standardization and class-balanced training.

## Environment Setup

The experiments used Python 3.10.

Install the PyTorch build appropriate for the local CUDA version first. For the Windows CUDA 12.4 environment used in this feasibility study:

```bash
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Reproduction

The three verifier inference scripts accept `--resume`. When an output
already exists, this option validates that it is an exact input prefix and
that the model revision and critical inference settings match before
appending the remaining examples.

Each new inference output also receives a sibling `.metadata.json` file.
It records input and output hashes, the script hash, model revision, command,
Git state, inference settings, timestamps, and the software and GPU
environment. The metadata run signature must also match when resuming.

Formal runtime measurements use three untimed warmup examples, exclude model
loading, and synchronize CUDA around each measured inference region. All
verifiers used in a direct runtime comparison must run on the same GPU.

### Prepare the feasibility subset

```bash
python experiments/feasibility/prm_router/prepare_subset.py --n-correct 50 --n-error 50 --seed 2026 --output data/prm_router/feasibility_100.jsonl
```

### Run ReasonEval

```bash
python experiments/feasibility/prm_router/run_disprm.py --input data/prm_router/feasibility_100.jsonl --output outputs/prm_router/reasoneval_feasibility_100.jsonl --limit 100 --threshold 0.96
```

### Run GenPRM

```bash
python experiments/feasibility/prm_router/run_genprm.py --input data/prm_router/feasibility_100.jsonl --output outputs/prm_router/genprm_feasibility_100.jsonl --limit 100 --max-input-tokens 4096
```

### Run PathFinder

PathFinder requires a Linux CUDA environment with Flash Attention 2 for the official inference path.

```bash
python experiments/feasibility/prm_router/run_pathfinder.py --input data/prm_router/feasibility_100.jsonl --output outputs/prm_router/pathfinder/pathfinder_feasibility_100.jsonl --limit 100 --attention-implementation flash_attention_2
```

### Analyze verifier complementarity

```bash
python experiments/feasibility/prm_router/analyze_complementarity.py --disprm outputs/prm_router/reasoneval_feasibility_100.jsonl --genprm outputs/prm_router/genprm_feasibility_100.jsonl
```

### Evaluate heuristic routing

```bash
python experiments/feasibility/prm_router/evaluate_routing.py
```

### Train the benefit-aware router

```bash
python experiments/feasibility/prm_router/train_benefit_router.py
```

### Train the harm-aware expected-gain router

```bash
python experiments/feasibility/prm_router/train_expected_gain_router.py
```

### Train the ReasonEval-to-PathFinder expected-gain router

```bash
python experiments/feasibility/prm_router/train_expected_gain_router.py --disprm outputs/prm_router/reasoneval_feasibility_100.jsonl --second-stage outputs/prm_router/pathfinder/pathfinder_feasibility_100.jsonl --second-stage-kind pathfinder --output outputs/prm_router/robustness/pathfinder_expected_gain_router.json
```

### Evaluate the PathFinder cascade

```bash
python experiments/feasibility/prm_router/evaluate_pathfinder_cascade.py
```

### Analyze DisPRM threshold sensitivity

```bash
python experiments/feasibility/prm_router/analyze_threshold_sensitivity.py
```

### Analyze original-question-grouped cross-validation

```bash
python experiments/feasibility/prm_router/analyze_grouped_cv.py
```

## Preliminary Results

ReasonEval and GenPRM were evaluated locally on an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB VRAM. PathFinder was evaluated separately on an NVIDIA A40 with 48 GB VRAM. Runtime ratios across these different devices are not treated as comparable compute measurements.

### Individual Verifiers

| Model                        | Accuracy | Mean runtime per example |
| ---------------------------- | -------: | -----------------------: |
| ReasonEval, threshold `0.96` |     0.66 |                 2.0189 s |
| GenPRM-1.5B                  |     0.74 |                 3.0894 s |

GenPRM inference alone was approximately `1.53×` slower than ReasonEval. In a cascade, ReasonEval is always executed first, so the total cost also includes GenPRM inference for routed examples.

### Complementarity

| Outcome                          | Count |
| -------------------------------- | ----: |
| Both verifiers correct           |    48 |
| ReasonEval correct, GenPRM wrong |    18 |
| ReasonEval wrong, GenPRM correct |    26 |
| Both verifiers wrong             |     8 |

The 26 cases where ReasonEval was wrong and GenPRM was correct are potentially beneficial calls.

The 18 cases where ReasonEval was correct and GenPRM was wrong are potentially harmful calls. Their presence shows why calling GenPRM indiscriminately is not optimal.

A ground-truth oracle router would obtain an upper-bound accuracy of `0.92`. This oracle is not an implementable method because it uses the true outcomes after inference. It only quantifies the available verifier complementarity.

### Composition of Beneficial Calls

| Benefit type                                       | Count |
| -------------------------------------------------- | ----: |
| Correct steps rescued after ReasonEval rejection   |    24 |
| Erroneous steps caught after ReasonEval acceptance |     2 |

Most of the observed benefit comes from rescuing correct steps that ReasonEval incorrectly classified as erroneous.

Two cases demonstrate the more difficult complementary behavior in which GenPRM corrects a highly confident false acceptance by ReasonEval:

| Example                                 | Step | ReasonEval score |
| --------------------------------------- | ---: | ---------------: |
| `domain_inconsistency_prm_train_p1_220` |   13 |           0.9975 |
| `deception_prm_train_p1_125`            |   14 |           0.9947 |

These two cases show that high-confidence complementary errors exist, but the current sample is too small to claim that they are common.

## Heuristic Routing Results

| GenPRM budget | Low-score routing | Uncertainty routing | Random routing | Oracle upper bound | Cascade cost |
| ------------: | ----------------: | ------------------: | -------------: | -----------------: | -----------: |
|            0% |              0.66 |                0.66 |          0.660 |               0.66 |        1.00× |
|           10% |              0.62 |                0.69 |          0.668 |               0.76 |        1.15× |
|           20% |              0.60 |                0.67 |          0.678 |               0.86 |        1.31× |
|           30% |              0.61 |                0.67 |          0.683 |               0.92 |        1.46× |
|           40% |              0.64 |                0.70 |          0.694 |               0.92 |        1.61× |
|           50% |              0.69 |                0.71 |          0.700 |               0.92 |        1.77× |
|           75% |              0.72 |                0.80 |          0.719 |               0.92 |        2.15× |
|          100% |              0.74 |                0.74 |          0.740 |               0.92 |        2.53× |

Low-score routing performs poorly at small budgets. Uncertainty routing provides some gains but is inconsistent and remains far below the oracle upper bound.

## Learned Benefit Router

The logistic-regression router was evaluated using five-fold stratified out-of-fold predictions.

For every example, its reported router probability was produced by a model trained on the other four folds. Therefore, no example was evaluated by a router trained on that same example.

### Benefit-Prediction Metrics

| Metric                            | Result |
| --------------------------------- | -----: |
| Benefit-positive examples         |     26 |
| Benefit-negative examples         |     74 |
| Positive rate                     |   0.26 |
| OOF ROC-AUC                       | 0.9033 |
| OOF average precision             | 0.8141 |
| Random average-precision baseline | 0.2600 |

### Budgeted Routing Performance

| GenPRM budget | Learned router | Uncertainty | Random | Oracle | Beneficial/harmful calls | Cascade cost |
| ------------: | -------------: | ----------: | -----: | -----: | -----------------------: | -----------: |
|            0% |           0.66 |        0.66 |  0.660 |   0.66 |                    0 / 0 |        1.00× |
|           10% |           0.73 |        0.69 |  0.668 |   0.76 |                    8 / 1 |        1.15× |
|           20% |           0.79 |        0.67 |  0.678 |   0.86 |                   16 / 3 |        1.31× |
|           30% |           0.81 |        0.67 |  0.683 |   0.92 |                   21 / 6 |        1.46× |
|           40% |           0.80 |        0.70 |  0.694 |   0.92 |                   23 / 9 |        1.61× |
|           50% |           0.82 |        0.71 |  0.700 |   0.92 |                   25 / 9 |        1.77× |
|           75% |           0.80 |        0.80 |  0.719 |   0.92 |                  25 / 11 |        2.15× |
|          100% |           0.74 |        0.74 |  0.740 |   0.92 |                  26 / 18 |        2.53× |

At a 20% GenPRM budget, the learned router obtains an accuracy of `0.79`, compared with:

* `0.67` for uncertainty routing;
* approximately `0.678` for random routing;
* `0.66` for ReasonEval alone;
* `0.74` for GenPRM alone.

At this budget, the learned router selects 16 beneficial calls and only 3 harmful calls.

The best observed learned-routing accuracy is `0.82` at a 50% GenPRM budget. The non-monotonic budget curve is expected because additional GenPRM calls may include harmful replacements.

## DisPRM Threshold Sensitivity

To test whether the routing result depends excessively on the DisPRM decision threshold, the complete out-of-fold router evaluation was repeated at six thresholds. The table below reports routing accuracy under a 20% GenPRM-call budget.

| DisPRM threshold | DisPRM only | Beneficial / harmful calls | Router ROC-AUC | Average precision | Learned routing | Uncertainty routing | Random routing | Oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.63 | 20 / 9 | 0.873 | 0.691 | 0.75 | 0.71 | 0.652 | 0.83 |
| 0.80 | 0.74 | 10 / 10 | 0.617 | 0.153 | 0.75 | 0.76 | 0.740 | 0.84 |
| 0.90 | 0.69 | 17 / 12 | 0.833 | 0.539 | 0.76 | 0.77 | 0.701 | 0.86 |
| 0.95 | 0.71 | 20 / 17 | 0.854 | 0.629 | 0.76 | 0.69 | 0.717 | 0.91 |
| 0.96 | 0.66 | 26 / 18 | 0.903 | 0.814 | 0.79 | 0.67 | 0.675 | 0.92 |
| 0.98 | 0.59 | 32 / 17 | 0.856 | 0.670 | 0.71 | 0.64 | 0.620 | 0.91 |

Verifier complementarity persists across all tested thresholds: every threshold produces beneficial GenPRM calls, and the oracle-routing accuracy ranges from `0.83` to `0.92`.

At a 20% budget, the learned router improves over the corresponding DisPRM-only baseline at all six thresholds. It outperforms uncertainty routing at four of the six thresholds, but is lower by `0.01` at thresholds `0.80` and `0.90`. Therefore, the existence of useful routing opportunities appears robust, while the magnitude of the learned router's advantage remains threshold-sensitive.

The threshold `0.96` was selected using the separate 20-example pilot subset before evaluating the 100-example feasibility subset. It was not selected by optimizing performance on these 100 examples. This sensitivity sweep is diagnostic and does not replace a final held-out evaluation.

## Original-Question Grouped Evaluation

The 100-example subset contains `96` unique original-question groups. Four original questions occur twice, accounting for eight records. Under the original `StratifiedKFold` evaluation, three of these four duplicated groups were split across training and test folds.

The router was therefore reevaluated using `StratifiedGroupKFold`, with normalized PRMBench `original_question` text as the group identifier. This ensures that all variants of the same original problem remain in the same fold.

| Evaluation method | Duplicated groups split across folds | ROC-AUC | Average precision | Learned accuracy at 10% | Learned accuracy at 20% | Learned accuracy at 30% |
|---|---:|---:|---:|---:|---:|---:|
| Standard stratified OOF | 3 | 0.903 | 0.814 | 0.73 | 0.79 | 0.81 |
| Grouped by original question | 0 | 0.899 | 0.780 | 0.72 | 0.78 | 0.80 |

Removing all detected cross-fold original-question overlap reduces the learned router's accuracy by only `0.01` at each tested budget. At the 20% budget, grouped routing still achieves `0.78`, compared with `0.67` for uncertainty routing and `0.66` for the DisPRM-only baseline.

This suggests that the preliminary benefit-prediction signal is not primarily explained by duplicated-question leakage. However, grouped out-of-fold evaluation on the same 100 examples is still not equivalent to evaluation on a separate held-out dataset.

## Harm-Aware Expected-Gain Extension

A three-class router was evaluated to distinguish beneficial, neutral, and harmful GenPRM calls:

```text
+1 = DisPRM is wrong and GenPRM is correct
 0 = both verifiers have the same correctness
-1 = DisPRM is correct and GenPRM is wrong
```

The routing score is defined as:

```text
expected gain = P(beneficial) - P(harmful)
```

The expected-gain router and the original benefit-only router were evaluated using the same five original-question-grouped out-of-fold splits.

| Metric | Result |
|---|---:|
| Beneficial calls | 26 |
| Neutral calls | 56 |
| Harmful calls | 18 |
| Benefit-only benefit AP | 0.7481 |
| Expected-gain benefit AP | 0.7329 |
| Expected-gain harm AP | 0.2926 |
| Random harm AP baseline | 0.1800 |

| GenPRM budget | Expected-gain router | Benefit-only router | Expected gain B/H | Benefit only B/H |
|---:|---:|---:|---:|---:|
| 10% | 0.72 | 0.73 | 8 / 2 | 8 / 1 |
| 20% | 0.78 | 0.78 | 14 / 2 | 15 / 3 |
| 30% | 0.78 | 0.80 | 17 / 5 | 19 / 5 |
| 40% | 0.82 | 0.82 | 21 / 5 | 23 / 7 |
| 50% | 0.81 | 0.80 | 22 / 7 | 24 / 10 |

The harm-aware router detects harmful calls above the random baseline and sometimes selects fewer harmful calls. However, it does not consistently improve routing accuracy over the benefit-only router because avoiding harmful calls may also exclude beneficial calls.

This result suggests that changing the prediction target alone is insufficient. More informative error-specific features, such as separate mathematical-correctness and logical-consistency signals, may be required to predict harmful GenPRM calls reliably. This motivates the proposed PathFinder-PRM extension.

## PathFinder Extension Results

PathFinder was evaluated on the same 100-example feasibility subset using its official two-pass gated scoring procedure with Flash Attention 2.

### Verifier accuracy and complementarity

| Pair outcome | Examples |
|---|---:|
| Both ReasonEval and PathFinder correct | 55 |
| ReasonEval correct, PathFinder wrong | 11 |
| ReasonEval wrong, PathFinder correct | 26 |
| Both wrong | 8 |

| Verifier | Accuracy |
|---|---:|
| ReasonEval at threshold 0.96 | 0.66 |
| GenPRM | 0.74 |
| PathFinder | 0.81 |
| ReasonEval-PathFinder oracle | 0.92 |

PathFinder corrects 26 ReasonEval errors while damaging 11 initially correct predictions. This creates a meaningful selective-routing opportunity.

By contrast, PathFinder and GenPRM have only two examples where PathFinder is wrong and GenPRM is correct. Their oracle accuracy is `0.83`, compared with PathFinder-only accuracy of `0.81`. Therefore, PathFinder-to-GenPRM routing is not supported as the primary cascade by this pilot.

### Pre-call expected-gain routing

The pre-call router uses only ReasonEval and input-derived features. PathFinder outputs are used to construct training targets, but are unavailable when deciding whether to make the call.

| PathFinder budget | Expected-gain router | Benefit-only router | Uncertainty router | Random mean | Oracle |
|---:|---:|---:|---:|---:|---:|
| 10% | 0.76 | 0.75 | 0.72 | 0.675 | 0.76 |
| 20% | 0.82 | 0.80 | 0.71 | 0.691 | 0.86 |
| 30% | 0.82 | 0.82 | 0.71 | 0.705 | 0.92 |
| 40% | 0.84 | 0.82 | 0.74 | 0.721 | 0.92 |

The grouped out-of-fold expected-gain router obtains benefit average precision `0.8210` and harm average precision `0.2990`. At a 20% PathFinder-call budget, it selects 17 beneficial and one harmful call.

### Post-call trust arbitration

After PathFinder is called, a second model predicts whether its judgement should replace the ReasonEval judgement. Three post-call feature sets were compared.

| Feature set | Benefit AP | Harm AP |
|---|---:|---:|
| Overall PathFinder signals | 0.798 | 0.720 |
| Fine-grained signals | 0.796 | 0.623 |
| All signals | 0.772 | 0.675 |

| PathFinder budget | Always accept | Overall arbitration | Fine-grained arbitration | All-signal arbitration | Oracle |
|---:|---:|---:|---:|---:|---:|
| 10% | 0.76 | 0.76 | 0.76 | 0.76 | 0.76 |
| 20% | 0.82 | 0.82 | 0.82 | 0.82 | 0.86 |
| 40% | 0.84 | 0.83 | 0.82 | 0.82 | 0.92 |
| 75% | 0.84 | 0.85 | 0.84 | 0.84 | 0.92 |
| 100% | 0.81 | 0.85 | 0.84 | 0.84 | 0.92 |

Post-call arbitration predicts harmful replacements substantially better than their `0.11` prevalence baseline. It does not improve low-budget routing, but the overall-signal version reaches `0.85` accuracy at 75% and 100% call budgets.

The fine-grained PathFinder signals do not outperform the simpler overall signals in this pilot. They should therefore be treated as an ablation or an inconclusive negative result, not as an established improvement.

## Preliminary Interpretation

The feasibility study supports the following conclusions:

1. ReasonEval and PathFinder exhibit stronger complementarity than the tested PathFinder-GenPRM pairing.
2. ReasonEval-to-PathFinder expected-gain routing is the most promising cascade found in the pilot.
3. Learned pre-call routing substantially outperforms random and uncertainty routing on the current subset.
4. Post-call trust arbitration can identify many harmful PathFinder replacements, especially at high call budgets.
5. Fine-grained PathFinder signals do not provide an incremental improvement over overall signals on these 100 examples.
6. GenPRM remains a useful baseline, but it does not increase the three-verifier oracle beyond the ReasonEval-PathFinder oracle in this pilot.

The strongest supported project framing is therefore pre-call benefit-aware routing followed by optional post-call trust arbitration between complementary process verifiers.
## Why This Counts as Feasibility Evidence

The purpose of this stage is to decide whether the research direction is technically and empirically promising.

The current experiment confirms:

* all three public verifiers and PRMBench can be loaded;
* ReasonEval, GenPRM, and PathFinder inference pipelines can run;
* verifier outputs can be aligned on identical reasoning steps;
* meaningful complementary errors occur;
* random, uncertainty, and learned routing baselines can be implemented;
* expected-gain routing learns a non-trivial grouped out-of-fold signal;
* post-call PathFinder arbitration can predict harmful replacements;
* the required cloud inference is affordable at feasibility scale.

A separate held-out evaluation is not required for project selection. It is required before treating the reported performance as a generalization result.
## Limitations

* In 5 of 100 feasibility examples, the sampled judgement text differed from the probability-based prediction. Reported accuracy uses the probability-based prediction.
* The feasibility experiment contains only 100 balanced examples.
* The class distribution does not represent the natural PRMBench distribution.
* Router performance is based on out-of-fold predictions rather than a separate held-out dataset.
* The feature set and budget analysis were inspected using the same feasibility dataset.
* The `0.96` ReasonEval threshold was selected using a small 20-example pilot set.
* Benefit labels depend on one stochastic GenPRM generation per example.
* The local GenPRM version does not execute generated verification code.
* ReasonEval and GenPRM have different parameter counts.
* Runtime was measured on one Windows laptop GPU.
* Most beneficial calls are false-negative corrections.
* Statistical significance and cross-dataset generalization have not been established.
* Oracle routing uses ground-truth outcomes and is only an upper bound.
* Only 11 harmful ReasonEval-to-PathFinder replacements are available for training and evaluating the harm predictor.
* PathFinder and the earlier verifiers were timed on different GPUs, so the current cross-model runtime ratios are not valid compute comparisons.
* PathFinder results were inspected on the same 100-example development subset used for feature and method exploration.

## Formal Evaluation Preparation

The formal evaluation setup now completed after this feasibility study includes:

1. freezing the 100-example set as a development pilot;
2. excluding its 96 original-question groups from formal data;
3. creating grouped splits with 600 training, 200 validation, and 400 held-out test examples;
4. pinning the PRMBench and verifier revisions;
5. recording dataset provenance, split statistics, and SHA256 hashes in a tracked reference manifest;
6. adding revision-pinned and resumable inference, sidecar metadata, a same-hardware timing protocol, and a guarded formal inference runner.

The runner defaults to a plan-only development pass over train and validation:

```bash
python experiments/feasibility/prm_router/run_formal_inference.py
```

Add `--execute` only on the selected common GPU when the team is ready to
start formal development inference. Test execution is separately locked and
requires both a validation-selected ReasonEval threshold and
`--confirm-test-protocol-frozen`.

If the team selects PRM Router, the remaining formal work is:

1. run ReasonEval and PathFinder on train and validation using the same GPU;
2. select thresholds, features, utility weights, and budgets without test access;
3. freeze the routing and arbitration protocol;
4. unlock and evaluate once on the held-out test set;
5. report accuracy-budget and accuracy-latency curves with group-bootstrap confidence intervals and paired significance tests;
6. retain GenPRM as an alternative second-stage baseline if time and compute permit;
7. evaluate a more natural benchmark distribution or ProcessBench only as a secondary extension.


## Relevant Resources

* [PathFinder-PRM paper](https://arxiv.org/abs/2505.19706)
* [PathFinder-PRM official repository](https://github.com/declare-lab/PathFinder-PRM)
* [PathFinder-PRM-7B checkpoint](https://huggingface.co/declare-lab/PathFinder-PRM-7B)
* [GenPRM paper](https://arxiv.org/abs/2504.00891)
* [GenPRM official repository](https://github.com/RyanLiu112/GenPRM)
* [GenPRM-1.5B checkpoint](https://huggingface.co/GenPRM/GenPRM-1.5B)
* [ReasonEval-7B checkpoint](https://huggingface.co/GAIR/ReasonEval-7B)
* [PRMBench Preview](https://huggingface.co/datasets/hitsmy/PRMBench_Preview)

## Feasibility Decision

**Passed — promising enough for project selection.**

The current experiment demonstrates verifier complementarity, measurable routing benefit, and a substantial gap between heuristic and learned routing. The direction is technically feasible and empirically promising.

The reported values remain exploratory. If the team selects this direction, the next decisive step is a frozen held-out evaluation.
