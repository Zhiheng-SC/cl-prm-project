# PRM Router Feasibility Study

## Status

**Feasibility decision: passed.**

This preliminary study confirms that:

1. the complete DisPRM–GenPRM inference pipeline is runnable;
2. the two verifiers make complementary errors;
3. invoking GenPRM can be either beneficial or harmful;
4. simple uncertainty routing leaves substantial room for improvement;
5. a lightweight logistic-regression router can learn a useful benefit signal in an out-of-fold feasibility evaluation.

These results are sufficient to support project selection. They are exploratory and must not be interpreted as final held-out benchmark results.

## Research Question

Can process verification achieve a better accuracy–compute trade-off by first applying a discriminative PRM and selectively invoking a generative PRM only when the second verifier is expected to improve the decision?

The proposed pipeline is:

1. Run a discriminative PRM on every reasoning step.
2. Extract features available before calling GenPRM.
3. Use a lightweight router to estimate the benefit of calling GenPRM.
4. Invoke GenPRM only for selected examples under a fixed budget.
5. Compare the resulting accuracy and cost with single-model and heuristic-routing baselines.

## Models

### Discriminative PRM

* Model: `GAIR/ReasonEval-7B`
* Output: scalar score indicating whether the current reasoning step is correct
* Inference type: discriminative forward pass

The default threshold of `0.50` was poorly calibrated on the pilot set. A diagnostic threshold of `0.96` was selected using the 20-example pilot experiment and then fixed for the 100-example feasibility experiment.

### Generative PRM

* Model: `GenPRM/GenPRM-1.5B`
* Output: generated analysis followed by a `Yes/No` judgement
* Number of generations: one per example
* Local mode: analysis without model-generated Python code execution

The current Windows-compatible implementation is a simplified feasibility version. It does not yet reproduce the complete official GenPRM pipeline with vLLM, iterative code execution, and majority voting.

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
├── analyze_complementarity.py
├── evaluate_routing.py
└── train_benefit_router.py
```

| File                         | Purpose                                                                              |
| ---------------------------- | ------------------------------------------------------------------------------------ |
| `prepare_subset.py`          | Create balanced PRMBench step-prefix subsets                                         |
| `run_disprm.py`              | Run ReasonEval and save its scores and predictions                                   |
| `run_genprm.py`              | Run the Windows-compatible GenPRM feasibility inference                              |
| `analyze_complementarity.py` | Identify beneficial, harmful, and shared verifier outcomes                           |
| `evaluate_routing.py`        | Evaluate random, low-score, uncertainty, and oracle routing                          |
| `train_benefit_router.py`    | Train and evaluate a logistic-regression benefit router with out-of-fold predictions |

## Experimental Pipeline

```text
PRMBench step prefix
        |
        v
ReasonEval score and prediction
        |
        +----------------------+
        |                      |
        v                      v
Router features          GenPRM judgement
        |                      |
        +----------+-----------+
                   |
                   v
       Benefit label and routing evaluation
```

The benefit target is defined as:

```text
positive = ReasonEval is wrong and GenPRM is correct
negative = all other cases
```

The router input never includes the ground-truth label, GenPRM prediction, GenPRM score, or PRMBench error-type classification.

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

### Prepare the feasibility subset

```bash
python experiments/feasibility/prm_router/prepare_subset.py --n-correct 50 --n-error 50 --seed 2026 --output data/prm_router/feasibility_100.jsonl
```

### Run ReasonEval

```bash
python experiments/feasibility/prm_router/run_disprm.py --input data/prm_router/feasibility_100.jsonl --output outputs/prm_router/reasoneval_feasibility_100.json --limit 100
```

### Run GenPRM

```bash
python experiments/feasibility/prm_router/run_genprm.py --input data/prm_router/feasibility_100.jsonl --output outputs/prm_router/genprm_feasibility_100.json --limit 100 --max-input-tokens 4096
```

### Analyze verifier complementarity

```bash
python experiments/feasibility/prm_router/analyze_complementarity.py --disprm outputs/prm_router/reasoneval_feasibility_100.json --genprm outputs/prm_router/genprm_feasibility_100.json
```

### Evaluate heuristic routing

```bash
python experiments/feasibility/prm_router/evaluate_routing.py
```

### Train the benefit-aware router

```bash
python experiments/feasibility/prm_router/train_benefit_router.py
```

## Preliminary Results

All experiments were run locally on an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB VRAM.

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

## Preliminary Interpretation

The feasibility results support the following observations:

1. ReasonEval and GenPRM exhibit meaningful complementary errors.
2. GenPRM is more accurate but slower on the current balanced subset.
3. Calling GenPRM can improve or damage the final prediction.
4. Low-score and uncertainty heuristics do not reliably identify the most beneficial calls.
5. Simple observable features contain a strong benefit-prediction signal in the current out-of-fold experiment.
6. A lightweight learned router can outperform both uncertainty and random routing under the same GenPRM budget.

Most learned-routing benefit currently comes from identifying correct steps rejected by ReasonEval. The experiment provides only two examples of GenPRM correcting highly confident false acceptance. Therefore, the strongest supported research framing is predicting the marginal benefit of invoking GenPRM, rather than focusing exclusively on high-confidence DisPRM errors.

## Why This Counts as Feasibility Evidence

The purpose of this stage is to decide whether the research direction is technically and empirically promising.

The current experiment confirms:

* the required public models and dataset can be loaded;
* both models can run on available hardware;
* their outputs can be aligned on the same reasoning steps;
* complementary errors occur;
* computational costs can be measured;
* heuristic baselines can be implemented;
* a lightweight router can learn a non-trivial signal.

A separate held-out evaluation is not required to make the project-selection decision. It is required before treating the reported router performance as a final generalization result.

## Limitations

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

## Next Steps If This Direction Is Selected

If the team selects PRM Router as the final project direction, the next experiments should be:

1. Freeze the current feature set and routing protocol.
2. Create a separate held-out evaluation set.
3. Train the router on one split and evaluate it once on the held-out split.
4. Report confidence intervals and paired significance tests.
5. Perform feature ablations.
6. Compare different discriminative PRMs and matched model sizes.
7. Evaluate multiple GenPRM samples and majority voting.
8. Run the official GenPRM code-verification pipeline on Linux or an A100 GPU.
9. Evaluate on a naturally distributed benchmark split.
10. Compare accuracy under equal measured compute budgets.

These steps belong to the full project and are not required before choosing among the three candidate topics.

## Relevant Resources

* [GenPRM paper](https://arxiv.org/abs/2504.00891)
* [GenPRM official repository](https://github.com/RyanLiu112/GenPRM)
* [GenPRM-1.5B checkpoint](https://huggingface.co/GenPRM/GenPRM-1.5B)
* [ReasonEval-7B checkpoint](https://huggingface.co/GAIR/ReasonEval-7B)
* [PRMBench Preview](https://huggingface.co/datasets/hitsmy/PRMBench_Preview)

## Feasibility Decision

**Passed — promising enough for project selection.**

The current experiment demonstrates verifier complementarity, measurable routing benefit, and a substantial gap between heuristic and learned routing. The direction is technically feasible and empirically promising.

The reported values remain exploratory. If the team selects this direction, the next decisive step is a frozen held-out evaluation.
