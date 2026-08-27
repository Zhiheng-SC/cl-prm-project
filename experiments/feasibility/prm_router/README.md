# PRM Router Feasibility Study

## Status

Preliminary feasibility experiment completed on 27 August 2026.

The current results suggest that a discriminative PRM and GenPRM make complementary errors, leaving room for a learned benefit-aware router. These results are exploratory and are not yet final benchmark results.

## Research Question

Can we improve the accuracy–compute trade-off of process verification by first applying a discriminative PRM and selectively invoking a slower generative PRM only when it is expected to improve the decision?

The proposed pipeline is:

1. Run a discriminative PRM on every reasoning step.
2. Use a lightweight router to estimate whether GenPRM is likely to correct the discriminative PRM.
3. Invoke GenPRM only for selected examples under a fixed computational budget.
4. Compare accuracy and runtime with single-model and heuristic-routing baselines.

## Models

### Discriminative PRM

* Model: `GAIR/ReasonEval-7B`
* Output: scalar probability that the current reasoning step is correct
* Inference type: single discriminative forward pass

The default threshold of `0.50` was poorly calibrated on the pilot data. A diagnostic threshold of `0.96` was selected using the 20-example pilot set and then applied to the separate 100-example feasibility set.

### Generative PRM

* Model: `GenPRM/GenPRM-1.5B`
* Output: generated analysis followed by a `Yes/No` judgement
* Sampling: one generation per example
* Local mode: analysis without model-generated Python code execution

The local Windows implementation is a simplified feasibility version. It does not yet reproduce the full official GenPRM inference procedure with vLLM, iterative code verification, and majority voting.

## Dataset

Dataset: `hitsmy/PRMBench_Preview`

Two balanced subsets were prepared:

* Pilot set: 10 correct and 10 erroneous step prefixes
* Feasibility set: 50 correct and 50 erroneous step prefixes
* Feasibility seed: `2026`

Each record contains the question, reasoning prefix up to the current step, current-step label, step position, and related metadata.

The generated datasets are stored locally under `data/prm_router/` and are not committed to Git.

## Repository Files

```text
experiments/feasibility/prm_router/
├── README.md
├── prepare_subset.py
├── run_disprm.py
├── run_genprm.py
├── analyze_complementarity.py
└── evaluate_routing.py
```

| File                         | Purpose                                                                             |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| `prepare_subset.py`          | Create balanced PRMBench step-prefix subsets                                        |
| `run_disprm.py`              | Run ReasonEval and save scores and predictions                                      |
| `run_genprm.py`              | Run the local GenPRM feasibility inference                                          |
| `analyze_complementarity.py` | Compare model errors and identify beneficial and harmful GenPRM calls               |
| `evaluate_routing.py`        | Evaluate random, low-score, uncertainty, and oracle routing under different budgets |

## Reproduction

Create the 100-example feasibility subset:

```bash
python experiments/feasibility/prm_router/prepare_subset.py --n-correct 50 --n-error 50 --seed 2026 --output data/prm_router/feasibility_100.jsonl
```

Run ReasonEval:

```bash
python experiments/feasibility/prm_router/run_disprm.py --input data/prm_router/feasibility_100.jsonl --output outputs/prm_router/reasoneval_feasibility_100.json --limit 100
```

Run GenPRM:

```bash
python experiments/feasibility/prm_router/run_genprm.py --input data/prm_router/feasibility_100.jsonl --output outputs/prm_router/genprm_feasibility_100.json --limit 100 --max-input-tokens 4096
```

Analyze complementarity:

```bash
python experiments/feasibility/prm_router/analyze_complementarity.py --disprm outputs/prm_router/reasoneval_feasibility_100.json --genprm outputs/prm_router/genprm_feasibility_100.json
```

Evaluate routing baselines:

```bash
python experiments/feasibility/prm_router/evaluate_routing.py
```

## Preliminary Results

The experiments were run locally on an NVIDIA GeForce RTX 4060 Laptop GPU with 8 GB VRAM.

### Individual Models

| Model                        | Accuracy | Mean runtime per example |
| ---------------------------- | -------: | -----------------------: |
| ReasonEval, threshold `0.96` |     0.66 |                 2.0189 s |
| GenPRM-1.5B                  |     0.74 |                 3.0894 s |

GenPRM inference alone was approximately `1.53×` slower than ReasonEval. In the actual cascade, ReasonEval is always executed first, so the total cost also includes GenPRM inference for routed examples.

### Complementarity

| Outcome                          | Number of examples |
| -------------------------------- | -----------------: |
| Both models correct              |                 48 |
| ReasonEval correct, GenPRM wrong |                 18 |
| ReasonEval wrong, GenPRM correct |                 26 |
| Both models wrong                |                  8 |

The 26 cases where ReasonEval was wrong and GenPRM was correct are potentially beneficial calls. The 18 reverse cases are potentially harmful calls.

A ground-truth oracle router would obtain an upper-bound accuracy of `0.92`. This oracle result is not an achievable method because it uses the true label after inference; it only measures the available complementarity.

### Budgeted Routing

| GenPRM budget | Low-score routing | Uncertainty routing | Random routing | Oracle upper bound | Estimated cascade cost |
| ------------: | ----------------: | ------------------: | -------------: | -----------------: | ---------------------: |
|            0% |              0.66 |                0.66 |          0.660 |               0.66 |                  1.00× |
|           10% |              0.62 |                0.69 |          0.668 |               0.76 |                  1.15× |
|           20% |              0.60 |                0.67 |          0.678 |               0.86 |                  1.31× |
|           30% |              0.61 |                0.67 |          0.683 |               0.92 |                  1.46× |
|           40% |              0.64 |                0.70 |          0.694 |               0.92 |                  1.61× |
|           50% |              0.69 |                0.71 |          0.700 |               0.92 |                  1.77× |
|           75% |              0.72 |                0.80 |          0.719 |               0.92 |                  2.15× |
|          100% |              0.74 |                0.74 |          0.740 |               0.92 |                  2.53× |

## Preliminary Interpretation

The results provide three main observations:

1. GenPRM is more accurate than ReasonEval on this subset, but it is slower.
2. Calling GenPRM on examples with the lowest ReasonEval scores performs poorly at small budgets.
3. Uncertainty routing provides limited and inconsistent gains and remains far below the oracle upper bound.

The gap between uncertainty routing and oracle routing motivates predicting the marginal benefit of invoking GenPRM rather than relying only on discriminative-model uncertainty.

The feasibility experiment therefore supports continuing with a lightweight benefit-aware router, while not yet establishing that such a router can learn the required distinction.

## Current Limitations

* The feasibility set contains only 100 balanced examples.
* The class distribution does not represent the natural PRMBench distribution.
* The ReasonEval threshold was selected on a small 20-example pilot set.
* GenPRM uses only one stochastic generation per example.
* The local GenPRM implementation does not execute generated verification code.
* ReasonEval and GenPRM have different parameter counts.
* Runtime was measured on one local Windows laptop GPU.
* Statistical significance has not yet been established.
* The oracle router uses ground-truth outcomes and is only an upper bound.

## Next Steps

1. Construct benefit labels:

   ```text
   positive = ReasonEval is wrong and GenPRM is correct
   negative = all other cases
   ```

2. Train a lightweight logistic-regression router using only information available before invoking GenPRM.

3. Use out-of-fold predictions to avoid evaluating the router on its own training examples.

4. Compare the learned router with random, low-score, and uncertainty baselines under the same GenPRM budgets.

5. Expand the dataset if the initial router results are promising.

6. Later evaluate the official GenPRM code-verification and majority-voting inference on Linux or an A100 GPU.

## Relevant Resources

* [GenPRM paper](https://arxiv.org/abs/2504.00891)
* [GenPRM official repository](https://github.com/RyanLiu112/GenPRM)
* [GenPRM-1.5B checkpoint](https://huggingface.co/GenPRM/GenPRM-1.5B)
* [ReasonEval-7B checkpoint](https://huggingface.co/GAIR/ReasonEval-7B)
* [PRMBench Preview](https://huggingface.co/datasets/hitsmy/PRMBench_Preview)

## Feasibility Decision

**Current decision: promising enough to continue.**

The two verifiers exhibit meaningful complementarity, and simple routing heuristics leave a substantial gap to the oracle upper bound. The next decisive experiment is whether a lightweight router can predict beneficial GenPRM calls better than uncertainty and random routing.
