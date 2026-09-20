# Formal Result Template

Use this file as a compact record while formal results are produced. Values
should be copied only from verified formal artifacts.

## Provenance

| Item | Value |
|---|---|
| Git commit | TBD |
| GPU | TBD |
| Python | TBD |
| PyTorch | TBD |
| CUDA | TBD |
| FlashAttention | TBD |
| Dataset revision | `5cc7683d0ae5797f84d7aeac0607966f277c39e1` |
| ReasonEval revision | `0a6556ef5c937bb17d265ba681b501fd60056cfe` |
| PathFinder revision | `84a7412511836cb4ed74377d9c703eb5638d814c` |

## Formal split audit

| Split | Examples | Labels 0/1 | Original-question groups | Input SHA256 |
|---|---:|---:|---:|---|
| Train | 600 | 300 / 300 | 358 | TBD |
| Validation | 200 | 100 / 100 | 113 | TBD |
| Test | 400 | 200 / 200 | 242 | LOCKED until protocol freeze |

## Train + validation verifier summary

| Metric | Train | Validation |
|---|---:|---:|
| ReasonEval accuracy | TBD | TBD |
| PathFinder accuracy | TBD | TBD |
| Both correct | TBD | TBD |
| ReasonEval only correct / harmful replacement | TBD | TBD |
| PathFinder only correct / beneficial replacement | TBD | TBD |
| Both wrong | TBD | TBD |
| Pairwise oracle accuracy | TBD | TBD |
| ReasonEval median runtime (s) | TBD | TBD |
| PathFinder median runtime (s) | TBD | TBD |

## Validation-selected choices

| Choice | Selected value |
|---|---|
| ReasonEval threshold | TBD |
| `lambda_h` | TBD |
| `mu` | TBD |
| Secondary arbitration threshold/rule | TBD / not used |

## Frozen Router specification

Multinomial beneficial / neutral / harmful model.

Pre-call Router features:

```text
disprm_score
distance_to_threshold
disprm_predicted_correct
step_position
current_step
total_steps
question_characters
disprm_input_tokens
current_step_characters
```

Cost predictor:

```text
StandardScaler + Ridge(alpha=1.0)
```

Cost-predictor features:

```text
disprm_input_tokens
current_step
total_steps
step_position
question_characters
prefix_characters
current_step_characters
```

## Development routing results

Primary budget is 20%. Other budgets are secondary curve points.

| Method | Acc@10 | Acc@20 | Acc@30 | Acc@40 | Acc@50 | Acc@75 | Acc@100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ReasonEval only | — | TBD | — | — | — | — | — |
| Random | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Low-score | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Uncertainty | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Failure prediction | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Benefit-only | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Expected gain | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Cost-aware utility | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Oracle routing | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Primary held-out comparison

Fill this section only after protocol freeze and one-time test evaluation.

| Comparison at 20% PathFinder budget | Test accuracy | Paired delta vs expected gain | 95% group-bootstrap CI |
|---|---:|---:|---|
| ReasonEval only | TBD | TBD | TBD |
| Uncertainty routing | TBD | TBD | TBD |
| Failure prediction | TBD | TBD | TBD |
| Expected-gain routing | TBD | reference | TBD |

## Compute-aware evaluation

| Policy | PathFinder call rate | Final accuracy | Sequential cascade runtime | Relative runtime vs RE-only |
|---|---:|---:|---:|---:|
| ReasonEval only | 0% | TBD | TBD | 1.00x |
| Expected gain @ 20% | 20% | TBD | TBD | TBD |
| Cost-aware @ 20% | 20% | TBD | TBD | TBD |
| PathFinder only | 100% | TBD | TBD | TBD |

## Replacement accounting

| Split / policy | Beneficial calls | Neutral calls | Harmful calls |
|---|---:|---:|---:|
| Validation, expected gain @ 20% | TBD | TBD | TBD |
| Test, expected gain @ 20% | TBD | TBD | TBD |
| Test, cost-aware @ 20% | TBD | TBD | TBD |

## Diagnostic analyses

Record only pre-specified diagnostics:

- ReasonEval confidence bins: 5;
- reasoning-step-position bins: 4;
- input-length bins: 4;
- PRMBench error type: analysis only, never a Router input;
- oracle-versus-learnability decomposition;
- secondary post-call arbitration, if retained.

## Interpretation notes

Keep observations separated into:

**Observed formal result:** direct numerical finding from verified artifacts.

**Interpretation:** proposed explanation consistent with the observed result.

**Open question:** hypothesis or follow-up that is not established by the
current experiment.

This separation is especially important for small harmful-replacement counts.
