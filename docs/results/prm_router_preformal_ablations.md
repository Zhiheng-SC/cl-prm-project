# Pre-formal PRM Router Ablations

## Status

**Development-only analysis completed on 2026-09-20.**

This analysis reuses the already inspected expanded 200-example subset from the
formal training pool. It is method-development evidence only: no formal
validation or held-out test data were accessed, and no new verifier inference
was run.

The purpose was to resolve three pre-formal design questions before freezing
the Router specification:

1. whether the lightweight PathFinder runtime predictor needs nonlinear terms;
2. whether the Router length representation can be improved without increasing
   model complexity;
3. whether rare harmful replacements are better handled by direct or
   factorized utility models.

The analysis uses five original-question-grouped CV seeds:

`7, 42, 2026, 20260903, 20260915`.

The 200-example development subset contains 170 original-question groups with
37 beneficial, 153 neutral, and 10 harmful ReasonEval-to-PathFinder calls.

## Cost predictor ablation

| Variant | MAE (s) | RMSE (s) | Spearman |
|---|---:|---:|---:|
| Current linear Ridge | 0.008156 +/- 0.000056 | 0.010114 +/- 0.000083 | 0.9797 +/- 0.0006 |
| + input tokens squared | 0.008368 +/- 0.000030 | 0.010673 +/- 0.000195 | 0.9788 +/- 0.0003 |
| + input tokens squared + tokens x step position | 0.008373 +/- 0.000047 | 0.011812 +/- 0.000561 | 0.9788 +/- 0.0002 |

The nonlinear basis expansions did not improve held-out grouped-CV runtime
prediction. The formal cost predictor therefore remains the pre-specified
`StandardScaler + Ridge(alpha=1.0)` model with its existing seven pre-call
features.

## Router feature ablation

The current nine-feature Router was compared with small additions and
substitutions using the same multinomial expected-gain estimator.

| Features | Benefit AP | Harm AP | Acc@20 | B/H@20 | Acc@40 |
|---|---:|---:|---:|---:|---:|
| Current 9 features | 0.6292 +/- 0.0261 | 0.1056 +/- 0.0201 | 0.7740 +/- 0.0058 | 21.20 / 2.40 | 0.8270 +/- 0.0024 |
| + ReasonEval positive probability | 0.6225 +/- 0.0293 | 0.1038 +/- 0.0204 | 0.7730 +/- 0.0060 | 21.00 / 2.40 | 0.8280 +/- 0.0040 |
| + exact input-token count | 0.6416 +/- 0.0265 | 0.1059 +/- 0.0165 | 0.7810 +/- 0.0080 | 22.00 / 1.80 | 0.8210 +/- 0.0058 |
| Replace prefix characters with exact input tokens | 0.6581 +/- 0.0139 | 0.1100 +/- 0.0146 | 0.7820 +/- 0.0051 | 22.20 / 1.80 | 0.8230 +/- 0.0051 |
| + positive probability + input tokens | 0.6353 +/- 0.0273 | 0.1032 +/- 0.0189 | 0.7790 +/- 0.0058 | 21.60 / 1.80 | 0.8200 +/- 0.0045 |
| + score x step-position interaction | 0.6324 +/- 0.0272 | 0.0985 +/- 0.0181 | 0.7790 +/- 0.0080 | 22.00 / 2.20 | 0.8210 +/- 0.0037 |

The most defensible lightweight refinement is to replace
`prefix_characters` with the already saved exact
`disprm_input_tokens` value. This keeps the Router at nine features, improves
the primary 20% development routing result and benefit AP, and avoids adding a
new model component.

The formal Router feature set is therefore frozen as:

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

## Utility-estimator ablation

Direct Ridge regression on gain labels `g in {-1, 0, +1}` underperformed the
multinomial expected-gain model. With the current feature set, 20% routing
accuracy fell from 0.774 to 0.759 and benefit AP from 0.629 to 0.558.

A factorized correction-utility model was also tested:

```text
P(B | x) = P(RE wrong | x) * P(PF correct | RE wrong, x)
P(H | x) = P(RE correct | x) * P(PF wrong | RE correct, x)
```

The factorized model exposed useful conditional structure but did not improve
final routing. With the token-replacement feature set, its mean 20% accuracy
was 0.774 compared with 0.782 for the multinomial model.

Component diagnostics for the token-replacement features were:

| Conditional task | Average precision |
|---|---:|
| ReasonEval failure | 0.6788 +/- 0.0112 |
| PathFinder correction given ReasonEval wrong | 0.7649 +/- 0.0322 |
| PathFinder harm given ReasonEval correct | 0.2100 +/- 0.0258 |

The harmful-replacement task remains data-limited: only 10 harmful examples are
available in this development subset. The conditional harm AP is above its rare
class prevalence, indicating some pre-call signal, but the factorized
probability decomposition does not improve the final utility ranking.

## Frozen development decision

After this analysis, no further feature or model-family tuning will be
performed on the expanded-200 development subset.

The formal specification retains:

- multinomial logistic regression for beneficial/neutral/harmful gain
  probabilities;
- expected gain based on `P(beneficial) - P(harmful)`;
- the nine-feature Router with exact ReasonEval input-token count replacing raw
  prefix-character count;
- the existing linear Ridge PathFinder runtime predictor;
- validation-only selection of the already pre-specified ReasonEval threshold,
  `lambda_h`, `mu`, and secondary arbitration threshold.

The held-out formal test remains locked until all validation-selected choices
and the analysis plan are frozen.
