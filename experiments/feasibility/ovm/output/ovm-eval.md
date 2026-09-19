# OVM Feasibility Results

**Decision: PASS**

Pilot Study: 100 GSM8K questions x 3 seeds
## Average Accuracy


| Method | Accuracy |
|---:|---:|
| Greedy | 57.00% |
| Self-consistency (K=20) | 62.67% |
| ORM post-selection (K=20) | 79.67% |
| OVM guided (K=20, b=10) | 80.33% |
## Seed accuracy


| Seed | Greedy | SC | ORM | OVM |
|---:|---:|---:|---:|---:|
| 41 | 57.00% | 61.00% | 83.00% | 79.00% |
| 42 | 57.00% | 64.00% | 78.00% | 80.00% |
| 43 | 57.00% | 63.00% | 78.00% | 82.00% |
## Feasibility gates


| Requirement | Observed | Threshold | Result |
|---|---:|---:|:---:|
| `minimum_completed_seeds` | 3 | 3 | PASS |
| `ovm_minus_self_consistency_accuracy` | 0.1767 | 0.05 | PASS |
| `no_seed_worse_than_self_consistency` | 0.16 | 0 | PASS |
| `ovm_minus_greedy_accuracy` | 0.2333 | 0.15 | PASS |
| `last_expansion_minus_vanilla_correct_fraction` | 0.2772 | 0.15 | PASS |
| `ovm_minus_orm_accuracy` | 0.006667 | -0.02 | PASS |
| `ovm_to_sampling_runtime_ratio` | 0.6895 | 3 | PASS |
| `maximum_peak_allocated_gib` | 38.9 | 78 | PASS |

## Observability

- Vanilla sample correct fraction: 48.25%
- OVM last-expansion correct fraction: 75.97%
- OVM final-beam correct fraction: 79.07%
- OVM / sampling runtime ratio: 0.69
- Maximum peak allocated CUDA memory: 38.90 GiB
