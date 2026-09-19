# OVM Proposal

## Status

**Feasibility study completed and passed; retained as an archived alternative direction.**

The OVM study is not the selected primary project direction, but its completed feasibility results are kept for comparison and for lessons about inference-time compute. The implementation, configuration, and compact summaries remain in GitHub. Raw generated seed outputs and metadata are stored in the team's private Hugging Face artifact repository.

## Feasibility Summary

The frozen feasibility study used 100 GSM8K examples for each of three seeds (`41`, `42`, `43`) on an NVIDIA A100 80 GB GPU. Aggregated across seeds:

- greedy accuracy: `0.5700`;
- self-consistency accuracy: `0.6267`;
- ORM accuracy: `0.7967`;
- OVM accuracy: `0.8033`;
- OVM minus self-consistency: `+0.1767`;
- OVM minus ORM: `+0.0067`;
- vanilla candidate correctness: `0.4825`;
- last-expansion candidate correctness: `0.7597`;
- final-beam candidate correctness: `0.7907`;
- mean OVM runtime: `10.47 s`;
- mean sampling runtime: `15.19 s`.

The pre-specified feasibility checks passed. These results are feasibility evidence only and are not part of the PRM Router held-out formal evaluation.

See:

- `experiments/feasibility/ovm/output/ovm-eval.json` for the machine-readable compact summary;
- `experiments/feasibility/ovm/output/ovm-eval.md` for the human-readable summary;
- `experiments/feasibility/ovm/` for the runnable feasibility code and configuration.
