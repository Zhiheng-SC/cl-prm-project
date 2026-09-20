# GenPRM-7B official-path smoke test

## Scope

This record establishes only that `GenPRM/GenPRM-7B` can be loaded and run on
one NVIDIA A40 through the upstream GenPRM vLLM inference path, including its
generated-code verification stage. It is a deployment feasibility check, not
an accuracy result and not part of the frozen formal PRM Router evaluation.

The existing `run_genprm.py` remains the separate 1.5B Transformers-based
feasibility implementation. This smoke test does not replace its results or
silently change the alternative verifier in the formal configuration.

## Observed environment

Test date: 2026-09-19.

| Component | Observed value |
|---|---|
| GPU | NVIDIA A40 |
| GPU memory reported by `nvidia-smi` | 46,068 MiB |
| Driver | 570.211.01 |
| Host CUDA capability reported by `nvidia-smi` | 12.8 |
| Python environment | Python 3.10 virtual environment at `/opt/genprm-env` |
| PyTorch | 2.5.1+cu124 |
| CUDA reported by PyTorch | 12.4 |
| vLLM | 0.7.1 |
| Transformers | 4.48.2 |
| Upstream repository target | `RyanLiu112/GenPRM` at `a08da3f6b636be370e0d53f9bdbdc455cdece939` |
| Model | `GenPRM/GenPRM-7B` |

The Hugging Face model revision was not explicitly pinned in this exploratory
run. A future formal comparison must record and pin the resolved snapshot SHA.

`HF_HUB_ENABLE_HF_TRANSFER=1` was present in the Pod. The first attempt failed
because `hf_transfer` was not installed; installing that package resolved the
download error.

## Observed result

| Measurement | Value |
|---|---:|
| Model-load wall time, including first download | 94.69 s |
| Model weights reported by vLLM | 14.2717 GiB |
| GPU memory after load | 40,127 / 46,068 MiB |
| One-example inference wall time | 6.01 s |
| GPU memory after inference | 40,171 / 46,068 MiB |
| Reward | 1.0 |
| Parsed judgement in generated output | `Yes` |

The model produced analysis, generated and executed a Python check that
returned `5050`, and emitted `\\boxed{Yes}`. The remaining GPU-memory headroom
after this example was about 5.8 GiB. An NCCL process-group cleanup warning was
printed at interpreter exit; it occurred after the successful result and did
not invalidate the smoke test.

## Reproduction on a new server

From the `cl-prm-project` repository root:

```bash
bash scripts/bootstrap_genprm_official.sh
source /opt/genprm-env/bin/activate
python scripts/smoke_genprm_7b.py --allow-generated-code
```

The bootstrap script pins the upstream code revision, creates or reuses the
separate Python 3.10 environment, installs the upstream requirements and
`hf_transfer`, and validates CUDA plus package versions. Model weights remain
in `HF_HOME`, which defaults to `/workspace/.cache/huggingface`.

The official inference path executes Python emitted by the model. Run it only
inside an isolated disposable container with no unnecessary credentials or
sensitive mounted data. The explicit command-line flag is intentional.

## Decision

GenPRM-7B is technically feasible on a single A40 and can remain an optional
comparison extension. The primary study remains ReasonEval-to-PathFinder.
Before any formal GenPRM-7B experiment, the team must separately decide its
scope, pin the Hugging Face snapshot, adapt the dataset runner to the official
semantics, and budget the additional generative inference cost.
