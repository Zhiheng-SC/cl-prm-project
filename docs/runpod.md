# RunPod environment

This document records the cloud setup verified for ReasonEval and PathFinder
inference. It is intentionally separate from the Windows-oriented
`requirements.txt`.

## Verified stack

| Component | Version |
|---|---|
| Python | 3.12.x (verified with 3.12.3) |
| PyTorch | 2.8.0+cu128 |
| CUDA reported by PyTorch | 12.8 |
| FlashAttention | 2.8.3.post1 |
| Transformers | 4.48.2 |
| Datasets | 4.8.5 |
| Accelerate | 1.2.0 |
| GPU | NVIDIA A40, 48 GB |
| Precision | BF16 |

PyTorch and FlashAttention are ABI-coupled binary packages. The verified
RunPod PyTorch image supplies both. Do not run `pip install -r
requirements.txt` on this image because that file pins the Windows/local
PyTorch build and can replace the working cloud binary stack.

## Recommended Pod configuration

- GPU: one NVIDIA A40;
- image: the same official RunPod PyTorch image providing the verified stack;
- container disk: 20--30 GB;
- volume disk: approximately 80 GB;
- mount: keep the repository, Hugging Face cache, generated data, and outputs
  under `/workspace`;
- do not attach a network volume unless Pods must be recreated frequently.

Record the exact image identifier shown in RunPod's template details before
formal inference. A saved template preserves configuration, but it is not a
replacement for a versioned Docker image or the bootstrap verification.

## Secrets and environment variables

Create RunPod secrets for the Hugging Face and GitHub tokens. Reference them
from template environment variables rather than storing tokens in the
repository or shell history.

```text
HF_TOKEN={{ RUNPOD_SECRET_huggingface_token }}
GH_TOKEN={{ RUNPOD_SECRET_github_token }}
HF_HOME=/workspace/.cache/huggingface
HF_HUB_ENABLE_HF_TRANSFER=0
PYTHONUNBUFFERED=1
```

Use read-only or minimum-scope tokens where possible.

## First startup

From the Pod terminal:

```bash
cd /workspace

gh auth status
hf auth whoami

gh repo clone Zhiheng-SC/cl-prm-project
cd cl-prm-project

bash scripts/bootstrap_runpod.sh
```

If the repository already exists:

```bash
cd /workspace/cl-prm-project
git switch main
git fetch origin --prune
git merge --ff-only origin/main
bash scripts/bootstrap_runpod.sh
```

The bootstrap script performs the following operations:

1. verifies Python, PyTorch, CUDA, GPU, and FlashAttention;
2. executes a real BF16 FlashAttention CUDA kernel;
3. installs only the packages in `requirements-runpod.txt` and the local `cl_prm` package in editable mode;
4. verifies that the binary stack still works;
5. runs the lightweight inference-IO, formal-runner, and evaluation-helper tests;
6. previews the formal inference plan when the frozen split files exist.

The script never starts model inference.

## Data preparation

If generated data are absent, reconstruct them after bootstrap:

```bash
export HF_HOME=/workspace/.cache/huggingface
export HF_HUB_ENABLE_HF_TRANSFER=0

python experiments/feasibility/prm_router/prepare_subset.py \
    --n-correct 50 \
    --n-error 50 \
    --seed 2026 \
    --output data/prm_router/feasibility_100.jsonl

python experiments/formal/prm_router/prepare_formal_splits.py \
    --config configs/experiments/prm_router_formal.json \
    --output-dir data/prm_router/formal
```

Then verify the guarded plan:

```bash
python experiments/formal/prm_router/run_formal_inference.py
```

The default command is plan-only and must end with:

```text
Plan only; no model inference was started.
```

Do not add `--execute` until the team has confirmed the common
GPU/environment and approved the formal development run. Do not unlock the
test phase before the validation-selected protocol is frozen.

## Stop and termination policy

Keep all recoverable working state under `/workspace`.

- Stop a Pod only for a short interruption when retaining its volume disk and
  model cache is worth the storage charge.
- Terminate after outputs and metadata are archived locally and the next run is
  not imminent.
- A volume disk survives a stop but is deleted on termination.
- A network volume survives termination but continues to incur storage cost.
- RunPod storage is not the authoritative long-term backup.

Before termination, verify output row counts, metadata status and SHA256, then
archive the generated outputs and metadata in the private
`cl-prm-team/cl-prm-artifacts` repository. The Git commit, frozen
configuration, and tracked split manifest remain in GitHub.
