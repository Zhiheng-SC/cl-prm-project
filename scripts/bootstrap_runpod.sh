#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f requirements-runpod.txt ]]; then
    echo "Run this script from the cl-prm-project repository root." >&2
    exit 1
fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

EXPECTED_TORCH="${CL_PRM_EXPECTED_TORCH:-2.8.0+cu128}"
EXPECTED_FLASH_ATTN="${CL_PRM_EXPECTED_FLASH_ATTN:-2.8.3.post1}"
EXPECTED_GPU="${CL_PRM_EXPECTED_GPU:-NVIDIA A40}"

verify_binary_stack() {
    CL_PRM_EXPECTED_TORCH="${EXPECTED_TORCH}" \
    CL_PRM_EXPECTED_FLASH_ATTN="${EXPECTED_FLASH_ATTN}" \
    CL_PRM_EXPECTED_GPU="${EXPECTED_GPU}" \
    python - <<'PY'
import os
import platform

import torch
import flash_attn
from flash_attn import flash_attn_func

expected_torch = os.environ["CL_PRM_EXPECTED_TORCH"]
expected_flash_attn = os.environ["CL_PRM_EXPECTED_FLASH_ATTN"]
expected_gpu = os.environ["CL_PRM_EXPECTED_GPU"]

if not platform.python_version().startswith("3.12."):
    raise RuntimeError(
        "Expected Python 3.12.x, got " + platform.python_version()
    )
if torch.__version__ != expected_torch:
    raise RuntimeError(
        f"Expected torch {expected_torch}, got {torch.__version__}. "
        "Do not install requirements.txt on this Pod."
    )
if flash_attn.__version__ != expected_flash_attn:
    raise RuntimeError(
        f"Expected flash-attn {expected_flash_attn}, "
        f"got {flash_attn.__version__}."
    )
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

gpu_name = torch.cuda.get_device_name(0)
if gpu_name != expected_gpu:
    raise RuntimeError(
        f"Expected GPU {expected_gpu!r}, got {gpu_name!r}."
    )

query = torch.randn(
    1,
    128,
    4,
    64,
    device="cuda",
    dtype=torch.bfloat16,
)
output = flash_attn_func(query, query, query)
torch.cuda.synchronize()

print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("FlashAttention:", flash_attn.__version__)
print("GPU:", gpu_name)
print("Kernel:", tuple(output.shape), output.dtype, output.device)
PY
}

echo "Checking the image-provided CUDA binary stack..."
verify_binary_stack

echo "Installing RunPod-safe project dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements-runpod.txt
python -m pip install -e . --no-deps
python -m pip check

echo "Rechecking the CUDA binary stack after installation..."
verify_binary_stack

echo "Running lightweight repository tests..."
python tests/test_inference_io.py -v
python tests/test_formal_inference.py -v
python tests/test_formal_evaluation.py -v

if [[ -f data/prm_router/formal/train.jsonl ]] \
    && [[ -f data/prm_router/formal/validation.jsonl ]] \
    && [[ -f data/prm_router/formal/split_manifest.json ]]; then
    echo "Checking the guarded formal plan..."
    python experiments/formal/prm_router/run_formal_inference.py
else
    echo "Formal split files are absent; guarded plan check skipped."
    echo "Reconstruct the data using the commands in docs/runpod.md."
fi

echo
echo "RUNPOD ENVIRONMENT READY"
echo "No model inference was started."
