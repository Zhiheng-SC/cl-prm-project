#!/usr/bin/env bash
set -euo pipefail

GENPRM_ROOT="${GENPRM_ROOT:-/workspace/GenPRM}"
GENPRM_VENV="${GENPRM_VENV:-/opt/genprm-env}"
GENPRM_UPSTREAM_COMMIT="${GENPRM_UPSTREAM_COMMIT:-a08da3f6b636be370e0d53f9bdbdc455cdece939}"
HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"

export HF_HOME
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ ! -d "${GENPRM_ROOT}/.git" ]]; then
    git clone https://github.com/RyanLiu112/GenPRM.git "${GENPRM_ROOT}"
fi

git -C "${GENPRM_ROOT}" fetch origin "${GENPRM_UPSTREAM_COMMIT}"
git -C "${GENPRM_ROOT}" checkout --detach "${GENPRM_UPSTREAM_COMMIT}"

if [[ ! -x "${GENPRM_VENV}/bin/python" ]]; then
    if command -v python3.10 >/dev/null 2>&1; then
        python3.10 -m venv "${GENPRM_VENV}"
    elif command -v uv >/dev/null 2>&1; then
        uv venv --python 3.10 "${GENPRM_VENV}"
    else
        echo "Python 3.10 or uv is required to create ${GENPRM_VENV}." >&2
        echo "Install uv once, then rerun this script: python -m pip install uv" >&2
        exit 1
    fi
fi

"${GENPRM_VENV}/bin/python" -m pip install --upgrade pip
"${GENPRM_VENV}/bin/python" -m pip install \
    -r "${GENPRM_ROOT}/src/requirements.txt" \
    hf_transfer

GENPRM_ROOT="${GENPRM_ROOT}" \
GENPRM_UPSTREAM_COMMIT="${GENPRM_UPSTREAM_COMMIT}" \
"${GENPRM_VENV}/bin/python" - <<'PY'
import os
import platform
import subprocess

import torch
import transformers
import vllm

root = os.environ["GENPRM_ROOT"]
expected_commit = os.environ["GENPRM_UPSTREAM_COMMIT"]
actual_commit = subprocess.check_output(
    ["git", "-C", root, "rev-parse", "HEAD"],
    text=True,
).strip()

if not platform.python_version().startswith("3.10."):
    raise RuntimeError(
        f"Expected Python 3.10.x, got {platform.python_version()}"
    )
if actual_commit != expected_commit:
    raise RuntimeError(
        f"Expected upstream commit {expected_commit}, got {actual_commit}"
    )
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available in the GenPRM environment.")

print("Python:", platform.python_version())
print("Torch:", torch.__version__)
print("Torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("vLLM:", vllm.__version__)
print("Transformers:", transformers.__version__)
print("GenPRM upstream commit:", actual_commit)
print("HF_HOME:", os.environ["HF_HOME"])
PY

echo
echo "GENPRM OFFICIAL ENVIRONMENT READY"
echo "Activate with: source ${GENPRM_VENV}/bin/activate"
