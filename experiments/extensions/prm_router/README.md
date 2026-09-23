# Alternative second-stage PRMs (exploratory)

The frozen main experiment remains ReasonEval (RE) -> PathFinder. This directory
adds three *separate* RE -> alternative-verifier comparisons on the same
PRMBench 600/200/400 question-group split: official GenPRM-7B,
Qwen2.5-Math-PRM-7B, and Skywork-o1-Open-PRM-Qwen-2.5-7B. Do not use the
formal runner's `--verifiers genprm` for the official condition: it invokes the
older 256-token analysis-only implementation.

## Mario's A100 notebook

`experiments/formal/prm_router/run_gpu_environment.ipynb` contains executed
cells and uploads for the main RE -> PathFinder development and test run. Keep
those cells and outputs intact. It clones the project once, so an already open
session needs `git pull` before using these new files. Run the following
commands in *new cells after* the main workflow, or from a separate terminal.
In particular, do not rerun the notebook's formal split preparation or test
cells just to add these models. Its HF uploads target `formal/`; use separate
`extensions/` paths for any extension artifacts.

Only one 7B checkpoint should be loaded per process. Use the same A100 for
timing comparisons if possible; compare separate-environment runtimes with
their hardware and dependency versions recorded in the metadata sidecars.

## Inference

The extension config pins all three checkpoint revisions. The runner checks
the frozen split manifest's hash and sample count before executing. Start with
a small pilot in its own output namespace:

```bash
python experiments/extensions/prm_router/run_extensions.py \
  --split validation --models math_prm --limit 5
# Add --execute once the plan is correct.
```

For the Qwen Math PRM, the existing RunPod/Colab PyTorch environment with
`requirements-runpod.txt` is suitable in principle; verify it on the A100.
The script uses the checkpoint's own `<extra_0>` marker and reads the positive
class probability at the *last prefix step*.

Skywork uses the authors' inference wrapper rather than the Qwen classification
head. Prepare its official source beside the project checkout:

```bash
git clone https://github.com/SkyworkAI/skywork-o1-prm-inference.git ../skywork-o1-prm-inference
git -C ../skywork-o1-prm-inference checkout --detach 719b56b17447405e0f10e6c0360a581cf4ffa9c1
python experiments/extensions/prm_router/run_extensions.py \
  --split validation --models skywork_prm --limit 5 --execute
```

Skywork's official `prepare_input` splits on newline; this adapter replaces
line breaks *within* a PRMBench step with spaces, then separates the original
steps with newlines. It checks the marker count and reads the final step's
sigmoid reward. Inspect a few original versus normalized prefixes in the
pilot before treating its numbers as a comparable experiment.

Official GenPRM requires the separate environment from
`scripts/bootstrap_genprm_official.sh` (Python 3.10 and vLLM). The upstream
implementation executes its generated Python; use only an isolated disposable
GPU environment without credentials or sensitive mounts. If running in Mario's
Colab, the existing notebook's Python 3.12 environment is not the verified
official GenPRM environment. Run this condition in its dedicated environment
and record that hardware difference, or create an equivalent isolated A100
environment first.

```bash
source /opt/genprm-env/bin/activate
python experiments/extensions/prm_router/run_extensions.py \
  --split validation --models genprm_official --limit 5 \
  --allow-generated-code --execute
```

After the pilot, omit `--limit` and run **train and validation separately**.
Use `--models math_prm skywork_prm` in one plan if both environments are ready;
the script launches a fresh process for each. For example:

```bash
python experiments/extensions/prm_router/run_extensions.py \
  --split train --models math_prm skywork_prm --execute
python experiments/extensions/prm_router/run_extensions.py \
  --split validation --models math_prm skywork_prm --execute
```

Outputs live under `outputs/prm_router/extensions/{pilot|full}/{split}/`,
never under `outputs/prm_router/formal/`. Each has a JSON metadata sidecar;
`--resume` validates the exact input prefix, checkpoint revision, script hash,
parameters, and environment. The score threshold of 0.5 is a starting point,
not a calibration claim. No model generates tokens except GenPRM; its official
API does not return total generation usage, so this runner records null for
that field and measures wall-clock time instead. Three initial calls warm up
each loaded model before timed records. It does not silently truncate
overlong inputs.

## Train and validation analysis

Reuse Mario's existing RE JSONL outputs and the validation-selected RE
threshold. Each extension gets *its own* gain router and Ridge runtime model;
the frozen RE features, 20% call budget, and lambda/mu grid are reused.

```bash
python experiments/extensions/prm_router/evaluate_development.py \
  --model math_prm \
  --train-reasoneval outputs/prm_router/formal/train/reasoneval.jsonl \
  --validation-reasoneval outputs/prm_router/formal/validation/reasoneval.jsonl \
  --reason-eval-threshold 0.5
```

Replace `math_prm` with `skywork_prm` or `genprm_official` for other runs and
pass the actual frozen RE threshold if different from 0.5. Review MAE against
the median-runtime baseline and R²; if the cost model is weak, report the
gain-only (`mu=0`) and selected routing accuracies and measured times without
claiming precise per-example cost prediction.

The main study's test labels/results have already been viewed. Treat any
extension test as exploratory and disclose that status. `run_extensions.py`
requires `--confirm-exploratory-test` for test execution.

Official model usage references:

- Qwen: https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B
- Skywork: https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B
- Skywork inference: https://github.com/SkyworkAI/skywork-o1-prm-inference
- GenPRM: https://github.com/RyanLiu112/GenPRM
