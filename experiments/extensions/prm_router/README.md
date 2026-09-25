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

If Mario has not started the official 600/200 run, ask him to pull `main`,
run the five-example pilot above, then run these two commands in that same
official environment. Keep the JSONL and metadata files for both splits;
do not use the formal runner's older `--verifiers genprm` condition.

```bash
python experiments/extensions/prm_router/run_extensions.py \
  --split train --models genprm_official --allow-generated-code --execute
python experiments/extensions/prm_router/run_extensions.py \
  --split validation --models genprm_official --allow-generated-code --execute
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

## Optional GenPRM prefill probe

**Prefill is already part of every GenPRM generation call.** The model
processes the input prompt and builds its attention cache before decoding
output tokens. The official upstream vLLM runner even enables *chunked
prefill*. Its current API returns the generated text and reward, however,
not the intermediate hidden vectors from prefill. Stage 1 and later
verification calls may each process a different prompt; `max_tokens=2048`
is a generation limit, not a prefill length.

The official 600/200 JSONL files supply the **target**
(`extension_runtime_seconds`). A separate, optional probe in
`extract_genprm_prefill.py` reconstructs the *first analysis request* using
the pinned GenPRM prompt format and checkpoint, then runs a Transformers
forward pass with **zero generated tokens**. It stores the final-layer hidden
vector of the last input token, its prompt length, and a synchronized forward
time. It does not change official scores, execute generated code, or require
rerunning official inference. Its forward time is **not** the official vLLM
prefill latency, so do not subtract it from `extension_runtime_seconds`.

Use an A100 after the official GenPRM process exits. The official environment
already contains Transformers and the pinned model; only one copy of the 7B
checkpoint should be in GPU memory at a time. Run a small pilot first and
confirm that its vectors are finite and the recorded prompt sizes fit the
limit. `--limit 5` is available for a quick smoke check but simply takes the
first five rows. For one pilot that also estimates full-run duration,
`--pilot-size 20` selects examples spread across the split's actual tokenized
prompt lengths. Run it on **train** before deciding whether to process both
full splits:

```bash
source /opt/genprm-env/bin/activate
python experiments/extensions/prm_router/extract_genprm_prefill.py \
  --split train --pilot-size 20
# After checking the pilot's per-example times and memory use:
python experiments/extensions/prm_router/extract_genprm_prefill.py --split train
python experiments/extensions/prm_router/extract_genprm_prefill.py --split validation
```

The pilot measures a Transformers forward pass, excluding checkpoint loading,
tokenization, and output writing. Scale its total cautiously by the 600/200
sample counts; prompt length alone does not capture all timing variation.

The probe writes `genprm_prefill_features.npz` (rows of hidden vectors),
`genprm_prefill_index.jsonl` (row-to-`example_id` mapping), and a metadata
file under `outputs/prm_router/extensions/prefill/{pilot|full}/{split}/`.
Join features to official results by `example_id`, and check the full split
SHA and pinned revision before fitting. Fit any dimensionality reduction
and cost predictor **only on train groups**, then evaluate once on validation.
For a deployed prefill router, the probe's extra full-model forward pass must
also be included in total routing cost for every candidate; compare its
net time against a no-probe baseline. The current script tests whether the
representation carries predictive information; it is not an online router.
It cannot reuse its Transformers attention cache inside the separate official
vLLM process, so selected candidates still pay for GenPRM's normal prefill.

## Train and validation analysis

Reuse Mario's existing RE JSONL outputs and the validation-selected RE
threshold. Each extension gets *its own* gain router and Ridge runtime model;
the frozen RE features, 20% call budget, and lambda/mu grid are reused.

```bash
python experiments/extensions/prm_router/evaluate_development.py \
  --model math_prm \
  --train-reasoneval outputs/prm_router/formal/train/reasoneval.jsonl \
  --validation-reasoneval outputs/prm_router/formal/validation/reasoneval.jsonl \
  --reason-eval-threshold 0.9469655402936041
```

Replace `math_prm` with `skywork_prm` or `genprm_official` for other runs.
The threshold shown above is the value printed by Mario's executed notebook;
verify it against `formal/validation/reasoneval_threshold.json` if that artifact
is regenerated. Review MAE against
the median-runtime baseline and R²; if the cost model is weak, report the
gain-only (`mu=0`) and selected routing accuracies and measured times without
claiming precise per-example cost prediction.

The main study's test labels/results have already been viewed. Treat any
extension test as exploratory and disclose that status. `run_extensions.py`
requires `--confirm-exploratory-test` for test execution.

## Exploratory 256-token GenPRM cost analysis

The four `*_256.py` and `genprm_cost_*.py` scripts in this directory analyze
previously collected **GenPRM-7B Transformers analysis-only** results, with
`max_analysis_tokens=256`. They are separate from the official vLLM GenPRM
extension above. They do not load GenPRM, collect prefill activations, or run
model inference; CPU is sufficient for the offline analyses. Run commands from
the project root. The example `../cl-prm-artifacts` path is a sibling directory
containing `formal/train/{genprm,reasoneval}.jsonl` and matching validation
files; use `--artifacts-zip` instead when working from a downloaded artifact ZIP.

- `genprm_cost_sweep.py`: compare fixed baseline and candidate cost models
  using question-group CV on 600 train and diagnostics on 200 validation.
- `genprm_cost_tune.py`: select a runtime regressor using train-only
  question-group CV; writes `validation_predictions.jsonl`.
- `genprm_cost_two_stage_256.py`: cross-fit a generated-token predictor,
  then use **predicted**, rather than observed, tokens to estimate runtime.
- `evaluate_genprm_cost_routing_256.py`: retrospectively apply predicted
  runtime to the same validation examples at the fixed call budget. Its
  `--predictions` input can be the direct or two-stage output.

For example, with CatBoost installed (`python -m pip install catboost`):

```bash
python experiments/extensions/prm_router/genprm_cost_sweep.py --artifacts-dir ../cl-prm-artifacts --preset full --target runtime --re-threshold 0.9469655402936041 --output-dir outputs/prm_router/extensions/cost_sweep_256
python experiments/extensions/prm_router/genprm_cost_tune.py --artifacts-dir ../cl-prm-artifacts --families rf histgb ridge_text catboost --output-dir outputs/prm_router/extensions/cost_tune_256/full
python experiments/extensions/prm_router/genprm_cost_two_stage_256.py --artifacts-dir ../cl-prm-artifacts --direct-predictions outputs/prm_router/extensions/cost_tune_256/full/validation_predictions.jsonl --output-dir outputs/prm_router/extensions/cost_two_stage_256/full
python experiments/extensions/prm_router/evaluate_genprm_cost_routing_256.py --artifacts-dir ../cl-prm-artifacts --predictions outputs/prm_router/extensions/cost_two_stage_256/full/validation_predictions.jsonl --model-columns two_stage selected mean ridge_re --re-threshold 0.9469655402936041 --output-dir outputs/prm_router/extensions/cost_routing_256/two_stage
```

Verify the RE threshold against the archived development selection before
running the sweep or routing comparison. The tuning and two-stage scripts
currently construct their RE-distance feature around `0.5`; the routing
command above uses the selected RE decision threshold (approximately
`0.947`). Do not describe those features as distance to the same threshold.

All model selection stays within train question groups. Validation has already
been examined repeatedly, so comparisons and `mu` sweeps here are
**exploratory**; do not select a new method from these results and report it as
an untouched validation result. These scripts do not read the 400-example test.
The 256-token analysis cap and the separate judgement generation also mean
their token and runtime targets do not directly represent official GenPRM with
a 2048-token setting. Prediction errors alone do not account for the cost of
prefilling every candidate in a future activation-based router.

Official model usage references:

- Qwen: https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B
- Skywork: https://huggingface.co/Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B
- Skywork inference: https://github.com/SkyworkAI/skywork-o1-prm-inference
- GenPRM: https://github.com/RyanLiu112/GenPRM
