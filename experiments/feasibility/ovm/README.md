# OVM Feasibility Study

> Does OVM-guided beam search improve GSM8K accuracy and final-candidate quality over unguided sampling at K=20, and at
> what runtime cost?
---

## Status

**Completed**

---

## Results

Generated Markdown: [ovm-eval.md](output/ovm-eval.md)

Generated JSON: [ovm-eval.json](output/ovm-eval.json)

---

## Decision

* The team has decided to finalize experiments with `PRM-Router`
* The completed feasibility study shows future potential for research into efficient value-guided beam search approaches

---

## Description

* Uses Mistral-7B OVM available on Hugging Face
* Compares greedy decoding, self-consistency, ORM reranking (post-selection) and OVM-guided beam search
* Randomness is handled by runs across three seeds
* The main implementation (generation, value-guided beam search) are unchanged, only adapted for the study from the
  original OVM repo

---

## Schema Versioning

- schema versioning for configs, manifests and other data files is desirable but not strictly required
- the team needs to discuss and accept formal versioning before modifying existing changes

---

## OVM

| File                     | Purpose                                                                          |
|--------------------------|----------------------------------------------------------------------------------|
| `download_models.py`     | Downloads the released Mistral generator and verifier weights and files          |
| `prepare_pilot.py`       | Prepares and selects the pilot study (100 questions) together with its' manifest |
| `run_pilot.py`           | Runs methods with runtime and memory metrics                                     |
| `helpers.py`             | Helpers for frequent operations                                                  |
| `metrics.py`             | Defines constant variable names                                                  |
| `analyze_results.py`     | Calculates metrics and provides formal evaluation report                         |
| `ovm_feasibility.ipynb`  | The Colab notebook used for execution (1 GPU - A100 80 GB)                       |
| `requirements-colab.txt` | Requirements file, ideally to be satisfied from a fresh Colab environment        |

## Workflow

- **Requirements**
    - A100 80GB GPU on Google Colab
    - Google Drive Storage with at least 50GBs free

- Follow the notebook for execution

- CLI

```bash
python3 download_models.py --output ovm-models

python3 prepare_pilot.py --config configs/ovm_feasibility.json \
                         --output-dir data/feasibility

# smoke test and/or pilot study
python3 run_pilot.py --config configs/ovm_feasibility.json \
                     --model-snapshot ovm-models \
                     --pilot_manifest data/feasibility/pilot_manifest.json \
                     --output_dir outputs/ovm-smoke # or ovm \ 
                     --seeds 42 # or 41 42 43 \
                     --limit 2 # or None

# eval and formally assess feasibility study status
python3 analyze_results.py --config configs/ovm_feasibility.json \ 
                           --pilot-manifest data/feasibility/pilot_manifest.json \
                           --input-dir output/ovm \
                           --output-json output/ovm-eval.json \
                           --output-markdown output/ovm-eval.md
```

---

## DEV Policy

**DO NOT** commit:

- model checkpoints
- too large files

---