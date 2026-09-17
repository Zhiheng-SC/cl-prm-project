# OVM Feasibility Study

> Does OVM-guided step-level search improve GSM8K accuracy and final candidates quality over unguided sampling under a
> fixed K=20 sampling budget?
---

## Status

**Implemented -- final results pending [WIP]**

---

## Description

* Uses Mistral-7B OVM available on Hugging Face
* Compares greedy decoding, self-consistency, ORM reranking and OVM-guided beam search
* Randomness is handled by runs across three seeds
* The main implementation (generation, value-guided beam search) are unchanged, only adapted for the study

---

## Schema Versioning

- schema versioning for configs and manifests is desirable
- the team needs to discuss and accept formal versioning
- now is custom-defined

---

## OVM

| File                          | Purpose                                                                          |
|-------------------------------|----------------------------------------------------------------------------------|
| `download_models.py`          | Downloads the released Mistral generator and verifier weights and files          |
| `prepare_pilot.py`            | Prepares and selects the pilot study (100 questions) together with its' manifest |
| `run_pilot.py`                | Runs methods with runtime and memory metrics **[WIP]**                           |
| `utils.py`                    | Helpers for frequent operations                                                  |
| `analyze_results.py`          | Calculates metrics and provides formal evaluation report **[WIP]**               |
| `colab_ovm_feasibility.ipynb` | The Colab notebook used for execution (1 GPU - A100 80 GB)                       |
| `requirements-colab.txt`      | Requirements file, ideally to be satisfied from a fresh Colab environment        |

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

# smoke test or pilot study [WIP  ] 
python3 run_pilot.py --config configs/ovm_feasibility.json \
                     --model-snapshot ovm-models \
                     --pilot_manifest data/feasibility/pilot_manifest.json \
                     --output_dir outputs/ovm-smoke \ 
                     --seeds 42 # or 41 42 43 \
                     --limit 2 # or None

# [WIP]
#python3 analyze_results.py \
#  --config configs/ovm_feasibility.json \
#  --pilot-manifest data/feasibility/pilot_manifest.json  \
#  --input-dir outputs/ovm/feasibility \
#  --output-json outputs/ovm/feasibility/summary.json \
#  --output-markdown outputs/ovm/feasibility/summary.md
```

---

## DEV Policy

**DO NOT** commit:

- model checkpoints
- pilot study data
- generated metadata

---