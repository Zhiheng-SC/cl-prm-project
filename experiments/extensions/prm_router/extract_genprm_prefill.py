"""Extract input-only GenPRM prefill features on the frozen development splits.

This is a separate Transformers forward pass, not a hook into official vLLM.
It never decodes a token or changes official GenPRM predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments.formal.prm_router.run_formal_inference import validate_split  # noqa: E402
from experiments.extensions.prm_router.run_alternative import (  # noqa: E402
    messages_for_genprm,
    read_records,
)

CONFIG = REPO_ROOT / "configs/experiments/prm_router_extensions.json"
MANIFEST = REPO_ROOT / "configs/experiments/prm_router_formal_split_manifest.json"
OUTPUT = REPO_ROOT / "outputs/prm_router/extensions/prefill"


def stage_one_prompt(messages: list[dict[str, str]], tokenizer: object, current_step: int) -> str:
    """Mirror the pinned upstream build_prompt and default analyze_template."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    eos = tokenizer.eos_token
    if eos and prompt.endswith(eos + "\n"):
        prompt = prompt[: -len(eos + "\n")]
    elif eos and prompt.endswith(eos):
        prompt = prompt[: -len(eos)]
    return prompt + f"<analyze>\nLet's analyze the Paragraph {current_step} step by step: "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--limit", type=int, help="Use a separate pilot output directory.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--genprm-src", type=Path, default=REPO_ROOT.parent / "GenPRM/src")
    parser.add_argument("--warmup-examples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.warmup_examples < 0:
        raise ValueError("--warmup-examples cannot be negative.")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))["models"]["genprm_official"]
    source = args.genprm_src.parent if args.genprm_src.name == "src" else args.genprm_src
    actual_commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != config["upstream_commit"]:
        raise ValueError(f"Expected GenPRM source {config['upstream_commit']}, got {actual_commit}.")
    input_path = validate_split(args.split, json.loads(MANIFEST.read_text(encoding="utf-8")))
    rows = read_records(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("Split contains no rows.")

    import numpy as np
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the GenPRM prefill probe.")
    model_path = snapshot_download(repo_id=config["name"], revision=config["revision"])
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()

    def prefill(row: dict) -> tuple[np.ndarray, int, float, str]:
        prompt = stage_one_prompt(messages_for_genprm(row), tokenizer, int(row["current_step"]))
        inputs = tokenizer(prompt, return_tensors="pt")
        token_count = int(inputs["input_ids"].shape[1])
        if token_count > int(config["max_input_tokens"]):
            raise ValueError(f"Input {row['example_id']} has {token_count} tokens, exceeds limit.")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            result = model(**inputs, use_cache=False, output_hidden_states=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        vector = result.hidden_states[-1][0, -1, :].float().cpu().numpy().copy()
        return vector, token_count, elapsed, hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    # Warm up the forward path, excluding compilation and first-call effects from timings.
    for row in rows[: min(args.warmup_examples, len(rows))]:
        prefill(row)

    features = []
    records = []
    for index, row in enumerate(rows):
        vector, token_count, elapsed, prompt_hash = prefill(row)
        features.append(vector)
        records.append({
            "example_id": row["example_id"],
            "current_step": row["current_step"],
            "feature_row": index,
            "prefill_prompt_tokens": token_count,
            "prefill_forward_seconds": elapsed,
            "prefill_prompt_sha256": prompt_hash,
        })
        print(f"{index + 1}/{len(rows)} prefill={elapsed:.3f}s tokens={token_count}", flush=True)

    directory = args.output_dir / ("pilot" if args.limit else "full") / args.split
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "genprm_prefill_features.npz", hidden=np.stack(features))
    (directory / "genprm_prefill_index.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (directory / "genprm_prefill.metadata.json").write_text(
        json.dumps({
            "split": args.split,
            "examples": len(rows),
            "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "model": config["name"],
            "model_revision": config["revision"],
            "upstream_commit": actual_commit,
            "feature": "final_layer_hidden_state_at_last_input_token",
            "timing": "synchronized_transformers_forward_only; not official_vllm_prefill_latency",
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Saved prefill features to {directory}")


if __name__ == "__main__":
    main()
