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


def spread_by_prompt_length(rows: list[dict], tokenizer: object, count: int) -> list[dict]:
    """Choose deterministic prompt-token-length quantiles, keeping source order."""
    if count > len(rows):
        raise ValueError(f"--pilot-size {count} exceeds the {len(rows)} examples in the split.")
    lengths = [
        len(tokenizer.encode(stage_one_prompt(
            messages_for_genprm(row), tokenizer, int(row["current_step"])
        )))
        for row in rows
    ]
    ranked = sorted(range(len(rows)), key=lambda index: (lengths[index], index))
    ranks = (
        [len(rows) // 2] if count == 1 else
        [round(i * (len(rows) - 1) / (count - 1)) for i in range(count)]
    )
    selected = set(ranked[rank] for rank in ranks)
    return [row for index, row in enumerate(rows) if index in selected]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    pilot = parser.add_mutually_exclusive_group()
    pilot.add_argument("--limit", type=int, help="Smoke test on the first N examples.")
    pilot.add_argument("--pilot-size", type=int, help="Sample N examples spanning prompt lengths.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--genprm-src", type=Path, default=REPO_ROOT.parent / "GenPRM/src")
    parser.add_argument("--warmup-examples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.pilot_size is not None and args.pilot_size < 1:
        raise ValueError("--pilot-size must be positive.")
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
    if args.pilot_size is not None:
        rows = spread_by_prompt_length(rows, tokenizer, args.pilot_size)
        print(f"Selected {len(rows)} prompt-length-spread examples for the pilot.", flush=True)
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
        if not np.isfinite(vector).all():
            raise ValueError(f"Non-finite prefill vector for {row['example_id']}.")
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

    directory = args.output_dir / ("pilot" if args.limit or args.pilot_size else "full") / args.split
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
            "selection": (
                "first_examples" if args.limit else
                "prompt_length_spread" if args.pilot_size else "full_split"
            ),
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
