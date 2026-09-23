"""Score frozen PRMBench prefixes with one alternative second-stage verifier.

This entry point is independent of the frozen RE->PathFinder formal runner.
GPU dependencies and third-party inference implementations are imported lazily.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cl_prm.utils.inference_io import (  # noqa: E402
    ensure_unique_keys,
    finalize_run_metadata,
    initialize_run_metadata,
    prepare_resume,
)

DEFAULT_CONFIG = REPO_ROOT / "configs/experiments/prm_router_extensions.json"


def read_records(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ensure_unique_keys(rows, str(path))
    for row in rows:
        if not row.get("steps") or len(row["steps"]) != int(row["current_step"]):
            raise ValueError(f"Expected an exact step prefix for {row.get('example_id')}.")
    return rows


def messages_for_genprm(row: dict[str, Any]) -> list[dict[str, str]]:
    paragraphs = "\n".join(
        f"Paragraph {i}: {step}" for i, step in enumerate(row["steps"], 1)
    )
    return [
        {"role": "system", "content": "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step."},
        {"role": "user", "content": f"Question: {row['question']}\n\n{paragraphs}"},
    ]


def math_input(tokenizer: Any, row: dict[str, Any]) -> tuple[Any, int]:
    marker = tokenizer.encode("<extra_0>", add_special_tokens=False)
    if len(marker) != 1:
        raise ValueError("Math PRM checkpoint must tokenize <extra_0> as one token.")
    messages = [
        {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
        {"role": "user", "content": str(row["question"])},
        {"role": "assistant", "content": "<extra_0>".join(map(str, row["steps"])) + "<extra_0>"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    ids = tokenizer.encode(prompt, return_tensors="pt")
    positions = (ids[0] == marker[0]).nonzero(as_tuple=True)[0]
    if len(positions) != len(row["steps"]):
        raise ValueError("Math PRM step markers do not match the input prefix.")
    return ids, int(positions[-1])


def skywork_input(tokenizer: Any, row: dict[str, Any], prepare_input: Any) -> tuple[Any, int]:
    # The official helper splits on newline. Flatten internal line breaks so
    # exactly one reward marker corresponds to each PRMBench step.
    steps = [" ".join(str(step).splitlines()) for step in row["steps"]]
    ids, parsed_steps, flags = prepare_input(
        str(row["question"]), "\n".join(steps), tokenizer=tokenizer, step_token="\n"
    )
    if len(parsed_steps) != len(steps) or sum(flags) != len(steps) or flags[-1] != 1:
        raise ValueError("Skywork reward markers do not match the input prefix.")
    return ids, len(flags) - 1


def check_source(path: Path, relative_file: str, commit: str) -> None:
    import subprocess

    source = path / relative_file
    if not source.is_file():
        raise FileNotFoundError(f"Missing official source: {source}")
    root = path.parent if path.name == "src" else path
    actual = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise ValueError(f"Official source at {root} has commit {actual}; expected {commit}.")


def load_verifier(name: str, config: dict[str, Any], args: argparse.Namespace) -> tuple[Any, Any, str]:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the 7B extension runners.")
    spec = config["models"][name]
    model_path = snapshot_download(repo_id=spec["name"], revision=spec["revision"])
    if name == "genprm_official":
        if not args.allow_generated_code:
            raise ValueError("Official GenPRM executes generated Python; pass --allow-generated-code in an isolated environment.")
        source = args.genprm_src
        check_source(source, "prm_evaluation/genprm_inference.py", spec["upstream_commit"])
        sys.path.insert(0, str(source))
        from prm_evaluation.genprm_inference import CodeExecutor, GenPRM

        model = GenPRM(model_path, tensor_parallel_size=args.tensor_parallel_size)
        return model, CodeExecutor, model_path

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if name == "math_prm":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        return model, tokenizer, model_path

    source = args.skywork_src
    check_source(source, "model_utils/prm_model.py", spec["upstream_commit"])
    sys.path.insert(0, str(source))
    from model_utils.io_utils import prepare_input
    from model_utils.prm_model import PRM_MODEL

    model = PRM_MODEL.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()
    return model, (tokenizer, prepare_input), model_path


def score_one(name: str, model: Any, helper: Any, row: dict[str, Any],
              max_input_tokens: int, max_tokens: int) -> tuple[float, str | None, int]:
    import torch

    if name == "genprm_official":
        messages = messages_for_genprm(row)
        prompt = model.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_tokens = len(model.tokenizer.encode(prompt))
        if input_tokens > max_input_tokens:
            raise ValueError(f"Input has {input_tokens} tokens, limit {max_input_tokens}.")
        paths, reward = model.inference(
            messages, cur_step=int(row["current_step"]),
            analyze=True, verify=True, execute=True, max_tokens=max_tokens,
            majority_num=1, code_executor=helper(), logging=False,
        )
        return float(reward), paths[0], input_tokens

    if name == "math_prm":
        ids, position = math_input(helper, row)
        input_tokens = ids.shape[1]
        if input_tokens > max_input_tokens:
            raise ValueError(f"Input has {input_tokens} tokens, limit {max_input_tokens}.")
        with torch.inference_mode():
            outputs = model(input_ids=ids.to(model.device))
        probability = torch.softmax(outputs[0][0, position].float(), dim=-1)[1]
        return float(probability.item()), None, input_tokens

    tokenizer, prepare_input = helper
    ids, last_flag = skywork_input(tokenizer, row, prepare_input)
    input_tokens = len(ids)
    if input_tokens > max_input_tokens:
        raise ValueError(f"Input has {input_tokens} tokens, limit {max_input_tokens}.")
    with torch.inference_mode():
        _, _, rewards = model(input_ids=torch.tensor([ids], device=model.pretrained_model.device), return_probs=True)
    return float(rewards[0, last_flag].item()), None, input_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verifier", choices=("genprm_official", "math_prm", "skywork_prm"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--warmup-examples", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-generated-code", action="store_true")
    parser.add_argument("--genprm-src", type=Path, default=REPO_ROOT.parent / "GenPRM/src")
    parser.add_argument("--skywork-src", type=Path, default=REPO_ROOT.parent / "skywork-o1-prm-inference")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    import torch

    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    spec = config["models"][args.verifier]
    rows = read_records(args.input)
    if not rows:
        raise ValueError("Input split has no records.")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive.")
        rows = rows[:args.limit]
    if args.warmup_examples < 0:
        raise ValueError("--warmup-examples cannot be negative.")
    if args.verifier == "genprm_official" and not args.allow_generated_code:
        raise ValueError("Official GenPRM executes generated Python; pass --allow-generated-code in an isolated environment.")
    max_input_tokens = int(spec["max_input_tokens"])
    max_tokens = int(spec.get("max_tokens", 0))
    threshold = float(spec["threshold"])
    expected = {
        "extension_verifier": args.verifier,
        "extension_model_revision": spec["revision"],
        "extension_threshold": threshold,
        "extension_max_input_tokens": max_input_tokens,
        "extension_max_tokens": max_tokens,
    }
    state = prepare_resume(output_path=args.output, input_records=rows,
                           resume=args.resume, expected_metadata=expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parameters = {**expected, "upstream_commit": spec.get("upstream_commit"),
                  "tensor_parallel_size": args.tensor_parallel_size if args.verifier == "genprm_official" else None,
                  "warmup_examples": args.warmup_examples,
                  "code_execution": args.verifier == "genprm_official"}
    metadata_path = initialize_run_metadata(
        script_path=Path(__file__), repo_root=REPO_ROOT, input_path=args.input,
        output_path=args.output, selected_examples=len(rows), model_name=spec["name"],
        model_revision=spec["revision"], inference_parameters=parameters,
        resume=args.resume,
    )
    if state.completed < len(rows):
        model, helper, _ = load_verifier(args.verifier, config, args)
        for row in rows[:min(args.warmup_examples, len(rows))]:
            score_one(args.verifier, model, helper, row, max_input_tokens, max_tokens)
        torch.cuda.synchronize()
        with args.output.open(state.file_mode, encoding="utf-8") as out:
            for index, row in enumerate(rows[state.completed:], state.completed + 1):
                torch.cuda.synchronize()
                start = time.perf_counter()
                score, path, input_tokens = score_one(
                    args.verifier, model, helper, row, max_input_tokens, max_tokens
                )
                torch.cuda.synchronize()
                runtime = time.perf_counter() - start
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f"Invalid score for {row['example_id']}: {score}")
                status = (
                    "missing_judgement"
                    if args.verifier == "genprm_official"
                    and not re.search(r"(Yes|No)\}", path or "", re.IGNORECASE)
                    else "ok"
                )
                prediction = int(score >= threshold) if status == "ok" else None
                result = {
                    **row, **expected, "extension_score": score,
                    "extension_status": status,
                    "extension_prediction": prediction,
                    "extension_correct": prediction == int(row["label"]) if prediction is not None else None,
                    "extension_runtime_seconds": runtime,
                    "extension_input_tokens": input_tokens,
                    "extension_generated_tokens": None,  # official API does not expose cumulative generation tokens
                    "extension_output_path": path,
                }
                out.write(json.dumps(result, ensure_ascii=True) + "\n")
                out.flush()
                print(f"{index}/{len(rows)} {args.verifier} runtime={runtime:.2f}s score={score:.4f}", flush=True)
    from cl_prm.utils.inference_io import read_output_jsonl

    results = read_output_jsonl(args.output)
    valid = [r for r in results if r["extension_status"] == "ok"]
    finalize_run_metadata(metadata_path=metadata_path, output_path=args.output,
                          summary={"examples": len(results), "valid_examples": len(valid),
                                   "accuracy_on_valid": sum(r["extension_correct"] for r in valid) / len(valid) if valid else None,
                                   "mean_runtime_seconds": sum(r["extension_runtime_seconds"] for r in results) / len(results)})


if __name__ == "__main__":
    main()
