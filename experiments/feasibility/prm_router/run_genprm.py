"""Run GenPRM-1.5B on the PRMBench smoke-test subset.

This Windows-friendly feasibility version uses Hugging Face Transformers.
It performs GenPRM analysis followed by a Yes/No judgement, without
executing model-generated Python code.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

from cl_prm.utils.inference_io import (
    finalize_run_metadata,
    initialize_run_metadata,
    prepare_resume,
)


MODEL_NAME = "GenPRM/GenPRM-1.5B"
MODEL_REVISION = "a0fa69768f4524257e1730fec639aa7781c7fa82"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "data" / "prm_router" / "smoke_test.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "prm_router" / "genprm_smoke_test.jsonl"

SYSTEM_PROMPT = (
    "You are a math teacher. Your task is to review and critique "
    "the paragraphs in the solution step by step."
)


class StopOnTokenSequence(StoppingCriteria):
    """Stop generation when a specified token sequence appears."""

    def __init__(self, stop_ids: list[int]) -> None:
        super().__init__()
        self.stop_ids = stop_ids

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        if input_ids.shape[1] < len(self.stop_ids):
            return False

        suffix = input_ids[0, -len(self.stop_ids) :]
        target = torch.tensor(
            self.stop_ids,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        return bool(torch.equal(suffix, target))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--revision",
        default=MODEL_REVISION,
        help="Pinned Hugging Face model revision.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append after validating an existing output prefix.",
    )
    parser.add_argument(
        "--warmup-examples",
        type=int,
        default=3,
        help="Untimed model warmup examples before measured inference.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-analysis-tokens", type=int, default=256)
    parser.add_argument("--max-input-tokens", type=int, default=3072)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def build_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    paragraphs = "\n".join(
        f"Paragraph {index}: {step}"
        for index, step in enumerate(record["steps"], start=1)
    )

    user_content = f"Question: {record['question']}\n\n{paragraphs}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def prepare_prompt(
    tokenizer: Any,
    record: dict[str, Any],
) -> tuple[str, int]:
    messages = build_messages(record)

    chat_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    current_step = len(record["steps"])
    analysis_start = (
        f"<analyze>\nLet's analyze Paragraph {current_step} step by step: "
    )

    return chat_prompt + analysis_start, current_step


def generate_analysis(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    max_input_tokens: int,
) -> tuple[str, int, int, bool]:
    encoded = tokenizer(prompt, return_tensors="pt")
    input_tokens = encoded["input_ids"].shape[1]

    if input_tokens > max_input_tokens:
        raise ValueError(
            f"Input contains {input_tokens} tokens, exceeding the configured "
            f"limit of {max_input_tokens}."
        )

    encoded = {
        name: tensor.to(model.device)
        for name, tensor in encoded.items()
    }

    stop_ids = tokenizer.encode(
        "</analyze>\n",
        add_special_tokens=False,
    )

    stopping_criteria = StoppingCriteriaList(
        [StopOnTokenSequence(stop_ids)]
    )

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.0,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = generated[0, input_tokens:]
    analysis = tokenizer.decode(new_ids, skip_special_tokens=True)

    analysis_complete = "</analyze>" in analysis

    if analysis_complete:
        analysis = analysis.split("</analyze>", maxsplit=1)[0]
        analysis = analysis.rstrip() + "\n</analyze>\n"
    else:
        analysis = analysis.rstrip() + "\n</analyze>\n"

    return (
        analysis,
        input_tokens,
        int(new_ids.shape[0]),
        analysis_complete,
    )


def generate_judgement(
    model: Any,
    tokenizer: Any,
    full_prompt: str,
) -> tuple[str, float, int]:
    output_start = "<output>\n**Judgement**: $\\boxed"
    judgement_prompt = full_prompt + output_start

    encoded = tokenizer(judgement_prompt, return_tensors="pt")
    input_tokens = encoded["input_ids"].shape[1]

    encoded = {
        name: tensor.to(model.device)
        for name, tensor in encoded.items()
    }

    with torch.inference_mode():
        generated = model.generate(
            **encoded,
            max_new_tokens=12,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = generated.sequences[0, input_tokens:]
    judgement = tokenizer.decode(new_ids, skip_special_tokens=True)

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[-1]
    no_id = tokenizer.encode("No", add_special_tokens=False)[-1]

    yes_probability = 0.5

    for position, token_id in enumerate(new_ids.tolist()):
        if token_id in {yes_id, no_id} and position < len(generated.scores):
            logits = generated.scores[position][0]
            pair_logits = torch.stack([logits[yes_id], logits[no_id]])
            pair_probabilities = torch.softmax(pair_logits.float(), dim=0)
            yes_probability = float(pair_probabilities[0].item())
            break

    return judgement, yes_probability, int(new_ids.shape[0])


def run_one(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    args: argparse.Namespace,
    index: int,
) -> dict[str, Any]:
    torch.manual_seed(args.seed + index)
    torch.cuda.manual_seed_all(args.seed + index)

    prompt, current_step = prepare_prompt(tokenizer, record)

    torch.cuda.synchronize()
    start_time = time.perf_counter()

    (
        analysis,
        input_tokens,
        analysis_tokens,
        analysis_complete,
    ) = generate_analysis(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_analysis_tokens,
        max_input_tokens=args.max_input_tokens,
    )

    judgement, yes_probability, judgement_tokens = generate_judgement(
        model=model,
        tokenizer=tokenizer,
        full_prompt=prompt + analysis,
    )

    torch.cuda.synchronize()
    runtime = time.perf_counter() - start_time

    match = re.search(r"\b(Yes|No)\b", judgement, flags=re.IGNORECASE)
    parsed_judgement = match.group(1).capitalize() if match else "Unknown"

    if parsed_judgement == "Unknown":
        raise ValueError(
            "Could not parse a Yes/No judgement for "
            f"example {record['example_id']!r}. "
            f"Generated text: {judgement!r}"
        )

    prediction = int(yes_probability >= 0.5)
    probability_judgement = "Yes" if prediction == 1 else "No"
    judgement_matches_prediction = (
        parsed_judgement == probability_judgement
    )
    label = int(record["label"])

    result = dict(record)
    result.update(
        {
            "genprm_model": args.model,
            "genprm_model_revision": args.revision,
            "genprm_seed_base": args.seed,
            "genprm_max_analysis_tokens": args.max_analysis_tokens,
            "genprm_max_input_tokens": args.max_input_tokens,
            "genprm_current_step": current_step,
            "genprm_analysis": analysis,
            "genprm_analysis_complete": analysis_complete,
            "genprm_judgement_text": judgement,
            "genprm_parsed_judgement": parsed_judgement,
            "genprm_probability_judgement": probability_judgement,
            "genprm_judgement_matches_prediction": (
                judgement_matches_prediction
            ),
            "genprm_score": yes_probability,
            "genprm_prediction": prediction,
            "genprm_correct": prediction == label,
            "genprm_runtime_seconds": runtime,
            "genprm_input_tokens": input_tokens,
            "genprm_generated_tokens": analysis_tokens + judgement_tokens,
            "genprm_mode": "analysis_without_code_execution",
        }
    )

    return result


def main() -> None:
    args = parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if args.max_analysis_tokens < 1:
        raise ValueError("--max-analysis-tokens must be positive.")
    if args.max_input_tokens < 1:
        raise ValueError("--max-input-tokens must be positive.")
    if args.warmup_examples < 0:
        raise ValueError("--warmup-examples cannot be negative.")

    records = read_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError(f"No input records were found in {args.input}.")

    resume_state = prepare_resume(
        output_path=args.output,
        input_records=records,
        resume=args.resume,
        expected_metadata={
            "genprm_model": args.model,
            "genprm_model_revision": args.revision,
            "genprm_seed_base": args.seed,
            "genprm_max_analysis_tokens": args.max_analysis_tokens,
            "genprm_max_input_tokens": args.max_input_tokens,
            "genprm_mode": "analysis_without_code_execution",
        },
    )
    pending_records = records[resume_state.completed :]

    if args.resume:
        print(
            f"Resume validation passed: {resume_state.completed}/"
            f"{len(records)} records already complete."
        )

    results = list(resume_state.existing_records)

    if pending_records and not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for GenPRM inference.")

    metadata_path = initialize_run_metadata(
        script_path=Path(__file__),
        repo_root=REPO_ROOT,
        input_path=args.input,
        output_path=args.output,
        selected_examples=len(records),
        model_name=args.model,
        model_revision=args.revision,
        inference_parameters={
            "seed": args.seed,
            "max_analysis_tokens": args.max_analysis_tokens,
            "max_input_tokens": args.max_input_tokens,
            "dtype": "bfloat16",
            "mode": "analysis_without_code_execution",
            "warmup_examples": args.warmup_examples,
            "timing_scope": (
                "synchronized_analysis_and_judgement_generation"
            ),
        },
        resume=args.resume,
    )

    if not pending_records:
        correct = sum(
            int(result["genprm_correct"]) for result in results
        )
        total_runtime = sum(
            float(result["genprm_runtime_seconds"])
            for result in results
        )
        accuracy = correct / len(records)
        average_runtime = total_runtime / len(records)
        finalize_run_metadata(
            metadata_path=metadata_path,
            output_path=args.output,
            summary={
                "examples": len(records),
                "correct": correct,
                "accuracy": accuracy,
                "average_runtime_seconds": average_runtime,
            },
        )
        print("GenPRM output is already complete.")
        print(f"Accuracy:        {accuracy:.4f}")
        print(f"Correct:         {correct}/{len(records)}")
        print(f"Average runtime: {average_runtime:.4f}s")
        print(f"Saved to:        {args.output.resolve()}")
        print(f"Metadata:        {metadata_path.resolve()}")
        return

    print(f"Loading model: {args.model}@{args.revision}")
    print(f"Input examples: {len(records)}")
    print(f"Remaining examples: {len(pending_records)}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.eval()

    warmup_count = min(args.warmup_examples, len(pending_records))
    if warmup_count:
        print(f"Warming up on {warmup_count} example(s).")
        for offset, record in enumerate(
            pending_records[:warmup_count],
            start=resume_state.completed,
        ):
            run_one(
                model=model,
                tokenizer=tokenizer,
                record=record,
                args=args,
                index=offset,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open(
        resume_state.file_mode,
        encoding="utf-8",
    ) as output_file:
        for index, record in enumerate(
            pending_records,
            start=resume_state.completed,
        ):
            print(
                f"[{index + 1}/{len(records)}] "
                f"{record['example_id']} "
                f"(label={record['label']})"
            )

            result = run_one(
                model=model,
                tokenizer=tokenizer,
                record=record,
                args=args,
                index=index,
            )

            results.append(result)
            output_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            output_file.flush()

            print(
                f"  score={result['genprm_score']:.4f}, "
                f"prediction={result['genprm_prediction']}, "
                f"correct={result['genprm_correct']}, "
                f"runtime={result['genprm_runtime_seconds']:.2f}s"
            )

    correct = sum(int(result["genprm_correct"]) for result in results)
    accuracy = correct / len(results)
    average_runtime = sum(
        float(result["genprm_runtime_seconds"])
        for result in results
    ) / len(results)
    finalize_run_metadata(
        metadata_path=metadata_path,
        output_path=args.output,
        summary={
            "examples": len(results),
            "correct": correct,
            "accuracy": accuracy,
            "average_runtime_seconds": average_runtime,
        },
    )

    print()
    print("GenPRM inference completed.")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Correct:         {correct}/{len(results)}")
    print(f"Average runtime: {average_runtime:.4f}s")
    print(f"Saved to:        {args.output.resolve()}")
    print(f"Metadata:        {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
