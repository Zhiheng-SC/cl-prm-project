"""Run PathFinder-PRM with fine-grained math and consistency signals.

The dry-run mode loads only the tokenizer and validates prompt construction,
special-token assumptions, and mask placement. Normal and 4-bit modes run the
two-stage PathFinder inference procedure and save JSONL predictions.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from inference_io import (
    finalize_run_metadata,
    initialize_run_metadata,
    prepare_resume,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "declare-lab/PathFinder-PRM-7B"
DEFAULT_REVISION = "84a7412511836cb4ed74377d9c703eb5638d814c"
DEFAULT_INPUT = (
    REPO_ROOT / "data" / "prm_router" / "feasibility_100.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs"
    / "prm_router"
    / "pathfinder"
    / "pathfinder_feasibility_100.jsonl"
)

PROMPT_PREFIX = (
    "You are a Math Teacher. Given a question and a student's solution, "
    "evaluate the mathemetical correctness, logic consistency of the "
    "current step and whether it will lead to the correct final solution"
)


def read_records(path: Path) -> list[dict[str, Any]]:
    """Read non-empty UTF-8 JSONL records."""
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}."
                ) from error

    if not records:
        raise ValueError(f"No input records were found in {path}.")

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--revision",
        type=str,
        default=DEFAULT_REVISION,
        help="Pinned Hugging Face model revision.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N records.",
    )
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for PathFinder's official gated score.",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=4096,
        help="Reject prompts longer than this limit instead of truncating.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load only the tokenizer and validate one prompt. "
            "Model weights are not downloaded."
        ),
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help=(
            "Load model weights with bitsandbytes NF4 quantization. "
            "Intended only for a local smoke test."
        ),
    )
    parser.add_argument(
        "--attention-implementation",
        choices=["sdpa", "flash_attention_2", "eager"],
        default="flash_attention_2",
        help=(
            "Attention backend. The official feasibility run used "
            "flash_attention_2; sdpa and eager are diagnostic alternatives."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")
    if args.max_input_tokens < 1:
        raise ValueError("--max-input-tokens must be positive.")
    if args.warmup_examples < 0:
        raise ValueError("--warmup-examples cannot be negative.")
    if args.dry_run and args.load_in_4bit:
        raise ValueError(
            "--dry-run does not load model weights, so do not combine it "
            "with --load-in-4bit."
        )
    if args.dry_run and args.resume:
        raise ValueError("--resume cannot be combined with --dry-run.")


def get_single_token_id(tokenizer, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Expected {token!r} to map to one token, got {token_ids}."
        )
    return int(token_ids[0])


def extract_steps(record: dict[str, Any]) -> tuple[list[str], str]:
    steps = [str(step) for step in record.get("steps", [])]
    current_step_number = int(record["current_step"])
    current_step_text = str(record.get("current_step_text", ""))

    if not current_step_text:
        if not steps:
            raise ValueError(
                f"Record {record.get('example_id')!r} contains no steps."
            )
        current_step_text = steps[-1]

    previous_count = max(current_step_number - 1, 0)
    previous_steps = steps[:previous_count]

    if previous_steps and previous_steps[-1] == current_step_text:
        previous_steps = previous_steps[:-1]

    return previous_steps, current_step_text


def build_first_pass_messages(
    record: dict[str, Any],
) -> list[dict[str, str]]:
    previous_steps, current_step_text = extract_steps(record)
    previous_text = "\n\n".join(previous_steps)

    assistant_parts = []
    if previous_text:
        assistant_parts.append(previous_text)
    assistant_parts.append(
        "Current Step: "
        + current_step_text
        + " Math reasoning: <extra>, Consistency: <extra>"
    )

    return [
        {
            "role": "user",
            "content": (
                PROMPT_PREFIX
                + "\n\n Question: "
                + str(record["question"])
            ),
        },
        {
            "role": "assistant",
            "content": "\n\n".join(assistant_parts),
        },
    ]


def build_second_pass_messages(
    first_pass_messages: list[dict[str, str]],
    math_token: str,
    consistency_token: str,
) -> list[dict[str, str]]:
    messages = [dict(message) for message in first_pass_messages]
    assistant_text = messages[-1]["content"]
    assistant_text = assistant_text.replace("<extra>", math_token, 1)
    assistant_text = assistant_text.replace(
        "<extra>",
        consistency_token,
        1,
    )

    if "<extra>" in assistant_text:
        raise ValueError("Unexpected extra mask remained after replacement.")

    messages[-1]["content"] = (
        assistant_text + ", Correctness: <extra>"
    )
    return messages


def encode_messages(
    tokenizer,
    messages: list[dict[str, str]],
    max_input_tokens: int,
) -> dict[str, torch.Tensor]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=False,
    )
    input_tokens = int(encoded["input_ids"].shape[1])
    if input_tokens > max_input_tokens:
        raise ValueError(
            f"Prompt has {input_tokens} tokens, exceeding the configured "
            f"limit of {max_input_tokens}. Truncation is intentionally "
            "disabled because it could remove the question or mask tokens."
        )
    return encoded


def shifted_mask_positions(
    input_ids: torch.Tensor,
    mask_token_id: int,
    expected_masks: int,
) -> torch.Tensor:
    token_masks = input_ids == mask_token_id
    actual_masks = int(token_masks.sum().item())
    if actual_masks != expected_masks:
        raise ValueError(
            f"Expected {expected_masks} <extra> masks, found {actual_masks}."
        )

    shifted_mask = torch.cat(
        [
            token_masks[:, 1:],
            torch.zeros(
                token_masks.size(0),
                1,
                dtype=torch.bool,
                device=token_masks.device,
            ),
        ],
        dim=1,
    )
    return shifted_mask


def positive_probabilities(
    logits: torch.Tensor,
    shifted_mask: torch.Tensor,
    positive_token_id: int,
    negative_token_id: int,
) -> torch.Tensor:
    masked_logits = logits[shifted_mask]
    if masked_logits.ndim != 2:
        raise ValueError("Unexpected masked-logit shape.")

    allowed_ids = torch.tensor(
        [positive_token_id, negative_token_id],
        device=logits.device,
    )
    binary_logits = masked_logits[:, allowed_ids]
    return F.softmax(binary_logits.float(), dim=-1)[:, 0]


def run_probability_self_test() -> None:
    fake_logits = torch.zeros((1, 4, 8), dtype=torch.float32)
    fake_mask = torch.tensor(
        [[False, True, False, True]],
        dtype=torch.bool,
    )
    positive_token_id = 2
    negative_token_id = 5

    fake_logits[0, 1, positive_token_id] = 2.0
    fake_logits[0, 1, negative_token_id] = 0.0
    fake_logits[0, 3, positive_token_id] = 0.0
    fake_logits[0, 3, negative_token_id] = 2.0

    probabilities = positive_probabilities(
        logits=fake_logits,
        shifted_mask=fake_mask,
        positive_token_id=positive_token_id,
        negative_token_id=negative_token_id,
    )
    if not (probabilities[0] > 0.8 and probabilities[1] < 0.2):
        raise AssertionError(
            "Positive/negative logit extraction self-test failed."
        )


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def model_input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def run_one(
    record: dict[str, Any],
    model,
    tokenizer,
    positive_token_id: int,
    negative_token_id: int,
    mask_token_id: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    first_messages = build_first_pass_messages(record)
    first_encoded = encode_messages(
        tokenizer,
        first_messages,
        args.max_input_tokens,
    )
    first_input_tokens = int(first_encoded["input_ids"].shape[1])

    device = model_input_device(model)
    first_encoded = {
        key: value.to(device)
        for key, value in first_encoded.items()
    }
    first_shifted_mask = shifted_mask_positions(
        input_ids=first_encoded["input_ids"],
        mask_token_id=mask_token_id,
        expected_masks=2,
    )

    synchronize_cuda()
    start_time = time.perf_counter()

    with torch.inference_mode():
        first_outputs = model(**first_encoded)
        first_probabilities = positive_probabilities(
            logits=first_outputs.logits,
            shifted_mask=first_shifted_mask,
            positive_token_id=positive_token_id,
            negative_token_id=negative_token_id,
        )

    math_probability = float(first_probabilities[0].item())
    consistency_probability = float(first_probabilities[1].item())
    math_prediction = int(math_probability >= 0.5)
    consistency_prediction = int(consistency_probability >= 0.5)
    math_token = "<+>" if math_prediction else "<->"
    consistency_token = "<+>" if consistency_prediction else "<->"

    second_messages = build_second_pass_messages(
        first_pass_messages=first_messages,
        math_token=math_token,
        consistency_token=consistency_token,
    )
    second_encoded = encode_messages(
        tokenizer,
        second_messages,
        args.max_input_tokens,
    )
    second_input_tokens = int(second_encoded["input_ids"].shape[1])
    second_encoded = {
        key: value.to(device)
        for key, value in second_encoded.items()
    }
    second_shifted_mask = shifted_mask_positions(
        input_ids=second_encoded["input_ids"],
        mask_token_id=mask_token_id,
        expected_masks=1,
    )

    with torch.inference_mode():
        second_outputs = model(**second_encoded)
        optimality_probability_tensor = positive_probabilities(
            logits=second_outputs.logits,
            shifted_mask=second_shifted_mask,
            positive_token_id=positive_token_id,
            negative_token_id=negative_token_id,
        )

    synchronize_cuda()
    runtime_seconds = time.perf_counter() - start_time
    optimality_probability = float(
        optimality_probability_tensor[0].item()
    )

    gate_passed = bool(math_prediction and consistency_prediction)
    official_score = optimality_probability if gate_passed else -1.0
    prediction = int(official_score >= args.threshold)
    label = int(record["label"])

    result = dict(record)
    result.update(
        {
            "pathfinder_model": args.model,
            "pathfinder_model_revision": args.revision,
            "pathfinder_attention_implementation": (
                args.attention_implementation
            ),
            "pathfinder_math_probability": math_probability,
            "pathfinder_math_prediction": math_prediction,
            "pathfinder_consistency_probability": (
                consistency_probability
            ),
            "pathfinder_consistency_prediction": (
                consistency_prediction
            ),
            "pathfinder_optimality_probability": (
                optimality_probability
            ),
            "pathfinder_gate_passed": gate_passed,
            "pathfinder_official_score": official_score,
            "pathfinder_threshold": args.threshold,
            "pathfinder_max_input_tokens": args.max_input_tokens,
            "pathfinder_prediction": prediction,
            "pathfinder_correct": prediction == label,
            "pathfinder_first_input_tokens": first_input_tokens,
            "pathfinder_second_input_tokens": second_input_tokens,
            "pathfinder_runtime_seconds": runtime_seconds,
            "pathfinder_quantization": (
                "bitsandbytes_nf4" if args.load_in_4bit else "none"
            ),
        }
    )
    return result


def run_dry_run(
    records: list[dict[str, Any]],
    tokenizer,
    mask_token_id: int,
    positive_token_id: int,
    negative_token_id: int,
    max_input_tokens: int,
) -> None:
    run_probability_self_test()

    maximum_first = (0, "")
    maximum_second = (0, "")
    sample_first_messages: list[dict[str, str]] | None = None
    sample_second_messages: list[dict[str, str]] | None = None

    for record in records:
        first_messages = build_first_pass_messages(record)

        first_encoded = encode_messages(
            tokenizer,
            first_messages,
            max_input_tokens,
        )

        shifted_mask_positions(
            input_ids=first_encoded["input_ids"],
            mask_token_id=mask_token_id,
            expected_masks=2,
        )

        second_messages = build_second_pass_messages(
            first_pass_messages=first_messages,
            math_token="<+>",
            consistency_token="<+>",
        )

        second_encoded = encode_messages(
            tokenizer,
            second_messages,
            max_input_tokens,
        )

        shifted_mask_positions(
            input_ids=second_encoded["input_ids"],
            mask_token_id=mask_token_id,
            expected_masks=1,
        )

        example_id = str(record["example_id"])
        first_length = int(first_encoded["input_ids"].shape[1])
        second_length = int(second_encoded["input_ids"].shape[1])

        if first_length > maximum_first[0]:
            maximum_first = (first_length, example_id)

        if second_length > maximum_second[0]:
            maximum_second = (second_length, example_id)

        if sample_first_messages is None:
            sample_first_messages = first_messages
            sample_second_messages = second_messages

    if sample_first_messages is None or sample_second_messages is None:
        raise AssertionError("Dry run received no records.")

    sample_record = records[0]

    print("PathFinder dry run completed.")
    print("Model weights loaded:          no")
    print("Probability self-test:         passed")
    print(f"Examples validated:            {len(records)}")
    print(f"Sample example ID:             {sample_record['example_id']}")
    print(f"Sample current step:           {sample_record['current_step']}")
    print(
        "Maximum first-pass tokens:    "
        f"{maximum_first[0]} ({maximum_first[1]})"
    )
    print(
        "Maximum second-pass tokens:   "
        f"{maximum_second[0]} ({maximum_second[1]})"
    )
    print("First-pass masks per example:  2")
    print("Second-pass masks per example: 1")
    print(f"<extra> token ID:              {mask_token_id}")
    print(f"<+> token ID:                  {positive_token_id}")
    print(f"<-> token ID:                  {negative_token_id}")
    print()
    print("First-pass assistant suffix:")
    print(sample_first_messages[-1]["content"][-500:])
    print()
    print("Second-pass assistant suffix:")
    print(sample_second_messages[-1]["content"][-160:])


def load_model(args: argparse.Namespace):
    model_kwargs: dict[str, Any] = {
        "device_map": "auto",
        "revision": args.revision,
        "trust_remote_code": True,
        "attn_implementation": args.attention_implementation,
    }

    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as error:
            raise RuntimeError(
                "4-bit loading requires a Transformers version that "
                "provides BitsAndBytesConfig."
            ) from error

        try:
            import bitsandbytes  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "Install bitsandbytes before using --load-in-4bit."
            ) from error

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Full-precision PathFinder inference requires a CUDA GPU. "
                "Use --dry-run locally or run this mode on a Linux CUDA "
                "server GPU."
            )
        model_kwargs["torch_dtype"] = torch.bfloat16

    return AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kwargs,
    ).eval()


def main() -> None:
    args = parse_args()
    validate_args(args)

    records = read_records(args.input)
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise ValueError(f"No input records were found in {args.input}.")

    resume_state = None
    pending_records = records

    if not args.dry_run:
        quantization = (
            "bitsandbytes_nf4" if args.load_in_4bit else "none"
        )
        resume_state = prepare_resume(
            output_path=args.output,
            input_records=records,
            resume=args.resume,
            expected_metadata={
                "pathfinder_model": args.model,
                "pathfinder_model_revision": args.revision,
                "pathfinder_attention_implementation": (
                    args.attention_implementation
                ),
                "pathfinder_threshold": args.threshold,
                "pathfinder_max_input_tokens": args.max_input_tokens,
                "pathfinder_quantization": quantization,
            },
        )
        pending_records = records[resume_state.completed :]

        if args.resume:
            print(
                f"Resume validation passed: {resume_state.completed}/"
                f"{len(records)} records already complete."
            )

        if (
            pending_records
            and not args.load_in_4bit
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "Full-precision PathFinder inference requires a CUDA GPU."
            )

        dtype_name = "float16" if args.load_in_4bit else "bfloat16"
        metadata_path = initialize_run_metadata(
            script_path=Path(__file__),
            repo_root=REPO_ROOT,
            input_path=args.input,
            output_path=args.output,
            selected_examples=len(records),
            model_name=args.model,
            model_revision=args.revision,
            inference_parameters={
                "threshold": args.threshold,
                "max_input_tokens": args.max_input_tokens,
                "attention_implementation": (
                    args.attention_implementation
                ),
                "quantization": quantization,
                "dtype": dtype_name,
                "warmup_examples": args.warmup_examples,
                "timing_scope": "synchronized_two_pass_forward",
            },
            resume=args.resume,
        )

        if not pending_records:
            correct_count = sum(
                int(result["pathfinder_correct"])
                for result in resume_state.existing_records
            )
            total_runtime = sum(
                float(result["pathfinder_runtime_seconds"])
                for result in resume_state.existing_records
            )
            accuracy = correct_count / len(records)
            average_runtime = total_runtime / len(records)
            finalize_run_metadata(
                metadata_path=metadata_path,
                output_path=args.output,
                summary={
                    "examples": len(records),
                    "correct": correct_count,
                    "accuracy": accuracy,
                    "average_runtime_seconds": average_runtime,
                },
            )
            print("PathFinder output is already complete.")
            print(f"Examples:        {len(records)}")
            print(f"Accuracy:        {accuracy:.4f}")
            print(f"Correct:         {correct_count}/{len(records)}")
            print(f"Average runtime: {average_runtime:.4f}s")
            print(f"Saved to:        {args.output.resolve()}")
            print(f"Metadata:        {metadata_path.resolve()}")
            return

    print(f"Loading tokenizer: {args.model}@{args.revision}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=True,
    )
    mask_token_id = get_single_token_id(tokenizer, "<extra>")
    positive_token_id = get_single_token_id(tokenizer, "<+>")
    negative_token_id = get_single_token_id(tokenizer, "<->")

    if args.dry_run:
        run_dry_run(
            records=records,
            tokenizer=tokenizer,
            mask_token_id=mask_token_id,
            positive_token_id=positive_token_id,
            negative_token_id=negative_token_id,
            max_input_tokens=args.max_input_tokens,
        )
        return

    if resume_state is None:
        raise AssertionError("Resume state was not initialized.")

    print(f"Remaining examples: {len(pending_records)}")
    print(f"Loading model: {args.model}@{args.revision}")
    model = load_model(args)

    warmup_count = min(args.warmup_examples, len(pending_records))
    if warmup_count:
        print(f"Warming up on {warmup_count} example(s).")
        for record in pending_records[:warmup_count]:
            run_one(
                record=record,
                model=model,
                tokenizer=tokenizer,
                positive_token_id=positive_token_id,
                negative_token_id=negative_token_id,
                mask_token_id=mask_token_id,
                args=args,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    correct_count = sum(
        int(result["pathfinder_correct"])
        for result in resume_state.existing_records
    )
    runtime_values = [
        float(result["pathfinder_runtime_seconds"])
        for result in resume_state.existing_records
    ]

    with args.output.open(
        resume_state.file_mode,
        encoding="utf-8",
    ) as output_file:
        for index, record in enumerate(
            pending_records,
            start=resume_state.completed + 1,
        ):
            result = run_one(
                record=record,
                model=model,
                tokenizer=tokenizer,
                positive_token_id=positive_token_id,
                negative_token_id=negative_token_id,
                mask_token_id=mask_token_id,
                args=args,
            )
            output_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            output_file.flush()

            correct_count += int(result["pathfinder_correct"])
            runtime_values.append(
                float(result["pathfinder_runtime_seconds"])
            )
            print(
                f"[{index}/{len(records)}] "
                f"{result['example_id']} "
                f"score={result['pathfinder_official_score']:.4f} "
                f"correct={result['pathfinder_correct']}"
            )

    accuracy = correct_count / len(records)
    average_runtime = sum(runtime_values) / len(runtime_values)
    finalize_run_metadata(
        metadata_path=metadata_path,
        output_path=args.output,
        summary={
            "examples": len(records),
            "correct": correct_count,
            "accuracy": accuracy,
            "average_runtime_seconds": average_runtime,
        },
    )
    print()
    print("PathFinder inference completed.")
    print(f"Examples:        {len(records)}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Correct:         {correct_count}/{len(records)}")
    print(f"Average runtime: {average_runtime:.4f}s")
    print(f"Saved to:        {args.output.resolve()}")
    print(f"Metadata:        {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
