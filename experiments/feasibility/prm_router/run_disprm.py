"""Run ReasonEval-7B on the PRMBench smoke-test subset.

ReasonEval outputs three probabilities for every reasoning step:

    [negative, neutral, positive]

We define validity as:

    P(correct) = P(neutral) + P(positive)

A step is predicted as correct when P(correct) >= 0.5.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    MistralModel,
    MistralPreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig

from inference_io import prepare_resume


DEFAULT_MODEL = "GAIR/ReasonEval-7B"
DEFAULT_REVISION = "0a6556ef5c937bb17d265ba681b501fd60056cfe"


class ReasonEval7B(MistralPreTrainedModel):
    """ReasonEval-7B architecture from the official implementation."""

    _keys_to_ignore_on_load_missing = ["lm_head.weight"]

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)

        self.model = MistralModel(config)

        self.score_head = nn.Linear(
            config.hidden_size,
            config.score_dimension,
            bias=config.use_bias,
        )

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        scores = self.score_head(hidden_states)

        return scores


def load_jsonl(path: Path) -> list[dict]:
    """Load UTF-8 JSONL records."""
    records = []

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from error

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def prepare_reason_eval_input(
    tokenizer,
    question: str,
    reasoning_steps: list[str],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Prepare ReasonEval input following its official implementation."""
    prompt_format = (
        "Question:\n"
        "{question}\n"
        "Answer:\n"
        "Let's think step by step.\n"
    )

    step_separator = tokenizer.pad_token

    if step_separator is None:
        raise ValueError(
            "The tokenizer has no pad token, but ReasonEval requires one "
            "as its step separator."
        )

    combined_steps = ""

    for step in reasoning_steps:
        combined_steps += str(step).strip() + step_separator

    prompt = prompt_format.format(question=question.strip())

    tokenized_result = tokenizer(
        prompt + step_separator + combined_steps
    )["input_ids"]

    separator_token_id = tokenizer(step_separator)["input_ids"][-1]

    labeled_token_indices = []
    adjusted_token_ids = []
    separator_count = 0

    for index, token_id in enumerate(tokenized_result):
        if token_id == separator_token_id:
            labeled_token_indices.append(
                index - 1 - separator_count
            )
            separator_count += 1
        else:
            adjusted_token_ids.append(token_id)

    # Official ReasonEval-7B preprocessing adjustments.
    adjusted_token_ids = [1] + adjusted_token_ids
    labeled_token_indices = labeled_token_indices[2:]

    if len(labeled_token_indices) != len(reasoning_steps):
        raise ValueError(
            "Could not align ReasonEval scores with reasoning steps: "
            f"{len(labeled_token_indices)} score positions for "
            f"{len(reasoning_steps)} steps."
        )

    input_ids = torch.tensor(
        [adjusted_token_ids],
        dtype=torch.long,
    )

    attention_mask = torch.ones_like(
        input_ids,
        dtype=torch.bool,
    )

    return input_ids, attention_mask, labeled_token_indices


def synchronize_if_needed(device: torch.device) -> None:
    """Synchronize CUDA so runtime measurements are accurate."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_record(
    model,
    tokenizer,
    record: dict,
    model_name: str,
    model_revision: str,
    threshold: float,
) -> dict:
    """Evaluate the current step of one PRMBench record."""
    question = record["question"]
    reasoning_steps = record["steps"]

    input_ids, attention_mask, labeled_indices = (
        prepare_reason_eval_input(
            tokenizer=tokenizer,
            question=question,
            reasoning_steps=reasoning_steps,
        )
    )

    model_device = model.model.embed_tokens.weight.device

    input_ids = input_ids.to(model_device)
    attention_mask = attention_mask.to(model_device)

    synchronize_if_needed(model_device)
    start_time = time.perf_counter()

    all_scores = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    synchronize_if_needed(model_device)
    runtime_seconds = time.perf_counter() - start_time

    current_step_index = labeled_indices[-1]
    current_step_logits = all_scores[0, current_step_index, :]
    probabilities = torch.softmax(
        current_step_logits.float(),
        dim=-1,
    ).cpu()

    if probabilities.numel() != 3:
        raise ValueError(
            "Expected three ReasonEval probabilities "
            "[negative, neutral, positive], but received "
            f"{probabilities.numel()}."
        )

    probability_negative = float(probabilities[0])
    probability_neutral = float(probabilities[1])
    probability_positive = float(probabilities[2])

    validity_score = probability_neutral + probability_positive
    prediction = int(validity_score >= threshold)
    ground_truth = int(record["label"])

    result = dict(record)

    result.update(
        {
            "disprm_model": model_name,
            "disprm_model_revision": model_revision,
            "disprm_threshold": threshold,
            "disprm_probability_negative": probability_negative,
            "disprm_probability_neutral": probability_neutral,
            "disprm_probability_positive": probability_positive,
            "disprm_score": validity_score,
            "disprm_prediction": prediction,
            "disprm_correct": prediction == ground_truth,
            "disprm_runtime_seconds": runtime_seconds,
            "disprm_input_tokens": int(input_ids.shape[1]),
        }
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/prm_router/smoke_test.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/prm_router/reasoneval_smoke_test.jsonl"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=DEFAULT_REVISION,
        help="Pinned Hugging Face model revision.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for the DisPRM validity score.",
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
        "--allow-cpu",
        action="store_true",
        help="Allow CPU inference. This will be very slow.",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    records = load_jsonl(args.input)
    if args.limit is not None:
        records = records[: args.limit]

    resume_state = prepare_resume(
        output_path=args.output,
        input_records=records,
        resume=args.resume,
        expected_metadata={
            "disprm_model": args.model,
            "disprm_model_revision": args.revision,
            "disprm_threshold": args.threshold,
        },
    )
    pending_records = records[resume_state.completed :]

    if args.resume:
        print(
            f"Resume validation passed: {resume_state.completed}/"
            f"{len(records)} records already complete."
        )

    correct_count = sum(
        int(result["disprm_correct"])
        for result in resume_state.existing_records
    )
    total_runtime = sum(
        float(result["disprm_runtime_seconds"])
        for result in resume_state.existing_records
    )

    if not pending_records:
        accuracy = correct_count / len(records)
        average_runtime = total_runtime / len(records)
        print("ReasonEval output is already complete.")
        print(f"Accuracy:        {accuracy:.4f}")
        print(f"Correct:         {correct_count}/{len(records)}")
        print(f"Average runtime: {average_runtime:.4f}s")
        print(f"Saved to:        {args.output.resolve()}")
        return

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "No CUDA GPU detected. ReasonEval-7B should be run on "
            "the A100/Linux environment. Use --allow-cpu only if "
            "you intentionally want very slow CPU inference."
        )

    device_description = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    print(f"Device: {device_description}")
    print(f"Input examples: {len(records)}")
    print(f"Remaining examples: {len(pending_records)}")
    print(f"Loading tokenizer: {args.model}@{args.revision}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
    )

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available()
        else torch.float32
    )

    print(f"Loading model with dtype={dtype}")
    model_load_start = time.perf_counter()

    model = ReasonEval7B.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    model_load_seconds = time.perf_counter() - model_load_start
    print(f"Model loaded in {model_load_seconds:.2f} seconds.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open(
        resume_state.file_mode,
        encoding="utf-8",
    ) as output_file:
        for index, record in enumerate(
            pending_records,
            start=resume_state.completed + 1,
        ):
            result = evaluate_record(
                model=model,
                tokenizer=tokenizer,
                record=record,
                model_name=args.model,
                model_revision=args.revision,
                threshold=args.threshold,
            )

            correct_count += int(result["disprm_correct"])
            total_runtime += result["disprm_runtime_seconds"]

            output_file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            output_file.flush()

            print(
                f"[{index}/{len(records)}] "
                f"id={result['example_id']} "
                f"label={result['label']} "
                f"prediction={result['disprm_prediction']} "
                f"score={result['disprm_score']:.4f} "
                f"correct={result['disprm_correct']} "
                f"time={result['disprm_runtime_seconds']:.4f}s"
            )

    accuracy = correct_count / len(records)
    average_runtime = total_runtime / len(records)

    print()
    print("ReasonEval inference completed.")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Correct:         {correct_count}/{len(records)}")
    print(f"Average runtime: {average_runtime:.4f}s")
    print(f"Saved to:        {args.output.resolve()}")


if __name__ == "__main__":
    main()
