import os

os.environ["OVM_USE_FLASH_ATTN"] = "0"
DEVICE = "cuda:0"

import argparse
import hashlib
import gc
from collections import defaultdict
import json
import random
import re
import time
from tqdm import tqdm
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import torch
import numpy
from accelerate import Accelerator

# from OVM repo
from FreedomIntelligence_OVM.utils.gsm8k.decoding import extract_answer, INVALID_ANS
from FreedomIntelligence_OVM.utils.sampling import SamplingWithCalculator
from FreedomIntelligence_OVM.utils.verifier_models import load_generator_and_verifier

from helpers import read_json, read_jsonl, hashing_func, initialize_seed_run, finalize_seed_run


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stable_seed(seed: int, source_index: int, method: str) -> int:
    """reproducible stable seed based on experiment details"""
    value = f"{seed}:{source_index}:{method}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def standardize_ground_truth(text: str) -> str:
    # capture everything after ####
    pattern = r"####\s*([\-0-9.,]+)"
    matches = re.findall(pattern, text)

    if not matches:
        raise ValueError(f"GSM8K ground truth answer extraction unsuccessful: {text}")

    value = matches[-1].replace(",", "")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value


def majority_answer(answers: list[str]) -> str:
    valid_answers = [answer for answer in answers if answer != INVALID_ANS]
    if not valid_answers:
        return INVALID_ANS

    counts, first_position = defaultdict(int), defaultdict(int)
    for index, answer in enumerate(valid_answers):
        counts[answer] += 1
        first_position.setdefault(answer, index)

    # resolve ties according to first occurrence
    return max(counts, key=lambda answer: (counts[answer], -first_position[answer]))


def call(func) -> tuple[Any, dict]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    value = func()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    return value, {
        "runtime_seconds": elapsed,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def sample_responses(sampler: Any, question: str, count: int, batch_size: int) -> list[str]:
    responses = []
    while len(responses) < count:
        batch = sampler.sample([question] * min(batch_size, count - len(responses)))
        responses.extend(batch)
    return responses


def _to_input_ids(tokenizer: Any, text: str, **kwargs) -> list:
    return tokenizer(text, **kwargs).input_ids


def score_responses(verifier: Any, tokenizer: Any, question: str, responses: list, batch_size: int) -> list:
    question_tokens = _to_input_ids(tokenizer, question, padding=False)

    all_tokens = [
        question_tokens + _to_input_ids(tokenizer, response, add_special_tokens=False) + [tokenizer.eos_token_id] for
        response in responses]

    scores = []
    for start in range(0, len(all_tokens), batch_size):
        token_batch = all_tokens[start: start + batch_size]
        max_per_batch = max(len(tokens) for tokens in token_batch)

        input_ids = torch.full(
            (len(token_batch), max_per_batch),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=DEVICE,
        )

        for row, tokens in enumerate(token_batch):
            input_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
        # score with verifier
        scores.extend(score for score in verifier.scoring_sequences(input_ids).tolist())

    return scores


def candidate_records(candidates: list, extract_answer_func: Any, ground_truth: str) -> list:
    records = []
    for candidate in candidates:
        response = candidate["str"]
        answer = extract_answer_func(response)

        records.append(
            {
                "response": response,
                "answer": answer,
                "correct": answer == ground_truth,
                "value_score": candidate["vscore"],
            }
        )
    return records


def summary(seq: list) -> list:
    summaries = []
    for depth, item_ in enumerate(seq):
        candidates = item_.get("sequences", [])
        scores = [candidate["vscore"] for candidate in candidates]
        if scores:
            min_, max_, mean_ = min(scores), max(scores), sum(scores) / len(scores)
        else:
            min_, max_, mean_ = None, None, None

        summary_dict = {
            "depth": depth + 1,
            "candidate_count": len(candidates),
            "chosen_indices": [int(value) for value in item_.get("choices", [])],
            "min": min_,
            "mean": mean_,
            "max": max_
        }

        summaries.append(summary_dict)

    return summaries


def mean(records: list[dict]) -> float:
    if not records:
        return 0.0

    total_correct = 0
    for record in records:
        if record["correct"]:
            total_correct += 1

    return total_correct / len(records)


def acc(records: list[dict], method: str) -> float:
    valid_methods = ["greedy", "sampling", "orm", "ovm"]
    if method not in valid_methods:
        raise ValueError(f"method {method} MUST be one of {valid_methods}")
    if not records:
        return 0.0

    correct = 0
    for record in records:
        if method != "sampling":
            correct += record[method]["correct"]
        else:
            correct += record[method]["majority_correct"]

    return correct / len(records)


def build_generation_args(config: dict, do_sample: bool) -> SimpleNamespace:
    # taken from the config
    inference = config["inference"]
    return SimpleNamespace(
        do_sample=do_sample,
        num_beams=1,
        temperature=inference["temperature"],
        top_k=inference["top_k"],
        top_p=float(inference["top_p"]),
        repetition_penalty=1.0,
        length_penalty=1.0,
        max_length=2048,
        max_new_tokens=int(inference["maximum_new_tokens"]),
    )


def run_example(record: dict, seed: int, config: dict, greedy_sampler: Any, sampling_sampler: Any, verifier: Any,
                tokenizer: Any, extract_answer: Any) -> dict:
    inference = config["inference"]
    pilot_idx = record["pilot_index"]
    question, ground_truth_answer = str(record["question"]).rstrip() + "\n", standardize_ground_truth(
        str(record["answer"]))
    source_index, sampling_size = record["source_index"], inference["sampling_size"]
    sampling_batch_size, verifier_batch_size = inference["sampling_batch_size"], inference["verifier_batch_size"]
    beam_size = inference["beam_size"]

    # Greedy
    set_random_seed(stable_seed(seed, record["source_index"], "greedy"))
    greedy_response, greedy_resources = call(lambda: greedy_sampler.sample([question])[0])
    greedy_answer = extract_answer(greedy_response)

    # Self-Consistency
    set_random_seed(stable_seed(seed, record["source_index"], "sampling"))
    sampled_responses, sampling_resources = call(
        lambda: sample_responses(sampling_sampler, question, sampling_size, sampling_batch_size))

    sampled_answers = [extract_answer(response) for response in sampled_responses]
    sampled_outputs = [
        {
            "response": response,
            "answer": answer,
            "correct": answer == ground_truth_answer,
        }
        for response, answer in zip(sampled_responses, sampled_answers)
    ]
    majority = majority_answer(sampled_answers)

    # ORM
    set_random_seed(stable_seed(seed, record["source_index"], "orm"))
    orm_scores, orm_resources = call(
        lambda: score_responses(verifier, tokenizer, question, sampled_responses, verifier_batch_size))
    # ORM-Post Selection
    orm_index = max(range(len(orm_scores)), key=orm_scores.__getitem__)

    # OVM
    set_random_seed(stable_seed(seed, record["source_index"], "ovm"))
    (ovm_response, intermediates), ovm_resources = call(
        lambda: sampling_sampler.sample_by_steps(
            qn_str=question,
            batch_size=inference["step_sampling_batch_size"],
            vs_batch_size=verifier_batch_size,
            n_beam=beam_size,
            n_sampling_steps=sampling_size,
            max_n_step=inference["maximum_steps"],
            max_step_length=inference["maximum_step_tokens"],
            inference_mode="beam",
            dedup_mode=int(bool(inference["deduplicate_candidates"])),
        ),
    )
    ovm_answer = extract_answer(ovm_response)
    beam_source = intermediates[-1].get("sequences", [])
    if len(intermediates) > 1:
        expansion_source = intermediates[-2].get("sequences", [])
    else:
        expansion_source = beam_source.copy()

    # last beam and expansion path
    final_beam = candidate_records(beam_source, extract_answer, ground_truth_answer)
    last_expansion = candidate_records(expansion_source, extract_answer, ground_truth_answer)

    # build response
    response_dict = {
        "pilot_index": pilot_idx,
        "source_index": source_index,
        "seed": seed,
        "question": question.strip(),
        "ground_truth": ground_truth_answer
    }

    greedy = {
        "response": greedy_response,
        "answer": greedy_answer,
        "correct": greedy_answer == ground_truth_answer,
        **greedy_resources
    }

    self_consistency = {
        "sampling_size": sampling_size,
        "outputs": sampled_outputs,
        "majority_answer": majority,
        "majority_correct": majority == ground_truth_answer,
        "correct_mean": mean(sampled_outputs),
        **sampling_resources
    }

    orm = {
        "scores": orm_scores,
        "selected_index": orm_index,
        "selected_answer": sampled_answers[orm_index],
        "correct": sampled_answers[orm_index] == ground_truth_answer,
        **orm_resources,
    }

    ovm = {
        "sampling_size": sampling_size,
        "beam_size": beam_size,
        "response": ovm_response,
        "answer": ovm_answer,
        "correct": ovm_answer == ground_truth_answer,
        "last_expansion": last_expansion,
        "last_expansion_correct_mean": mean(last_expansion),
        "final_beam": final_beam,
        "final_beam_correct_mean": mean(final_beam),
        "summary": summary(intermediates),
        **ovm_resources,
    }

    response_dict.update(
        {
            "greedy": greedy,
            "sampling": self_consistency,
            "orm": orm,
            "ovm": ovm,
        }
    )

    return response_dict


def main() -> None:
    # define the inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True, help="Access to model weights")
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    # load and validate inputs
    config, manifest = read_json(args.config), read_json(args.pilot_manifest)
    model_snapshot = args.model_snapshot
    pilot_path = Path(manifest["pilot_path"])

    if hashing_func(pilot_path) != manifest["pilot_sha256"]:
        raise ValueError(
            f"Manifest hashes do not match: expected -- {str(manifest['pilot_sha256'])}, actual -- {hashing_func(pilot_path)}")

    pilot_records = read_jsonl(pilot_path)
    if not pilot_records:
        raise ValueError(f"Pilot study with no records, first try `prepare_pilot.py`")

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        # slice records e.g. useful for smoke-tests
        pilot_records = pilot_records[: args.limit]

    # the study is intended for A100 GPU with 80GBs ~ my Colab Pro
    # can be more lenient here
    if not torch.cuda.is_available():
        raise RuntimeError("OVM Feasibility Study requires a A100 80GBs GPU")

    name = torch.cuda.get_device_name(0)
    memory_gib = torch.cuda.get_device_properties(0).total_memory / (2 ** 30)

    print(f"GPU: {name} with {memory_gib} GiB memory on device: {DEVICE}")

    if "A100" not in name or memory_gib < 75.0:
        raise RuntimeError("Your environment does not satisfy the requirements")

    torch.backends.cuda.matmul.allow_tf32 = True

    # prepare model loading
    models = config["models"]
    snapshot_manifest_path = model_snapshot / "ovm_snapshot_manifest.json"
    if not os.path.exists(snapshot_manifest_path):
        raise FileNotFoundError("Models manifest not found. Run again `download_models.py`")

    # validate models manifest
    snapshot_manifest = read_json(snapshot_manifest_path)
    generator_path = model_snapshot / str(models["generator_subdir"])
    verifier_path = model_snapshot / str(models["verifier_subdir"])

    for path in [generator_path, verifier_path]:

        # compare the expected against the actual hash produced
        expected_config_hash = snapshot_manifest["subdirectories"][path.name]["config_sha256"]
        actual_config_hash = hashing_func(path / "config.json")

        if actual_config_hash != expected_config_hash:
            raise ValueError(f"Model config hash mismatch: {path}")

    print("Loading Generator and Verifier checkpoints...")

    # model loading

    ## accelerator device
    accelerator = Accelerator(gradient_accumulation_steps=1, even_batches=True)

    generator, verifier, tokenizer = load_generator_and_verifier(SimpleNamespace(
        model_name_or_path=str(generator_path),
        verifier_model_name_or_path=str(verifier_path),
        fp16=False,
    ))
    generator.eval().cuda(device=DEVICE)
    verifier.eval().cuda(device=DEVICE)

    # Greedy
    greedy_sampler = SamplingWithCalculator(
        accelerator=accelerator,
        model=generator,
        verifier=verifier,
        tokenizer=tokenizer,
        generation_args=build_generation_args(config, do_sample=False),
    )

    sampling_sampler = SamplingWithCalculator(
        accelerator=accelerator,
        model=generator,
        verifier=verifier,
        tokenizer=tokenizer,
        generation_args=build_generation_args(config, do_sample=True),
    )

    # if not provided, seeds are taken from config, MUST be unique
    seeds = args.seeds if args.seeds is not None else [int(value) for value in config["inference"]["seeds"]]

    if len(set(seeds)) != len(seeds):
        raise ValueError("Provided seeds are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    warmup_count = config["inference"]["warmup_count"]
    if warmup_count > 0:
        if warmup_count > len(pilot_records):
            raise ValueError("Provided warmup_count > pilot_records")

        for _ in range(warmup_count):
            print(f"Warmup Example {_ + 1}/{warmup_count}")
            run_example(
                record=pilot_records[_],
                seed=seeds[0],
                config=config,
                greedy_sampler=greedy_sampler,
                sampling_sampler=sampling_sampler,
                verifier=verifier,
                tokenizer=tokenizer,
                extract_answer=extract_answer,
            )

    print("Warmup completed. Initiating Inference\n")
    for seed in tqdm(seeds, desc="Running seed..."):
        output_path = args.output_dir / f"seed_{seed}.jsonl"

        # metadata -> Path
        metadata_path = initialize_seed_run(
            output_path=output_path,
            config_path=args.config,
            pilot_path=pilot_path,
            seed=seed,
            selected_examples=len(pilot_records),
            model_snapshot=model_snapshot
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = []
        with open(output_path, "w") as output_file:
            for offset, record in enumerate(pilot_records):
                # visibility
                print(f"[{offset + 1}/{len(pilot_records)}]\nseed={seed} source_index={record['source_index']}")

                # result per example
                result = run_example(
                    record=record,
                    seed=seed,
                    config=config,
                    greedy_sampler=greedy_sampler,
                    sampling_sampler=sampling_sampler,
                    verifier=verifier,
                    tokenizer=tokenizer,
                    extract_answer=extract_answer
                )

                results.append(result)
                output_file.write(json.dumps(result, ensure_ascii=True) + "\n")

                print(f"Greedy: {result['greedy']['correct']}\n"
                      f"Self-Consistency: {result['sampling']['majority_correct']}\n"
                      f"ORM Post-Selection: {result['orm']['correct']}\n"
                      f"OVM: {result['ovm']['correct']}\n")

        summary = {
            "examples": len(results),
            "greedy_acc": acc(results, "greedy"),
            "self_consistency_acc": acc(results, "sampling"),
            "orm_acc": acc(results, "orm"),
            "ovm_acc": acc(results, "ovm")
        }

        finalize_seed_run(metadata_path, output_path, summary)

        print(f"seed {seed} results:\nexamples: {len(results)}\n")

    print("OVM feasibility seeds completed")


if __name__ == "__main__":
    main()
