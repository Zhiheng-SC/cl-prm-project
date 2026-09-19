"""Validate and run the frozen formal verifier-inference plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    REPO_ROOT / "configs" / "experiments" / "prm_router_formal.json"
)
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "prm_router_formal_split_manifest.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "prm_router" / "formal"
VERIFIER_RUNNER_DIR = REPO_ROOT / "experiments" / "feasibility" / "prm_router"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--phase",
        choices=("development", "test"),
        default="development",
        help="Development runs train and validation; test runs test only.",
    )
    parser.add_argument(
        "--verifiers",
        nargs="+",
        choices=("reasoneval", "pathfinder", "genprm"),
        default=("reasoneval", "pathfinder"),
    )
    parser.add_argument(
        "--reason-eval-threshold",
        type=float,
        default=None,
        help=(
            "Required for test ReasonEval inference after validation "
            "threshold selection."
        ),
    )
    parser.add_argument(
        "--confirm-test-protocol-frozen",
        action="store_true",
        help="Permit test execution after all choices are frozen.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the plan. Without this flag, only print commands.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as input_file:
        content = json.load(input_file)
    if not isinstance(content, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return content


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else REPO_ROOT / path


def validate_split(
    split_name: str,
    manifest: dict[str, Any],
) -> Path:
    try:
        split_manifest = manifest["splits"][split_name]
        path = resolve_repo_path(split_manifest["path"])
        expected_hash = str(split_manifest["sha256"])
        expected_examples = int(split_manifest["examples"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Manifest entry for split {split_name!r} is incomplete."
        ) from error

    if not path.is_file():
        raise FileNotFoundError(
            f"Formal {split_name} split is missing: {path}"
        )

    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            f"SHA-256 mismatch for {split_name}: expected "
            f"{expected_hash}, found {actual_hash}."
        )

    with path.open("r", encoding="utf-8") as input_file:
        actual_examples = sum(1 for line in input_file if line.strip())
    if actual_examples != expected_examples:
        raise ValueError(
            f"Example-count mismatch for {split_name}: expected "
            f"{expected_examples}, found {actual_examples}."
        )

    return path


def relative_display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def add_common_arguments(
    command: list[str],
    *,
    input_path: Path,
    output_path: Path,
    model: dict[str, Any],
    warmup_examples: int,
) -> None:
    command.extend(
        [
            "--input",
            relative_display(input_path),
            "--output",
            relative_display(output_path),
            "--model",
            str(model["name"]),
            "--revision",
            str(model["revision"]),
            "--warmup-examples",
            str(warmup_examples),
            "--resume",
        ]
    )


def build_command(
    *,
    verifier: str,
    split_name: str,
    input_path: Path,
    output_dir: Path,
    config: dict[str, Any],
    reason_eval_threshold: float,
) -> list[str]:
    models = config["models"]
    inference = config["inference"]
    warmup_examples = int(config["runtime"]["warmup_examples"])
    output_path = output_dir / split_name / f"{verifier}.jsonl"

    if verifier == "reasoneval":
        command = [
            sys.executable,
            relative_display(VERIFIER_RUNNER_DIR / "run_disprm.py"),
        ]
        add_common_arguments(
            command,
            input_path=input_path,
            output_path=output_path,
            model=models["base"],
            warmup_examples=warmup_examples,
        )
        command.extend(["--threshold", str(reason_eval_threshold)])
        return command

    if verifier == "pathfinder":
        settings = inference["pathfinder"]
        command = [
            sys.executable,
            relative_display(VERIFIER_RUNNER_DIR / "run_pathfinder.py"),
        ]
        add_common_arguments(
            command,
            input_path=input_path,
            output_path=output_path,
            model=models["primary_second_stage"],
            warmup_examples=warmup_examples,
        )
        command.extend(
            [
                "--threshold",
                str(settings["threshold"]),
                "--max-input-tokens",
                str(settings["max_input_tokens"]),
                "--attention-implementation",
                str(settings["attention_implementation"]),
            ]
        )
        if bool(settings["load_in_4bit"]):
            command.append("--load-in-4bit")
        return command

    if verifier == "genprm":
        settings = inference["genprm"]
        command = [
            sys.executable,
            relative_display(VERIFIER_RUNNER_DIR / "run_genprm.py"),
        ]
        add_common_arguments(
            command,
            input_path=input_path,
            output_path=output_path,
            model=models["alternative_second_stage"],
            warmup_examples=warmup_examples,
        )
        command.extend(
            [
                "--seed",
                str(settings["seed"]),
                "--max-analysis-tokens",
                str(settings["max_analysis_tokens"]),
                "--max-input-tokens",
                str(settings["max_input_tokens"]),
            ]
        )
        return command

    raise ValueError(f"Unsupported verifier: {verifier}")


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    manifest = read_json(args.manifest)

    verifiers = list(dict.fromkeys(args.verifiers))
    split_names = (
        ("train", "validation")
        if args.phase == "development"
        else ("test",)
    )

    if (
        args.phase == "test"
        and args.execute
        and not args.confirm_test_protocol_frozen
    ):
        raise ValueError(
            "Test execution is locked. Freeze the validation-selected "
            "threshold, router, arbitration rule, and analysis plan, then "
            "add --confirm-test-protocol-frozen."
        )

    if "reasoneval" in verifiers:
        if args.phase == "test":
            if args.reason_eval_threshold is None:
                raise ValueError(
                    "--reason-eval-threshold is required for the test phase."
                )
            reason_eval_threshold = args.reason_eval_threshold
        else:
            reason_eval_threshold = float(
                config["inference"]["reason_eval"][
                    "development_output_threshold"
                ]
            )
    else:
        reason_eval_threshold = 0.5

    if not 0.0 <= reason_eval_threshold <= 1.0:
        raise ValueError("ReasonEval threshold must be between 0 and 1.")

    input_paths = {
        split_name: validate_split(split_name, manifest)
        for split_name in split_names
    }

    commands = [
        build_command(
            verifier=verifier,
            split_name=split_name,
            input_path=input_paths[split_name],
            output_dir=args.output_dir,
            config=config,
            reason_eval_threshold=reason_eval_threshold,
        )
        for verifier in verifiers
        for split_name in split_names
    ]

    print("Formal verifier-inference plan")
    print(f"Phase:       {args.phase}")
    print(f"Splits:      {', '.join(split_names)}")
    print(f"Verifiers:   {', '.join(verifiers)}")
    print(f"Execute:     {args.execute}")
    print()

    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {shlex.join(command)}")

    if not args.execute:
        print()
        print("Plan only; no model inference was started.")
        print("Add --execute to run these commands sequentially.")
        return

    for index, command in enumerate(commands, start=1):
        print()
        print(f"Running command {index}/{len(commands)}")
        subprocess.run(command, cwd=REPO_ROOT, check=True)

    print()
    print("Formal verifier inference completed.")


if __name__ == "__main__":
    main()
