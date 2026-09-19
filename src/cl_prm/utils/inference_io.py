"""Shared validation for safely resuming JSONL verifier inference."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


@dataclass
class ResumeState:
    """Validated state for an inference output file."""

    existing_records: list[dict[str, Any]]
    completed: int
    file_mode: str


def record_key(record: dict[str, Any]) -> tuple[str, int]:
    """Return the stable example-and-step key used by inference outputs."""
    try:
        return str(record["example_id"]), int(record["current_step"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Every inference record must contain example_id and current_step."
        ) from error


def read_output_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read an existing JSONL output, allowing an empty file."""
    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}. "
                    "The output may end with a partial record; inspect it "
                    "before resuming."
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object on line {line_number} of {path}."
                )
            records.append(record)

    return records


def ensure_unique_keys(
    records: list[dict[str, Any]],
    description: str,
) -> list[tuple[str, int]]:
    """Return record keys after rejecting duplicates."""
    keys = [record_key(record) for record in records]
    seen: set[tuple[str, int]] = set()

    for key in keys:
        if key in seen:
            raise ValueError(
                f"Duplicate record key {key!r} in {description}."
            )
        seen.add(key)

    return keys


def prepare_resume(
    *,
    output_path: Path,
    input_records: list[dict[str, Any]],
    resume: bool,
    expected_metadata: dict[str, Any],
) -> ResumeState:
    """Validate an existing output as an exact prefix of the input."""
    if not resume or not output_path.exists():
        return ResumeState(
            existing_records=[],
            completed=0,
            file_mode="w",
        )

    existing_records = read_output_jsonl(output_path)

    if len(existing_records) > len(input_records):
        raise ValueError(
            f"Existing output has {len(existing_records)} records, but "
            f"the selected input has only {len(input_records)}."
        )

    input_keys = ensure_unique_keys(input_records, "selected input")
    output_keys = ensure_unique_keys(existing_records, "existing output")

    for index, output_record in enumerate(existing_records):
        expected_record = input_records[index]
        if output_keys[index] != input_keys[index]:
            raise ValueError(
                "Existing output is not an exact prefix of the selected "
                f"input at position {index + 1}: expected "
                f"{input_keys[index]!r}, found {output_keys[index]!r}."
            )

        if int(output_record["label"]) != int(expected_record["label"]):
            raise ValueError(
                f"Label mismatch at position {index + 1} for "
                f"{output_keys[index]!r}."
            )

        for field, expected_value in expected_metadata.items():
            if field not in output_record:
                raise ValueError(
                    f"Existing output is missing resume metadata {field!r} "
                    f"at position {index + 1}."
                )
            if output_record[field] != expected_value:
                raise ValueError(
                    f"Resume metadata mismatch for {field!r} at position "
                    f"{index + 1}: expected {expected_value!r}, found "
                    f"{output_record[field]!r}."
                )

    return ResumeState(
        existing_records=existing_records,
        completed=len(existing_records),
        file_mode="a",
    )


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metadata_path_for_output(output_path: Path) -> Path:
    """Return the sidecar metadata path for a JSONL output."""
    return output_path.with_suffix(".metadata.json")


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def package_version(distribution: str) -> str | None:
    """Return an installed distribution version when available."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    """Collect the software and accelerator environment."""
    import torch

    gpu_devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu_devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                }
            )

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in (
                "torch",
                "transformers",
                "accelerate",
                "flash-attn",
                "bitsandbytes",
            )
        },
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version()
            if torch.cuda.is_available()
            else None
        ),
        "gpu_devices": gpu_devices,
    }


def git_state(repo_root: Path) -> dict[str, Any]:
    """Return the current Git commit and tracked-worktree state."""
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "tracked_worktree_dirty": None}

    return {
        "commit": commit_result.stdout.strip(),
        "tracked_worktree_dirty": bool(status_result.stdout.strip()),
    }


def write_json_atomically(path: Path, content: dict[str, Any]) -> None:
    """Replace a JSON file only after its complete content is written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(content, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def read_metadata(path: Path) -> dict[str, Any]:
    """Read and validate a metadata sidecar."""
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read run metadata from {path}.") from error

    if not isinstance(content, dict):
        raise ValueError(f"Run metadata in {path} is not a JSON object.")
    return content


def initialize_run_metadata(
    *,
    script_path: Path,
    repo_root: Path,
    input_path: Path,
    output_path: Path,
    selected_examples: int,
    model_name: str,
    model_revision: str,
    inference_parameters: dict[str, Any],
    resume: bool,
) -> Path:
    """Create or validate the reproducibility sidecar for a run."""
    environment = collect_environment()
    signature = {
        "script_sha256": sha256_file(script_path),
        "input_sha256": sha256_file(input_path),
        "selected_examples": selected_examples,
        "model_name": model_name,
        "model_revision": model_revision,
        "inference_parameters": inference_parameters,
        "environment": environment,
    }
    metadata_path = metadata_path_for_output(output_path)
    timestamp = utc_now()

    if resume and output_path.exists():
        if not metadata_path.exists():
            raise ValueError(
                f"Cannot resume because metadata is missing: {metadata_path}"
            )
        metadata = read_metadata(metadata_path)
        if metadata.get("run_signature") != signature:
            raise ValueError(
                "Cannot resume because the run metadata does not match "
                "the current input, script, model, parameters, or "
                "execution environment."
            )
        metadata["status"] = "running"
        metadata["last_resumed_at_utc"] = timestamp
        metadata["resume_count"] = int(metadata.get("resume_count", 0)) + 1
    else:
        try:
            script_display = script_path.resolve().relative_to(
                repo_root.resolve()
            ).as_posix()
        except ValueError:
            script_display = script_path.as_posix()

        metadata = {
            "schema_version": 1,
            "status": "running",
            "started_at_utc": timestamp,
            "completed_at_utc": None,
            "resume_count": 0,
            "script": script_display,
            "input": {
                "path": input_path.as_posix(),
                "sha256": signature["input_sha256"],
                "selected_examples": selected_examples,
            },
            "output": {
                "path": output_path.as_posix(),
                "sha256": None,
            },
            "model": {
                "name": model_name,
                "revision": model_revision,
            },
            "inference_parameters": inference_parameters,
            "environment": environment,
            "git": git_state(repo_root),
            "command": [sys.executable, *sys.argv],
            "run_signature": signature,
            "summary": None,
        }

    write_json_atomically(metadata_path, metadata)
    return metadata_path


def finalize_run_metadata(
    *,
    metadata_path: Path,
    output_path: Path,
    summary: dict[str, Any],
) -> None:
    """Mark a run complete and record its output digest and summary."""
    metadata = read_metadata(metadata_path)
    metadata["status"] = "completed"
    metadata["completed_at_utc"] = utc_now()
    metadata["output"]["sha256"] = sha256_file(output_path)
    metadata["summary"] = summary
    write_json_atomically(metadata_path, metadata)
