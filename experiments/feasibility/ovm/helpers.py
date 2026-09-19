import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def time_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hashing_func(path) -> str:
    digest = hashlib.sha256()
    # hash on chunks
    with open(path, "rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path) -> dict:
    try:
        with open(path, "r") as f:
            value = json.load(f)
    except Exception:
        raise Exception("error when reading file")

    return value


def write_json(path, value) -> None:
    if not isinstance(path, Path):
        path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "w") as f:
        json_dump = json.dumps(value, indent=2)
        f.write(json_dump + "\n")
    temporary.replace(path)


def read_jsonl(path) -> list[dict]:
    records = []
    with open(path, "r") as input_file:
        lines = input_file.read().split("\n")
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"Unexpected data entry in {path}")
            except json.JSONDecodeError:
                raise ValueError(f"Error when reading lines from {path}")
            records.append(record)
    return records


def initialize_seed_run(output_path, config_path, pilot_path, seed, selected_examples, model_snapshot) -> Path:
    path = output_path.with_suffix(".metadata.json")

    signature = {
        "config_hash": hashing_func(config_path),
        "pilot_hash": hashing_func(pilot_path),
        "seed": seed,
        "selected_examples": selected_examples,
        "model_snapshot": str(model_snapshot),
    }

    content = {
        "schema_version": 1,
        "status": "started",
        "started_at": time_now(),
        "output_path": str(output_path),
        "signature": signature
    }
    write_json(path, content)
    return path


def finalize_seed_run(path, output_path, summary) -> None:
    content = read_json(path)
    content["status"] = "completed"
    content["completed_at"] = time_now()
    content["output_hashed"] = hashing_func(output_path)
    content["summary"] = summary

    write_json(path, content)
