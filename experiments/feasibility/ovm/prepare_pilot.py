import argparse
import json
import random
import os
from pathlib import Path

from helpers import read_json, read_jsonl, hashing_func, write_json


def select_pilot_indices(total, size, seed) -> list:
    # validate dimensions
    if size < 1 or size > total:
        raise ValueError(f"Pilot size must be within [1, {total}], but the given is {size}")

    sampler = random.Random(seed)
    indices = sampler.sample(range(total), size)

    # sampled indices
    return sorted(indices)


def prepare_pilot(config_path, output_dir) -> tuple[Path, Path]:
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    # prepare dir
    output_dir.mkdir(parents=True, exist_ok=True)

    config = read_json(config_path)
    dataset_config = config["dataset"]
    source_path = dataset_config["source"]

    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Missing GSM8K test data at: {source_path}")

    # hash check
    actual_hash = hashing_func(source_path)
    expected_hash = str(dataset_config["source_sha256"])

    if actual_hash != expected_hash:
        raise ValueError(f"GSM8K hashes mismatch: actual -- {actual_hash} != expected -- {expected_hash}")

    records = read_jsonl(source_path)
    expected_total = int(dataset_config["source_examples"])

    # we want the records to match to ensure proper loading
    if len(records) != expected_total:
        raise ValueError(f"Records mismatch: actual -- {len(records)} != expected -- {expected_total}")

    # from config
    size, seed = int(dataset_config["pilot_examples"]), int(dataset_config["pilot_seed"])

    # our samples for the pilot study
    indices = select_pilot_indices(len(records), size, seed)

    # prepare data
    save_pilot_path = output_dir / "test.jsonl"

    with open(save_pilot_path, "w") as output_file:
        for pilot_idx, src_idx in enumerate(indices):
            record = {"pilot_index": pilot_idx, "source_index": src_idx, **records[src_idx]}
            output_file.write(json.dumps(record, ensure_ascii=True) + "\n")

    # write manifest
    manifest = {
        "schema_version": 1,
        "source_path": str(source_path),
        "source_sha256": actual_hash,
        "source_examples": len(records),
        "pilot_path": str(save_pilot_path),
        "pilot_sha256": hashing_func(save_pilot_path),
        "pilot_examples": len(indices),
        "pilot_seed": seed,
        "source_indices": indices,
    }
    save_manifest_path = output_dir / "pilot_manifest.json"
    write_json(save_manifest_path, manifest)

    return save_pilot_path, save_manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    pilot_path, manifest_path = prepare_pilot(args.config, args.output_dir)

    print(f"Pilot: {pilot_path}, Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
