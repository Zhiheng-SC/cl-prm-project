import argparse
import hashlib
import json
from pathlib import Path
from huggingface_hub import snapshot_download

REPOSITORY_ID = "FreedomIntelligence/OVM-Mistral-7b"
GENERATOR = "mistral7b-ep2"
VERIFIER = "mistral7b-ep2-n100-scahead-mse-lm-token"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=REPOSITORY_ID,
        local_dir=str(output),
        local_dir_use_symlinks=False,
        allow_patterns=[f"{GENERATOR}/**", f"{VERIFIER}/**"],
        resume_download=True,
    )
    # verify download is successful
    for subdirectory in (GENERATOR, VERIFIER):
        path = Path(snapshot_path) / subdirectory
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Incomplete model download: {path}")

    # write manifest
    manifest_path = output / "ovm_snapshot_manifest.json"
    manifest = {
        "schema_version": 1,
        "subdirectories": {
            subdirectory: {
                "config_sha256": hashlib.sha256(
                    (Path(snapshot_path) / subdirectory / "config.json").read_bytes()
                ).hexdigest()
            }
            for subdirectory in (GENERATOR, VERIFIER)
        },
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Snapshot: {snapshot_path}, Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
