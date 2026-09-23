"""Plan or execute exploratory second-stage inference on frozen splits."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
from experiments.formal.prm_router.run_formal_inference import validate_split  # noqa: E402

CONFIG = REPO_ROOT / "configs/experiments/prm_router_extensions.json"
MANIFEST = REPO_ROOT / "configs/experiments/prm_router_formal_split_manifest.json"
RUNNER = REPO_ROOT / "experiments/extensions/prm_router/run_alternative.py"
OUTPUT = REPO_ROOT / "outputs/prm_router/extensions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--models", nargs="+", choices=("genprm_official", "math_prm", "skywork_prm"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--limit", type=int, help="Pilot sample count; uses separate output namespace.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--genprm-src", type=Path)
    parser.add_argument("--skywork-src", type=Path)
    parser.add_argument("--allow-generated-code", action="store_true")
    parser.add_argument("--confirm-exploratory-test", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    input_path = validate_split(args.split, manifest)
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive.")
    if args.split == "test" and args.execute and not args.confirm_exploratory_test:
        raise ValueError("Test was viewed for the main study. Freeze extension choices and explicitly confirm exploratory test reporting.")
    models = list(dict.fromkeys(args.models))
    for name in models:
        if name not in config["models"]:
            raise ValueError(f"No extension configuration for {name}.")
        directory = args.output_dir / ("pilot" if args.limit else "full") / args.split
        command = [sys.executable, str(RUNNER), "--config", str(args.config),
                   "--verifier", name, "--input", str(input_path),
                   "--output", str(directory / f"{name}.jsonl"), "--resume"]
        if args.limit:
            command += ["--limit", str(args.limit)]
        if name == "genprm_official":
            if args.allow_generated_code:
                command.append("--allow-generated-code")
            if args.genprm_src:
                command += ["--genprm-src", str(args.genprm_src)]
        if name == "skywork_prm" and args.skywork_src:
            command += ["--skywork-src", str(args.skywork_src)]
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
