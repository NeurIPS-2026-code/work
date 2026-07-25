#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from validation.common import load_config


ROOT = Path(__file__).resolve().parent
STAGES = {
    "prepare": "00_prepare.py",
    "locomo": "01_run_locomo_gam.py",
    "bank": "02_build_experience_bank.py",
    "hotpotqa": "03_run_hotpotqa.py",
    "report": "04_summarize.py",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stages",
        default="prepare,locomo,bank,hotpotqa,report",
        help=f"Comma-separated subset of: {', '.join(STAGES)}",
    )
    parser.add_argument(
        "--smoke-test",
        type=int,
        default=None,
        help="Run only N source/target questions. Never report smoke-test results.",
    )
    args = parser.parse_args()

    selected = [name.strip() for name in args.stages.split(",") if name.strip()]
    unknown = [name for name in selected if name not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stages: {', '.join(unknown)}")
    if args.smoke_test is not None and args.smoke_test <= 0:
        raise ValueError("--smoke-test must be positive")

    effective_config = Path(args.config).resolve()
    if args.smoke_test is not None:
        smoke_config = load_config(effective_config)
        smoke_config.pop("_config_path", None)
        original_output = Path(smoke_config["paths"]["output_root"])
        smoke_config["experiment_name"] = (
            f"{smoke_config.get('experiment_name', 'validation')}"
            f"_smoke_{args.smoke_test}"
        )
        smoke_config["paths"]["output_root"] = str(
            Path(f"{original_output}_smoke_{args.smoke_test}")
        )
        generated = ROOT / f".smoke_config_{args.smoke_test}.json"
        with generated.open("w", encoding="utf-8") as handle:
            json.dump(smoke_config, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        effective_config = generated
        print(
            f"Smoke test uses isolated output config: {effective_config}",
            flush=True,
        )

    for name in selected:
        command = [
            sys.executable,
            str(ROOT / STAGES[name]),
            "--config",
            str(effective_config),
        ]
        if args.smoke_test is not None:
            if name in {"locomo", "bank"}:
                command.extend(["--question-limit", str(args.smoke_test)])
            elif name in {"hotpotqa", "report"}:
                command.extend(["--start-idx", "0", "--end-idx", str(args.smoke_test)])
        print(f"\n=== Stage: {name} ===", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    print("\nRequested validation stages completed.")


if __name__ == "__main__":
    main()
