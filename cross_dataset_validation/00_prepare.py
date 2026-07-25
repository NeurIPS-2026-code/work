#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

from validation.common import (
    bootstrap_source,
    environment_manifest,
    load_config,
    output_root,
    sha256_file,
    write_json,
)


REQUIRED_SOURCE_FILES = [
    "eval/locomo_test.py",
    "eval/hotpotqa_test.py",
    "exp/judge_model.py",
    "exp/prompts/self_reflection_prompts.py",
    "exp/eval/datasets_test/hotpotqa_exp.py",
    "exp/eval/research_agent_exp_main.py",
    "exp/eval/experience_encoder.py",
]


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as response:
        with temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    temporary.replace(destination)


def validate_locomo(
    path: Path,
    expected_sample_id: str,
    expected_sha256: str,
) -> dict:
    actual_sha256 = sha256_file(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            "LoCoMo SHA-256 mismatch: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    samples = data.get("samples", []) if isinstance(data, dict) else data
    if not isinstance(samples, list):
        raise ValueError("LoCoMo data must be a list or {'samples': [...]}")

    matches = [
        (index, sample)
        for index, sample in enumerate(samples)
        if sample.get("sample_id") == expected_sample_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {expected_sample_id!r}, found {len(matches)}"
        )
    index, sample = matches[0]
    categories = {}
    for item in sample.get("qa", []):
        key = str(item.get("category"))
        categories[key] = categories.get(key, 0) + 1
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "sample_id": expected_sample_id,
        "array_index": index,
        "question_count": len(sample.get("qa", [])),
        "category_counts": categories,
        "evaluated_question_count": sum(
            value for key, value in categories.items() if key != "5"
        ),
    }


def validate_hotpotqa(
    path: Path,
    expected_examples: int,
    expected_sha256: str,
) -> dict:
    actual_sha256 = sha256_file(path)
    if expected_sha256 and actual_sha256 != expected_sha256:
        raise ValueError(
            "HotpotQA SHA-256 mismatch: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("HotpotQA data must be a JSON list")
    if expected_examples and len(data) != expected_examples:
        raise ValueError(
            f"Expected {expected_examples} HotpotQA examples, found {len(data)}"
        )
    required = {"context", "input", "answers"}
    malformed = [
        index for index, item in enumerate(data) if not required.issubset(item)
    ]
    if malformed:
        raise ValueError(f"Malformed HotpotQA examples at indices {malformed[:10]}")
    return {
        "path": str(path),
        "sha256": actual_sha256,
        "example_count": len(data),
        "first_index": data[0].get("index") if data else None,
        "last_index": data[-1].get("index") if data else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Fail instead of downloading HotpotQA when it is absent.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    source = bootstrap_source(config)
    missing_source = [
        str(source / relative)
        for relative in REQUIRED_SOURCE_FILES
        if not (source / relative).is_file()
    ]
    if missing_source:
        raise FileNotFoundError(
            "Required original source files are missing:\n"
            + "\n".join(missing_source)
        )

    locomo_path = Path(config["paths"]["locomo_data"])
    if not locomo_path.is_file():
        raise FileNotFoundError(f"LoCoMo file is missing: {locomo_path}")

    hotpotqa_path = Path(config["paths"]["hotpotqa_data"])
    if not hotpotqa_path.is_file():
        if args.no_download or not bool(
            config["data"].get("download_if_missing", True)
        ):
            raise FileNotFoundError(f"HotpotQA file is missing: {hotpotqa_path}")
        download(config["data"]["hotpotqa_url"], hotpotqa_path)

    manifest = {
        "protocol": {
            "source_dataset": "LoCoMo",
            "source_sample_id": config["data"]["locomo_source_sample_id"],
            "target_dataset": "HotpotQA",
            "hotpotqa_used_for_bank_construction": False,
        },
        "locomo": validate_locomo(
            locomo_path,
            config["data"]["locomo_source_sample_id"],
            config["data"].get("locomo_sha256", ""),
        ),
        "hotpotqa": validate_hotpotqa(
            hotpotqa_path,
            int(config["data"]["hotpotqa_expected_examples"]),
            config["data"].get("hotpotqa_sha256", ""),
        ),
        "source_files": {
            relative: {
                "path": str(source / relative),
                "sha256": sha256_file(source / relative),
            }
            for relative in REQUIRED_SOURCE_FILES
        },
        "environment": environment_manifest(config),
    }
    destination = output_root(config) / "00_manifest" / "data_manifest.json"
    write_json(destination, manifest)
    print(f"Preflight succeeded. Manifest: {destination}")


if __name__ == "__main__":
    main()
