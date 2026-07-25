#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from validation.common import (
    bootstrap_source,
    embedding_device,
    load_config,
    load_source_module,
    output_root,
    read_json,
    sha256_file,
    stable_hash,
    validate_retriever_artifacts,
    worker_call_args,
    write_json,
)


def patch_dense_config(
    module,
    embedding_model: str,
    device: str,
) -> None:
    original = module.DenseRetrieverConfig

    def configured_dense_retriever(**kwargs):
        kwargs["model_name"] = embedding_model
        kwargs["devices"] = [device]
        return original(**kwargs)

    module.DenseRetrieverConfig = configured_dense_retriever


def summarize(module, results: list[Dict[str, Any]]) -> Dict[str, Any]:
    by_category, details = module.compute_metrics_by_category(results)
    f1_values = [row["F1"] for row in details]
    bleu_values = [row["BLEU1"] for row in details]
    return {
        "total_questions": len(results),
        "overall_f1_avg": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "overall_bleu1_avg": (
            sum(bleu_values) / len(bleu_values) if bleu_values else 0.0
        ),
        "by_category": by_category,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--question-limit",
        type=int,
        default=None,
        help="Smoke test only: keep the first N non-category-5 questions.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    source = bootstrap_source(config)
    module = load_source_module(
        source / "eval" / "locomo_test.py",
        "_r2mem_original_locomo_test",
    )
    patch_dense_config(
        module,
        config["embedding"]["model"],
        embedding_device(config),
    )

    samples = module.load_locomo(config["paths"]["locomo_data"])
    sample_id = config["data"]["locomo_source_sample_id"]
    matches = [
        (index, sample)
        for index, sample in enumerate(samples)
        if sample.get("sample_id") == sample_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one source sample {sample_id!r}")
    sample_index, sample = matches[0]

    if args.question_limit is not None:
        if args.question_limit <= 0:
            raise ValueError("--question-limit must be positive")
        kept = []
        for question in sample.get("qa", []):
            if question.get("category") == 5:
                continue
            kept.append(question)
            if len(kept) >= args.question_limit:
                break
        sample = dict(sample)
        sample["qa"] = kept

    stage_root = output_root(config) / "01_locomo_gam"
    conv_root = stage_root / sample_id
    results_path = conv_root / "qa_results.json"
    stage_manifest_path = stage_root / "stage_manifest.json"
    protocol = {
        "source_data_sha256": sha256_file(config["paths"]["locomo_data"]),
        "source_sample_id": sample_id,
        "source_array_index": sample_index,
        "question_limit": args.question_limit,
        "worker_model": config["worker"]["model"],
        "worker_backend": config["worker"]["backend"],
        "worker_base_url": config["worker"]["base_url"],
        "embedding_model": config["embedding"]["model"],
        "embedding_device": embedding_device(config),
        "runner_sha256": sha256_file(source / "eval" / "locomo_test.py"),
    }
    protocol_signature = stable_hash(protocol)

    expected_questions = sum(
        1 for question in sample.get("qa", []) if question.get("category") != 5
    )
    if results_path.is_file():
        if not stage_manifest_path.is_file():
            raise RuntimeError(
                f"{results_path} exists without {stage_manifest_path}. "
                "Use a fresh output_root rather than mixing runs."
            )
        existing_manifest = read_json(stage_manifest_path)
        if existing_manifest.get("protocol_signature") != protocol_signature:
            raise RuntimeError(
                "Existing LoCoMo output was generated under a different "
                "protocol. Use a fresh output_root."
            )
        results = read_json(results_path)
        if len(results) == expected_questions:
            print(f"Reusing completed LoCoMo results: {results_path}")
        else:
            raise RuntimeError(
                f"{results_path} has {len(results)} questions; expected "
                f"{expected_questions}. Use a fresh output_root for a new run."
            )
    else:
        results, total_tokens = module.process_sample(
            sample=sample,
            sample_index=sample_index,
            outdir=str(stage_root),
            use_schema=False,
            **worker_call_args(config),
        )
        if len(results) != expected_questions:
            raise RuntimeError(
                f"LoCoMo produced {len(results)} results; expected "
                f"{expected_questions}. Inspect {conv_root}."
            )
        write_json(conv_root / "validation_token_total.json", {
            "research_agent_total_tokens": total_tokens
        })

    trace_paths = sorted(conv_root.glob("research_trace_q*.json"))
    if len(trace_paths) != expected_questions:
        raise RuntimeError(
            f"Found {len(trace_paths)} traces in {conv_root}; expected "
            f"{expected_questions}."
        )
    validate_retriever_artifacts(conv_root)

    summary = summarize(module, results)
    summary.update({
        "source_dataset": "LoCoMo",
        "source_sample_id": sample_id,
        "source_array_index": sample_index,
        "smoke_test_question_limit": args.question_limit,
    })
    write_json(stage_root / "batch_results_conv-26.json", results)
    write_json(stage_root / "batch_statistics_conv-26.json", summary)
    write_json(stage_manifest_path, {
        **protocol,
        "protocol_signature": protocol_signature,
        "source_data": str(config["paths"]["locomo_data"]),
        "question_count": len(results),
        "trace_count": len(trace_paths),
        "results_sha256": sha256_file(results_path),
    })
    print(f"LoCoMo GAM stage complete: {stage_root}")


if __name__ == "__main__":
    main()
