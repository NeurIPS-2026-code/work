#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from validation.common import (
    bootstrap_source,
    conventional_range_name,
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


def configure_r2mem(config: Dict[str, Any], bank_root: Path):
    from exp.eval import experience_encoder
    from exp.eval import research_agent_exp_main

    research_agent_exp_main.model_size = config["worker"]["model_size"]
    research_agent_exp_main.exp_bank_path = str(bank_root)
    experience_encoder._MODEL_PATH = config["embedding"]["model"]
    experience_encoder._MODEL = None

    configured_top_k = int(config["experience"].get("retrieval_top_k", 3))
    original_retrieve = research_agent_exp_main.retrieve_experience

    def retrieve_with_config(query, name, return_k=None, recall_k=None):
        effective_return_k = configured_top_k if return_k is None else return_k
        effective_recall_k = (
            max(10, effective_return_k)
            if recall_k is None
            else max(recall_k, effective_return_k)
        )
        return original_retrieve(
            query,
            name,
            return_k=effective_return_k,
            recall_k=effective_recall_k,
        )

    research_agent_exp_main.retrieve_experience = retrieve_with_config
    return research_agent_exp_main


def trace_statistics(result: Dict[str, Any]) -> Tuple[int, int]:
    trace_path = result.get("research_trace_file")
    if not trace_path or not Path(trace_path).is_file():
        return 0, 0
    trace = read_json(trace_path)
    iterations = len(trace.get("iterations", []))
    tokens = int(
        result.get("tokens", result.get("total_tokens", trace.get("tokens", 0)))
        or 0
    )
    return tokens, iterations


def summarize(
    results: List[Dict[str, Any]],
    start: int,
    end: int,
) -> Dict[str, Any]:
    successful = [item for item in results if "f1" in item]
    f1_scores = [float(item["f1"]) for item in successful]
    token_total = 0
    iteration_total = 0
    for item in successful:
        tokens, iterations = trace_statistics(item)
        token_total += tokens
        iteration_total += iterations
    return {
        "total_samples": end - start,
        "success_count": len(successful),
        "failed_count": (end - start) - len(successful),
        "success_rate": len(successful) / (end - start) if end > start else 0.0,
        "avg_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "f1_scores": f1_scores,
        "research_tokens_total": token_total,
        "research_tokens_avg": (
            token_total / len(successful) if successful else 0.0
        ),
        "search_iterations_total": iteration_total,
        "search_iterations_avg": (
            iteration_total / len(successful) if successful else 0.0
        ),
        "start_idx": start,
        "end_idx": end - 1,
    }


def load_existing_result(
    path: Path,
    protocol_signature: str,
) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    result = read_json(path)
    existing_signature = result.get("validation_protocol_signature")
    if existing_signature != protocol_signature:
        raise RuntimeError(
            f"{path} belongs to a different or untracked protocol. "
            "Use a fresh output_root instead of mixing experimental runs."
        )
    trace_path = result.get("research_trace_file")
    if (
        "f1" not in result
        or result.get("error")
        or not trace_path
        or not Path(trace_path).is_file()
    ):
        return None
    return result


def run_gam(
    config: Dict[str, Any],
    module,
    samples: List[Dict[str, Any]],
    start: int,
    end: int,
) -> Tuple[Path, List[Dict[str, Any]]]:
    stage_root = output_root(config) / "03_hotpotqa_gam"
    results: List[Dict[str, Any]] = []
    worker_args = worker_call_args(config)
    source = Path(config["paths"]["source_root"])
    protocol_signature = stable_hash({
        "method": "GAM",
        "target_data_sha256": sha256_file(config["paths"]["hotpotqa_data"]),
        "worker_model": config["worker"]["model"],
        "worker_backend": config["worker"]["backend"],
        "worker_base_url": config["worker"]["base_url"],
        "embedding_model": config["embedding"]["model"],
        "embedding_device": embedding_device(config),
        "max_chunk_tokens": config["hotpotqa"]["max_chunk_tokens"],
        "runner_sha256": sha256_file(source / "eval" / "hotpotqa_test.py"),
    })
    for sample_index in range(start, end):
        sample = samples[sample_index]
        sample_id = sample["_id"]
        result_path = stage_root / sample_id / "qa_result.json"
        existing = load_existing_result(result_path, protocol_signature)
        if existing is not None:
            validate_retriever_artifacts(result_path.parent)
            print(f"[GAM {sample_index + 1}/{end}] Reusing {sample_id}")
            results.append(existing)
            continue
        print(f"[GAM {sample_index + 1}/{end}] Running {sample_id}")
        result = module.process_sample(
            sample=sample,
            sample_index=sample_index,
            outdir=str(stage_root),
            max_tokens=int(config["hotpotqa"]["max_chunk_tokens"]),
            embedding_model_path=config["embedding"]["model"],
            use_schema=False,
            **worker_args,
        )
        result["validation_protocol_signature"] = protocol_signature
        write_json(result_path, result)
        validate_retriever_artifacts(result_path.parent)
        results.append(result)

    batch_name = conventional_range_name(start, end, "batch_results")
    stats_name = conventional_range_name(start, end, "batch_statistics")
    write_json(stage_root / batch_name, results)
    write_json(stage_root / stats_name, summarize(results, start, end))
    return stage_root, results


def run_r2mem(
    config: Dict[str, Any],
    module,
    agent_module,
    samples: List[Dict[str, Any]],
    start: int,
    end: int,
) -> Tuple[Path, List[Dict[str, Any]]]:
    stage_root = output_root(config) / "04_hotpotqa_r2mem"
    results: List[Dict[str, Any]] = []
    worker_args = worker_call_args(config)
    original_research = agent_module.ResearchAgent_exp.research
    bank_manifest_path = output_root(config) / "02_experience_bank" / "bank_manifest.json"
    source = Path(config["paths"]["source_root"])
    protocol_signature = stable_hash({
        "method": "R2-Mem",
        "target_data_sha256": sha256_file(config["paths"]["hotpotqa_data"]),
        "worker_model": config["worker"]["model"],
        "worker_backend": config["worker"]["backend"],
        "worker_base_url": config["worker"]["base_url"],
        "embedding_model": config["embedding"]["model"],
        "embedding_device": embedding_device(config),
        "max_chunk_tokens": config["hotpotqa"]["max_chunk_tokens"],
        "retrieval_top_k": config["experience"].get("retrieval_top_k", 3),
        "bank_manifest_sha256": sha256_file(bank_manifest_path),
        "runner_sha256": sha256_file(
            source / "exp" / "eval" / "datasets_test" / "hotpotqa_exp.py"
        ),
        "agent_sha256": sha256_file(
            source / "exp" / "eval" / "research_agent_exp_main.py"
        ),
    })

    for sample_index in range(start, end):
        sample = samples[sample_index]
        sample_id = sample["_id"]
        result_path = stage_root / sample_id / "qa_result.json"
        existing = load_existing_result(result_path, protocol_signature)
        if existing is not None and "experience_log" in existing:
            validate_retriever_artifacts(result_path.parent)
            print(f"[R2-Mem {sample_index + 1}/{end}] Reusing {sample_id}")
            results.append(existing)
            continue

        captured_logs: List[Any] = []

        def recording_research(agent_self, *call_args, **call_kwargs):
            output, experience_log = original_research(
                agent_self,
                *call_args,
                **call_kwargs,
            )
            captured_logs.append(experience_log)
            return output, experience_log

        agent_module.ResearchAgent_exp.research = recording_research
        print(f"[R2-Mem {sample_index + 1}/{end}] Running {sample_id}")
        try:
            result = module.process_sample(
                sample=sample,
                sample_index=sample_index,
                outdir=str(stage_root),
                max_tokens=int(config["hotpotqa"]["max_chunk_tokens"]),
                embedding_model_path=config["embedding"]["model"],
                use_schema=False,
                **worker_args,
            )
        finally:
            agent_module.ResearchAgent_exp.research = original_research

        result["experience_log"] = captured_logs[-1] if captured_logs else []
        result["validation_protocol_signature"] = protocol_signature
        write_json(result_path, result)
        validate_retriever_artifacts(result_path.parent)
        results.append(result)

    batch_name = conventional_range_name(start, end, "batch_results")
    stats_name = conventional_range_name(start, end, "batch_statistics")
    write_json(stage_root / batch_name, results)
    write_json(stage_root / stats_name, summarize(results, start, end))
    return stage_root, results


def validate_results(
    name: str,
    results: List[Dict[str, Any]],
    expected: int,
    strict: bool,
) -> None:
    successful = [item for item in results if "f1" in item and not item.get("error")]
    if name == "R²-Mem":
        missing_logs = [
            result_id
            for result_id, item in (
                (str(result.get("_id") or result.get("sample_id")), result)
                for result in successful
            )
            if not any(
                step.get("Planning") or step.get("Reflection")
                for step in item.get("experience_log", [])
            )
        ]
        if missing_logs and strict:
            raise RuntimeError(
                "R²-Mem results are missing retrieved-experience logs for: "
                + ", ".join(missing_logs[:10])
            )
    if len(successful) != expected and strict:
        raise RuntimeError(
            f"{name} completed {len(successful)}/{expected} HotpotQA questions. "
            "Inspect per-question qa_result.json files and rerun to retry failures."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("gam", "r2mem", "both"), default="both")
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    source = bootstrap_source(config)
    baseline_module = load_source_module(
        source / "eval" / "hotpotqa_test.py",
        "_r2mem_original_hotpotqa_gam",
    )
    dense_device = embedding_device(config)
    patch_dense_config(
        baseline_module,
        config["embedding"]["model"],
        dense_device,
    )
    samples = baseline_module.load_hotpotqa(config["paths"]["hotpotqa_data"])

    start = (
        int(config["hotpotqa"]["start_idx"])
        if args.start_idx is None
        else args.start_idx
    )
    end = (
        int(config["hotpotqa"]["end_idx"])
        if args.end_idx is None
        else args.end_idx
    )
    if not 0 <= start < end <= len(samples):
        raise ValueError(
            f"Invalid HotpotQA range [{start}, {end}) for {len(samples)} examples"
        )

    strict = bool(config["experience"].get("strict", True))
    manifests = {}
    if args.mode in {"gam", "both"}:
        gam_root, gam_results = run_gam(
            config,
            baseline_module,
            samples,
            start,
            end,
        )
        validate_results("GAM", gam_results, end - start, strict)
        manifests["gam"] = {
            "output": str(gam_root),
            "successful_questions": sum("f1" in item for item in gam_results),
        }

    if args.mode in {"r2mem", "both"}:
        bank_root = output_root(config) / "02_experience_bank"
        bank_manifest = bank_root / "bank_manifest.json"
        if not bank_manifest.is_file():
            raise FileNotFoundError(
                f"Experience bank is missing: {bank_manifest}. Run stage 02 first."
            )
        agent_module = configure_r2mem(config, bank_root)
        r2_module = load_source_module(
            source / "exp" / "eval" / "datasets_test" / "hotpotqa_exp.py",
            "_r2mem_original_hotpotqa_exp",
        )
        r2_module.bge_path = config["embedding"]["model"]
        patch_dense_config(
            r2_module,
            config["embedding"]["model"],
            dense_device,
        )
        r2_samples = r2_module.load_hotpotqa(config["paths"]["hotpotqa_data"])
        if [item["_id"] for item in r2_samples] != [
            item["_id"] for item in samples
        ]:
            raise RuntimeError("GAM and R²-Mem HotpotQA loaders disagree on IDs")

        r2_root, r2_results = run_r2mem(
            config,
            r2_module,
            agent_module,
            r2_samples,
            start,
            end,
        )
        validate_results("R²-Mem", r2_results, end - start, strict)
        manifests["r2mem"] = {
            "output": str(r2_root),
            "successful_questions": sum("f1" in item for item in r2_results),
            "bank_manifest": str(bank_manifest),
            "bank_manifest_sha256": sha256_file(bank_manifest),
        }

    write_json(output_root(config) / "hotpotqa_stage_manifest.json", {
        "target_dataset": "HotpotQA",
        "target_data": config["paths"]["hotpotqa_data"],
        "target_data_sha256": sha256_file(config["paths"]["hotpotqa_data"]),
        "range": {"start": start, "end_exclusive": end},
        "experience_source_dataset": "LoCoMo",
        "hotpotqa_used_for_bank_construction": False,
        "worker_model": config["worker"]["model"],
        "mode": args.mode,
        "outputs": manifests,
    })
    print("HotpotQA evaluation stage complete")


if __name__ == "__main__":
    main()
