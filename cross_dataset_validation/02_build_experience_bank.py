#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from validation.common import (
    bootstrap_source,
    embedding_device,
    load_config,
    make_generator,
    output_root,
    read_json,
    response_json,
    response_tokens,
    sha256_file,
    sha256_paths,
    stable_hash,
    write_json,
)


EXPERIENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "thinking": {"type": "string"},
        "summary": {"type": "string"},
        "situation": {"type": "string"},
        "experience": {"type": "string"},
    },
    "required": ["thinking", "summary", "situation", "experience"],
    "additionalProperties": False,
}


def trace_to_string(trace: Dict[str, Any]) -> Tuple[str, str]:
    text = ""
    for iteration in trace["iterations"]:
        text += f"(step {iteration['step']})\n"
        text += (
            "[Planning]\n"
            f"plan: {iteration['plan']}\n"
            "[Reflection]\n"
            f"'temp_memory': {iteration['temp_memory']}\n"
            f"'decision': {iteration['decision']}\n\n"
        )
    text += "[Answering]\n" + f"integrated_memory:{trace['integrated_memory']}"
    return trace["question"], text


def step_context(
    trace: Dict[str, Any],
    step: int,
    module: str,
) -> Tuple[str, str, str]:
    iterations = {
        int(iteration["step"]): iteration for iteration in trace["iterations"]
    }
    if step not in iterations:
        raise ValueError(f"Evaluator referenced absent trace step {step}")
    iteration = iterations[step]

    if module == "Planning":
        if step == 0:
            current_question = trace["question"]
        else:
            previous = iterations.get(step - 1)
            if previous is None:
                raise ValueError(f"Planning step {step} has no previous step")
            current_question = previous["decision"].get("new_request") or trace["question"]
        info = (
            f'"question": {current_question}\n'
            f'"plan": {iteration["plan"]}'
        )
        return current_question, info, current_question

    if module == "Reflection":
        current_question = trace["question"]
        temp_memory = iteration["temp_memory"].get("content", "")
        info = (
            f'"temp_memory": {temp_memory}\n'
            f'"decision": {iteration["decision"]}'
        )
        condition = f"question:{current_question} temp_memory:{temp_memory}"
        return current_question, info, condition

    raise ValueError(f"Unknown evaluator module: {module!r}")


def validate_experience(value: Dict[str, Any]) -> None:
    missing = [
        key
        for key in ("thinking", "summary", "situation", "experience")
        if not isinstance(value.get(key), str) or not value[key].strip()
    ]
    if missing:
        raise ValueError(f"Learner output has invalid fields: {missing}")


def learn_planning(
    learner,
    prompts,
    question: str,
    evaluation: Dict[str, Any],
    info: str,
    high_quality: bool,
) -> Tuple[Dict[str, Any], int]:
    if high_quality:
        role_desc = (
            "You are an AI [TRACE] Strategist: Distill best-practice planning "
            "patterns and reusable experience from high-quality planning traces."
        )
        content_type = "high_quality"
        thinking = (
            "Why is this [TRACE] considered high quality under the "
            "[DIAGNOSED REASON]. Where did it perform best?"
        )
        planning_task = (
            "summarize an effective and discriminative planning experience "
            "that leads to high-quality planning behaviors."
        )
    else:
        # Kept verbatim from the submitted implementation, including wording.
        role_desc = (
            "You are an AI [TRACE] Auditor: Extract abstract situation and "
            "corrective planning experience from low-quality planning traces."
        )
        content_type = "low_quality"
        thinking = (
            "Why is this [TRACE] considered low quality under the "
            "[DIAGNOSED REASON]. Where did it perform worst?"
        )
        planning_task = (
            "summarize a corrective and discriminative planning experience "
            "to prevent similar low-quality planning behaviors."
        )

    system_prompt = prompts.Planning_system_prompt.format(role_desc=role_desc)
    user_prompt = prompts.Planning_prompt_extra.format(
        QUESTION=question,
        content_type=content_type,
        info=info,
        content_reason=evaluation["reason and advice"],
        thinking=thinking,
        planning_task=planning_task,
    )
    response = learner.generate_single(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        schema=EXPERIENCE_SCHEMA,
    )
    result = response_json(response)
    validate_experience(result)
    return result, response_tokens(response)


def learn_reflection(
    learner,
    prompts,
    question: str,
    evaluation: Dict[str, Any],
    info: str,
    high_quality: bool,
) -> Tuple[Dict[str, Any], int]:
    if high_quality:
        role_desc = (
            "You are an AI Memory [TRACE] Strategist: Distill best-practice "
            "patterns and reusable experience from high-quality reasoning traces."
        )
        content_type = "high_quality"
        thinking = (
            "Why is this [TRACE] considered high quality under the evaluation? "
            "Where did it perform best?"
        )
        experience_instruction = (
            "summarize an abstract and reusable experience to lead high-quality "
            "reflection behaviors in future scenarios."
        )
    else:
        role_desc = (
            "You are an AI Memory [TRACE] Auditor: Extract abstract abstract "
            "situation and corrective from experience low-quality reasoning traces."
        )
        content_type = "low_quality"
        thinking = (
            "Why is this [TRACE] considered low quality under the evaluation? "
            "Where did it perform worst?"
        )
        experience_instruction = (
            "summarize an abstract and reusable experience to prevent similar "
            "low-quality reflection behaviors in future scenarios."
        )

    system_prompt = prompts.Reflection_system_prompt.format(role_desc=role_desc)
    user_prompt = prompts.Reflection_prompt.format(
        QUESTION=question,
        content_type=content_type,
        info=info,
        content_reason=evaluation["reason and advice"],
        thinking=thinking,
        experience=experience_instruction,
    )
    response = learner.generate_single(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        schema=EXPERIENCE_SCHEMA,
    )
    result = response_json(response)
    validate_experience(result)
    return result, response_tokens(response)


def build_index(
    experiences: List[Dict[str, Any]],
    module_name: str,
    model_size: str,
    bank_root: Path,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    if not experiences:
        raise RuntimeError(f"No {module_name} experiences were generated")

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(
        config["embedding"]["model"],
        device=embedding_device(config),
    )
    encoder.max_seq_length = int(
        config["embedding"].get("max_seq_length", 512)
    )
    keys = [
        f"({item['situation']})" + item["condition"]
        for item in experiences
    ]
    embeddings = encoder.encode(
        keys,
        normalize_embeddings=True,
        batch_size=int(config["embedding"].get("batch_size", 32)),
        show_progress_bar=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path = bank_root / f"{module_name}_{model_size}_situation.index"
    faiss.write_index(index, str(index_path))
    metadata = [
        {
            "condition": item["condition"],
            "situation": item["situation"],
            "summary": item["summary"],
            "experience": item["experience"],
        }
        for item in experiences
    ]
    metadata_path = bank_root / f"{module_name}_{model_size}_metadata.json"
    write_json(metadata_path, metadata)
    return {
        "entries": len(experiences),
        "embedding_dimension": int(embeddings.shape[1]),
        "index": str(index_path),
        "index_sha256": sha256_file(index_path),
        "metadata": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "unique_situations": len({item["situation"] for item in experiences}),
        "unique_conditions": len({item["condition"] for item in experiences}),
    }


def process_question(
    *,
    item: Dict[str, Any],
    trace_path: Path,
    evaluator,
    learner,
    judge_module,
    prompts,
    high_threshold: int,
    low_threshold: int,
) -> Dict[str, Any]:
    trace = read_json(trace_path)
    question, trace_text = trace_to_string(trace)
    if item.get("question") and item["question"] != question:
        raise ValueError(
            f"Question mismatch between qa_results and {trace_path.name}"
        )

    evaluation, evaluator_tokens = judge_module.judger_vllm(
        question,
        item["gold_answer"],
        item["summary_answer"],
        trace_text,
        evaluator,
    )
    if evaluation.get("error"):
        raise RuntimeError(f"Evaluator failed: {evaluation['error']}")
    evaluations = evaluation.get("results")
    if not isinstance(evaluations, list) or not evaluations:
        raise ValueError("Evaluator returned no module evaluations")

    experiences: List[Dict[str, Any]] = []
    score_rows = []
    learner_tokens = 0
    for module_evaluation in evaluations:
        module_name = module_evaluation.get("module")
        rubric_values = module_evaluation.get("rubrics", {}).values()
        scores = [int(value) for value in rubric_values]
        if len(scores) != 4 or any(value < 0 or value > 3 for value in scores):
            raise ValueError(f"Invalid rubric scores: {module_evaluation}")
        total_score = sum(scores)
        score_rows.append({
            "step": int(module_evaluation["step"]),
            "module": module_name,
            "score": total_score,
        })
        if low_threshold < total_score < high_threshold:
            continue

        step = int(module_evaluation["step"])
        current_question, info, condition = step_context(
            trace,
            step,
            module_name,
        )
        high_quality = total_score >= high_threshold
        if module_name == "Planning":
            learned, used_tokens = learn_planning(
                learner,
                prompts,
                current_question,
                module_evaluation,
                info,
                high_quality,
            )
        elif module_name == "Reflection":
            learned, used_tokens = learn_reflection(
                learner,
                prompts,
                current_question,
                module_evaluation,
                info,
                high_quality,
            )
        else:
            raise ValueError(f"Unknown module {module_name!r}")
        learned["condition"] = condition
        learned["module"] = module_name
        experiences.append(learned)
        learner_tokens += used_tokens

    return {
        "status": "ok",
        "question": question,
        "trace_file": str(trace_path),
        "trace_sha256": sha256_file(trace_path),
        "evaluator_output": evaluation,
        "scores": score_rows,
        "experiences": experiences,
        "evaluator_tokens": evaluator_tokens,
        "learner_tokens": learner_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--question-limit",
        type=int,
        default=None,
        help="Smoke test only: reflect on the first N LoCoMo results.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    bootstrap_source(config)
    from exp import judge_model
    from exp.prompts import self_reflection_prompts

    source_root = output_root(config) / "01_locomo_gam"
    conv_root = source_root / config["data"]["locomo_source_sample_id"]
    results_path = conv_root / "qa_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(
            f"LoCoMo GAM results are missing: {results_path}. Run stage 01 first."
        )
    items = read_json(results_path)
    if args.question_limit is not None:
        if args.question_limit <= 0:
            raise ValueError("--question-limit must be positive")
        items = items[: args.question_limit]

    bank_root = output_root(config) / "02_experience_bank"
    checkpoints = bank_root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)

    evaluator = make_generator(
        config,
        "evaluator",
        temperature=0.2,
        max_tokens=4096,
        use_schema=True,
    )
    learner = make_generator(
        config,
        "worker",
        temperature=0.5,
        max_tokens=1024,
        use_schema=True,
    )
    high_threshold = int(config["experience"]["high_threshold"])
    low_threshold = int(config["experience"]["low_threshold"])
    protocol = {
        "source_results_sha256": sha256_file(results_path),
        "worker_model": config["worker"]["model"],
        "worker_backend": config["worker"]["backend"],
        "evaluator_model": config["evaluator"]["model"],
        "evaluator_backend": config["evaluator"]["backend"],
        "evaluator_base_url": config["evaluator"]["base_url"],
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "question_limit": args.question_limit,
        "planning_prompt_sha256": sha256_file(
            bootstrap_source(config)
            / "exp"
            / "prompts"
            / "self_reflection_prompts.py"
        ),
        "evaluator_prompt_sha256": sha256_file(
            bootstrap_source(config) / "exp" / "judge_model.py"
        ),
    }
    protocol_signature = stable_hash(protocol)

    records = []
    errors = []
    for index, item in enumerate(items, start=1):
        trace_path = conv_root / f"research_trace_q{index}.json"
        if not trace_path.is_file():
            errors.append({
                "question_index": index,
                "error": f"Missing trace: {trace_path}",
            })
            continue
        checkpoint_path = checkpoints / f"q{index:04d}.json"
        trace_hash = sha256_file(trace_path)
        if checkpoint_path.is_file():
            existing = read_json(checkpoint_path)
            if (
                existing.get("status") == "ok"
                and existing.get("trace_sha256") == trace_hash
                and existing.get("protocol_signature") == protocol_signature
            ):
                print(f"[{index}/{len(items)}] Reusing {checkpoint_path.name}")
                records.append(existing)
                continue

        print(f"[{index}/{len(items)}] Evaluating and reflecting")
        try:
            record = process_question(
                item=item,
                trace_path=trace_path,
                evaluator=evaluator,
                learner=learner,
                judge_module=judge_model,
                prompts=self_reflection_prompts,
                high_threshold=high_threshold,
                low_threshold=low_threshold,
            )
            record["question_index"] = index
            record["protocol_signature"] = protocol_signature
            write_json(checkpoint_path, record)
            records.append(record)
        except Exception as error:
            record = {
                "status": "error",
                "question_index": index,
                "trace_file": str(trace_path),
                "trace_sha256": trace_hash,
                "protocol_signature": protocol_signature,
                "error": str(error),
            }
            write_json(checkpoint_path, record)
            errors.append(record)

    if errors and bool(config["experience"].get("strict", True)):
        write_json(bank_root / "build_errors.json", errors)
        raise RuntimeError(
            f"Experience construction failed for {len(errors)} questions. "
            f"See {bank_root / 'build_errors.json'} and rerun to retry."
        )

    planning: List[Dict[str, Any]] = []
    reflection: List[Dict[str, Any]] = []
    score_histograms = {
        "Planning": Counter(),
        "Reflection": Counter(),
    }
    for record in records:
        for score in record["scores"]:
            score_histograms[score["module"]][str(score["score"])] += 1
        for experience in record["experiences"]:
            if experience["module"] == "Planning":
                planning.append(experience)
            elif experience["module"] == "Reflection":
                reflection.append(experience)

    model_size = config["worker"]["model_size"]
    planning_raw = bank_root / f"Planning_exp_{model_size}_X.json"
    reflection_raw = bank_root / f"Reflection_exp_{model_size}_X.json"
    write_json(planning_raw, planning, sort_keys=True)
    write_json(reflection_raw, reflection, sort_keys=True)

    planning_manifest = build_index(
        planning,
        "Planning",
        model_size,
        bank_root,
        config,
    )
    reflection_manifest = build_index(
        reflection,
        "Reflection",
        model_size,
        bank_root,
        config,
    )
    trace_paths = [
        conv_root / f"research_trace_q{index}.json"
        for index in range(1, len(items) + 1)
    ]
    manifest = {
        "source_dataset": "LoCoMo",
        "source_sample_id": config["data"]["locomo_source_sample_id"],
        "target_data_used": False,
        "source_results": str(results_path),
        "source_results_sha256": sha256_file(results_path),
        "source_traces_sha256": sha256_paths(trace_paths),
        "source_question_count": len(items),
        "smoke_test_question_limit": args.question_limit,
        "protocol": protocol,
        "protocol_signature": protocol_signature,
        "worker_model": config["worker"]["model"],
        "evaluator_model": config["evaluator"]["model"],
        "evaluator_backend": config["evaluator"]["backend"],
        "embedding_model": config["embedding"]["model"],
        "composite_embedding_key": "(situation) + condition",
        "high_threshold": high_threshold,
        "low_threshold": low_threshold,
        "planning": planning_manifest,
        "reflection": reflection_manifest,
        "score_histograms": {
            key: dict(sorted(value.items(), key=lambda item: int(item[0])))
            for key, value in score_histograms.items()
        },
        "evaluator_tokens": sum(
            int(record.get("evaluator_tokens", 0)) for record in records
        ),
        "learner_tokens": sum(
            int(record.get("learner_tokens", 0)) for record in records
        ),
        "errors": errors,
    }
    write_json(bank_root / "bank_manifest.json", manifest)
    print(
        "Experience bank complete: "
        f"{len(planning)} Planning, {len(reflection)} Reflection entries"
    )


if __name__ == "__main__":
    main()
