#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from validation.common import (
    conventional_range_name,
    load_config,
    output_root,
    read_json,
    sha256_file,
    write_json,
)


def result_id(item: Dict[str, Any]) -> str:
    return str(item.get("_id") or item.get("sample_id"))


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def quantile(sorted_values: List[float], probability: float) -> float:
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def paired_bootstrap_ci(
    deltas: List[float],
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(seed)
    estimates = []
    count = len(deltas)
    for _ in range(samples):
        estimates.append(
            sum(deltas[rng.randrange(count)] for _ in range(count)) / count
        )
    estimates.sort()
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def trace_cost(item: Dict[str, Any]) -> Tuple[int, int]:
    trace_path = item.get("research_trace_file")
    trace = {}
    if trace_path and Path(trace_path).is_file():
        trace = read_json(trace_path)
    tokens = int(
        item.get("tokens", item.get("total_tokens", trace.get("tokens", 0))) or 0
    )
    return tokens, len(trace.get("iterations", []))


def experience_statistics(
    item: Dict[str, Any],
) -> Tuple[int, int, float, int]:
    events = 0
    retrieved = 0
    similarities: List[float] = []
    unique_ids = set()
    for step in item.get("experience_log", []):
        for module in ("Planning", "Reflection"):
            entries = step.get(module, [])
            events += 1
            retrieved += len(entries)
            for entry in entries:
                similarities.append(float(entry.get("similarity", 0.0)))
                unique_ids.add((module, int(entry.get("id", -1))))
    return events, retrieved, mean(similarities), len(unique_ids)


def load_successful(path: Path) -> Dict[str, Dict[str, Any]]:
    items = read_json(path)
    return {
        result_id(item): item
        for item in items
        if "f1" in item and not item.get("error")
    }


def markdown_report(report: Dict[str, Any]) -> str:
    paired = report["paired_comparison"]
    efficiency = report["efficiency"]
    bank = report["experience_bank"]
    retrieval = report["experience_retrieval"]
    ci = paired["delta_f1_bootstrap_95_ci"]
    return f"""# LoCoMo → HotpotQA cross-dataset validation

## Protocol

- Experience source: **LoCoMo `{report['protocol']['source_sample_id']}` only**
- Target evaluation: **HotpotQA**, {paired['paired_questions']} paired questions
- Worker: `{report['protocol']['worker_model']}`
- Evaluator: `{report['protocol']['evaluator_model']}`
- HotpotQA trajectories used to construct the bank: **No**
- Retrieval key: `(situation) + condition`

## Main paired result

| Method | HotpotQA F1 |
|---|---:|
| GAM | {paired['gam_f1'] * 100:.2f} |
| R²-Mem with LoCoMo bank | {paired['r2mem_f1'] * 100:.2f} |
| Paired Δ | {paired['delta_f1'] * 100:+.2f} |

The paired bootstrap 95% confidence interval for ΔF1 is
**[{ci[0] * 100:+.2f}, {ci[1] * 100:+.2f}]**. R²-Mem improves
{paired['improved_questions']} questions, ties {paired['tied_questions']}, and
worsens {paired['worsened_questions']}.

## Efficiency

| Method | Mean research tokens | Mean search iterations |
|---|---:|---:|
| GAM | {efficiency['gam_research_tokens_avg']:.2f} | {efficiency['gam_iterations_avg']:.3f} |
| R²-Mem | {efficiency['r2mem_research_tokens_avg']:.2f} | {efficiency['r2mem_iterations_avg']:.3f} |

## Experience bank and actual use

- Planning bank: {bank['planning_entries']} entries,
  {bank['planning_unique_situations']} unique situations,
  {bank['planning_unique_conditions']} unique conditions.
- Reflection bank: {bank['reflection_entries']} entries,
  {bank['reflection_unique_situations']} unique situations,
  {bank['reflection_unique_conditions']} unique conditions.
- Retrieval events on HotpotQA: {retrieval['events']}.
- Retrieved candidates: {retrieval['candidates']} total,
  mean similarity {retrieval['mean_similarity']:.4f}.
- Distinct bank entries used: {retrieval['distinct_entries_used']}.

## Leakage audit

The bank manifest declares LoCoMo `{report['protocol']['source_sample_id']}` as
its only trajectory source and records `target_data_used = false`. HotpotQA
question IDs are generated only during the frozen-bank target evaluation. This
is a transfer experiment, not an in-domain construction/evaluation split.

Machine-readable results and per-question paired deltas are stored in
`cross_dataset_report.json`.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    root = output_root(config)
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
    batch_name = conventional_range_name(start, end, "batch_results")
    gam_path = root / "03_hotpotqa_gam" / batch_name
    r2_path = root / "04_hotpotqa_r2mem" / batch_name
    bank_path = root / "02_experience_bank" / "bank_manifest.json"
    for path in (gam_path, r2_path, bank_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required result is missing: {path}")

    gam = load_successful(gam_path)
    r2mem = load_successful(r2_path)
    paired_ids = sorted(set(gam) & set(r2mem))
    expected = end - start
    if len(paired_ids) != expected and config["experience"].get("strict", True):
        raise RuntimeError(
            f"Only {len(paired_ids)}/{expected} paired successful results exist"
        )

    per_question = []
    retrieved_events = 0
    retrieved_candidates = 0
    weighted_similarity = 0.0
    distinct_entries = set()
    for sample_id in paired_ids:
        gam_item = gam[sample_id]
        r2_item = r2mem[sample_id]
        gam_tokens, gam_iterations = trace_cost(gam_item)
        r2_tokens, r2_iterations = trace_cost(r2_item)
        events, candidates, similarity, _ = experience_statistics(r2_item)
        retrieved_events += events
        retrieved_candidates += candidates
        weighted_similarity += similarity * candidates
        for step in r2_item.get("experience_log", []):
            for module in ("Planning", "Reflection"):
                for entry in step.get(module, []):
                    distinct_entries.add((module, int(entry.get("id", -1))))
        per_question.append({
            "sample_id": sample_id,
            "gam_f1": float(gam_item["f1"]),
            "r2mem_f1": float(r2_item["f1"]),
            "delta_f1": float(r2_item["f1"]) - float(gam_item["f1"]),
            "gam_research_tokens": gam_tokens,
            "r2mem_research_tokens": r2_tokens,
            "gam_iterations": gam_iterations,
            "r2mem_iterations": r2_iterations,
            "experience_retrieval_events": events,
            "experience_candidates": candidates,
            "experience_mean_similarity": similarity,
        })

    deltas = [row["delta_f1"] for row in per_question]
    epsilon = 1e-12
    bootstrap_samples = int(config["report"]["bootstrap_samples"])
    bootstrap_seed = int(config["report"]["bootstrap_seed"])
    ci = paired_bootstrap_ci(deltas, bootstrap_samples, bootstrap_seed)
    bank = read_json(bank_path)

    report = {
        "protocol": {
            "source_dataset": "LoCoMo",
            "source_sample_id": bank["source_sample_id"],
            "target_dataset": "HotpotQA",
            "target_data_used_for_bank": bool(bank["target_data_used"]),
            "worker_model": config["worker"]["model"],
            "evaluator_model": config["evaluator"]["model"],
            "bank_frozen_during_target_evaluation": True,
        },
        "paired_comparison": {
            "paired_questions": len(per_question),
            "gam_f1": mean(row["gam_f1"] for row in per_question),
            "r2mem_f1": mean(row["r2mem_f1"] for row in per_question),
            "delta_f1": mean(deltas),
            "relative_gain": (
                mean(row["r2mem_f1"] for row in per_question)
                / mean(row["gam_f1"] for row in per_question)
                - 1.0
                if mean(row["gam_f1"] for row in per_question) > 0
                else None
            ),
            "delta_f1_bootstrap_95_ci": list(ci),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
            "improved_questions": sum(delta > epsilon for delta in deltas),
            "tied_questions": sum(abs(delta) <= epsilon for delta in deltas),
            "worsened_questions": sum(delta < -epsilon for delta in deltas),
        },
        "efficiency": {
            "gam_research_tokens_avg": mean(
                row["gam_research_tokens"] for row in per_question
            ),
            "r2mem_research_tokens_avg": mean(
                row["r2mem_research_tokens"] for row in per_question
            ),
            "gam_iterations_avg": mean(
                row["gam_iterations"] for row in per_question
            ),
            "r2mem_iterations_avg": mean(
                row["r2mem_iterations"] for row in per_question
            ),
        },
        "experience_bank": {
            "planning_entries": bank["planning"]["entries"],
            "planning_unique_situations": bank["planning"]["unique_situations"],
            "planning_unique_conditions": bank["planning"]["unique_conditions"],
            "reflection_entries": bank["reflection"]["entries"],
            "reflection_unique_situations": bank["reflection"]["unique_situations"],
            "reflection_unique_conditions": bank["reflection"]["unique_conditions"],
            "manifest_sha256": sha256_file(bank_path),
        },
        "experience_retrieval": {
            "events": retrieved_events,
            "candidates": retrieved_candidates,
            "mean_similarity": (
                weighted_similarity / retrieved_candidates
                if retrieved_candidates
                else 0.0
            ),
            "distinct_entries_used": len(distinct_entries),
        },
        "inputs": {
            "gam_batch_results": str(gam_path),
            "gam_batch_results_sha256": sha256_file(gam_path),
            "r2mem_batch_results": str(r2_path),
            "r2mem_batch_results_sha256": sha256_file(r2_path),
            "bank_manifest": str(bank_path),
        },
        "per_question": per_question,
    }

    report_root = root / "05_report"
    json_path = report_root / "cross_dataset_report.json"
    markdown_path = report_root / "cross_dataset_report.md"
    write_json(json_path, report)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = markdown_path.with_suffix(".md.tmp")
    temporary.write_text(markdown_report(report), encoding="utf-8")
    temporary.replace(markdown_path)
    print(f"Cross-dataset report: {markdown_path}")


if __name__ == "__main__":
    main()

