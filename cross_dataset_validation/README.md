# Cross-dataset validation: LoCoMo → HotpotQA

This directory provides a clean, auditable validation pipeline for the
cross-dataset generalization of R²-Mem:

```text
LoCoMo conv-26
  → baseline GAM trajectories
  → rubric-guided evaluation and self-reflection
  → Planning/Reflection experience banks
  → HotpotQA evaluation with the frozen LoCoMo bank
  → paired comparison against GAM on the same HotpotQA questions
```

The pipeline reuses the submitted GAM agents, R²-Mem prompts, schemas,
retrievers, and experience-aware research agent. A clean copy of the required
submitted implementation is vendored under `r2mem_core/`, so this validation
folder has no dependency on a separate old source checkout. The wrappers replace
the submitted scripts' hard-coded paths and global variables with one explicit
JSON configuration. No HotpotQA trajectory or answer is used when building the
experience bank.

## Why `conv-26` is selected

`conv-26` is the first element (array index 0) of `data/locomo/locomo10.json`.
It contains 199 questions, of which the original LoCoMo runner evaluates 152
questions after excluding category 5. This is also the conversation selected by
the submitted `exp/self_reflection.py`.

## 1. Environment

The expected server layout is:

```text
/NAS/wangxy/rebuttal/
├── cross_dataset_validation/
│   └── r2mem_core/
├── hotpotqa/
│   ├── eval_400.json
│   ├── eval_1600.json
│   └── eval_3200.json
└── locomo/
    └── locomo10.json
```

Only `eval_400.json` is used by this validation. No dataset download is
performed when using `config.server.json`.

Use Python 3.10+ on a Linux machine with an NVIDIA GPU:

```bash
cd /NAS/wangxy/rebuttal/cross_dataset_validation
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./r2mem_core/gam
```

Start an OpenAI-compatible vLLM server. The served name must match the JSON
configuration:

```bash
vllm serve /NAS/wangxy/qwen2.5-3B-Instruct \
  --served-model-name qwen2.5-3B-Instruct \
  --trust-remote-code \
  --gpu-memory-utilization 0.70 \
  --max-model-len 32768 \
  --port 8001
```

`vllm` is only required in the server environment. The validation scripts talk
to it through its OpenAI-compatible API.

## 2. Configuration

The ready-to-run server configuration is `config.server.json`. It already uses:

- source code: `r2mem_core/`;
- LoCoMo: `../locomo/locomo10.json`;
- HotpotQA: `../hotpotqa/eval_400.json`;
- BGE-M3: `/NAS/wangxy/BAAI/bge-m3`;
- Qwen API: `http://localhost:8001/v1`;
- served model name: `qwen2.5-3B-Instruct`.

The server config intentionally uses the same Qwen-3B endpoint for `worker`
and `evaluator`. This is the self-evolution setting and rules out a stronger
evaluator as the source of cross-dataset gains. To reproduce the paper's main
setting, point `evaluator` to a GPT-4o-compatible endpoint and export its key:

```bash
export R2MEM_EVALUATOR_API_KEY=...
```

API keys should be supplied through environment variables, not committed to the
configuration file.

The example config also pins the submitted LoCoMo file by SHA-256. It pins
HotpotQA to Hugging Face dataset commit
`27275ff4fee67ac0acb6478e405e7ac07efbdc1a` and verifies the downloaded
`eval_400.json` against its SHA-256 checksum.

## 3. Run the complete pipeline

First validate the local datasets without downloading anything:

```bash
python 00_prepare.py --config config.server.json --no-download
```

Run a separate smoke test:

```bash
python run_pipeline.py --config config.server.json --smoke-test 10
```

Then run the complete experiment:

```bash
python run_pipeline.py --config config.server.json
```

The stages are resumable. Successfully completed per-question outputs and
self-reflection checkpoints are reused on rerun.

Run the dependency-free protocol tests:

```bash
python -m unittest discover -s tests -v
```

Individual stages:

```bash
python 00_prepare.py --config config.server.json --no-download
python 01_run_locomo_gam.py --config config.server.json
python 02_build_experience_bank.py --config config.server.json
python 03_run_hotpotqa.py --config config.server.json --mode both
python 04_summarize.py --config config.server.json
```

For a small end-to-end smoke test:

```bash
python run_pipeline.py --config config.server.json --smoke-test 10
```

The pipeline automatically gives the smoke test a separate output directory so
that it cannot contaminate the full run. Smoke-test outputs must not be reported
as experimental results.

## 4. Output contract

```text
output_root/
├── 00_manifest/
├── 01_locomo_gam/
│   └── conv-26/
│       ├── qa_results.json
│       └── research_trace_q*.json
├── 02_experience_bank/
│   ├── Planning_exp_3B_X.json
│   ├── Planning_3B_metadata.json
│   ├── Planning_3B_situation.index
│   ├── Reflection_exp_3B_X.json
│   ├── Reflection_3B_metadata.json
│   ├── Reflection_3B_situation.index
│   ├── checkpoints/
│   └── bank_manifest.json
├── 03_hotpotqa_gam/
│   ├── batch_results_0_127.json
│   └── batch_statistics_0_127.json
├── 04_hotpotqa_r2mem/
│   ├── batch_results_0_127.json
│   └── batch_statistics_0_127.json
└── 05_report/
    ├── cross_dataset_report.json
    └── cross_dataset_report.md
```

The final report uses paired question IDs and includes GAM/R²-Mem F1, paired
delta, bootstrap 95% confidence interval, improved/tied/worse counts, research
tokens, search iterations, and retrieved-experience statistics.

## Protocol decisions

- Source experience data: LoCoMo `conv-26` only.
- Target evaluation data: all configured HotpotQA examples.
- No HotpotQA construction subset is excluded because the bank contains no
  HotpotQA trajectories.
- High-quality threshold: rubric sum ≥ 10.
- Low-quality threshold: rubric sum ≤ 5.
- Experience embedding: normalized BGE-M3 embedding of
  `(situation) + condition`, indexed by exact inner-product FAISS search.
- Experience bank is frozen throughout HotpotQA evaluation.
- GAM and R²-Mem are evaluated on the same target question IDs.
