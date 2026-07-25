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
retrievers, and experience-aware research agent. It replaces the submitted
scripts' hard-coded paths and global variables with one explicit JSON
configuration. It does not use any HotpotQA trajectory or answer when building
the experience bank.

## Why `conv-26` is selected

`conv-26` is the first element (array index 0) of `data/locomo/locomo10.json`.
It contains 199 questions, of which the original LoCoMo runner evaluates 152
questions after excluding category 5. This is also the conversation selected by
the submitted `exp/self_reflection.py`.

## 1. Environment

Use Python 3.10+ on a Linux machine with an NVIDIA GPU. From the original
source root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r /path/to/cross_dataset_validation/requirements.txt
pip install -e ./gam
```

Start an OpenAI-compatible vLLM server. The served name must match the JSON
configuration:

```bash
vllm serve /path/to/Qwen2.5-3B-Instruct \
  --served-model-name qwen2.5-3B-Instruct \
  --trust-remote-code \
  --gpu-memory-utilization 0.70 \
  --max-model-len 32768 \
  --port 8001
```

`vllm` is only required in the server environment. The validation scripts talk
to it through its OpenAI-compatible API.

## 2. Configuration

```bash
cd /path/to/cross_dataset_validation
cp config.example.json config.local.json
```

Edit these fields:

- `paths.source_root`: original submitted source directory.
- `paths.locomo_data`: `locomo10.json`.
- `paths.hotpotqa_data`: local target path; the prepare stage downloads it when
  absent.
- `embedding.model`: local BGE-M3 path or `BAAI/bge-m3`.
- `worker`: the model used by GAM, the R²-Mem learner, and HotpotQA inference.
- `evaluator`: the rubric evaluator.

The example config intentionally uses the same Qwen-3B endpoint for `worker`
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

```bash
python run_pipeline.py --config config.local.json
```

The stages are resumable. Successfully completed per-question outputs and
self-reflection checkpoints are reused on rerun.

Run the dependency-free protocol tests:

```bash
python -m unittest discover -s tests -v
```

Individual stages:

```bash
python 00_prepare.py --config config.local.json
python 01_run_locomo_gam.py --config config.local.json
python 02_build_experience_bank.py --config config.local.json
python 03_run_hotpotqa.py --config config.local.json --mode both
python 04_summarize.py --config config.local.json
```

For a small end-to-end smoke test:

```bash
python run_pipeline.py --config config.local.json --smoke-test 10
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
