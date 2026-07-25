#!/bin/bash

# Activate your conda/virtual environment if needed
# source /path/to/your/conda/bin/activate your_env

export HF_DATASETS_CACHE=your_path/hf_cache_narrativeqa
mkdir -p your_path/hf_cache_narrativeqa

# Set output directory
outputdir=your_path/results/narrativeqa

# Create output directory
mkdir -p $outputdir

# Run NarrativeQA evaluation
python3 exp/eval/datasets_test/narrativeqa_exp.py \
    --data-dir data/narrativeqa \
    --split test \
    --outdir $outputdir \
    --start-idx 30 \
    --end-idx 300 \
    --max-tokens 2048 \
    --seed 42 \
    --memory-api-key "empty" \
    --memory-base-url "http://localhost:8001/v1" \
    --memory-model "qwen2.5-7B-Instruct" \
    --memory-api-type "vllm" \
    --research-api-key "empty" \
    --research-base-url "http://localhost:8001/v1" \
    --research-model "qwen2.5-7B-Instruct" \
    --research-api-type "vllm" \
    --working-api-key "empty" \
    --working-base-url "http://localhost:8001/v1" \
    --working-model "qwen2.5-7B-Instruct" \
    --working-api-type "vllm" \
    --embedding-model-path your_path/BAAI/bge-m3

