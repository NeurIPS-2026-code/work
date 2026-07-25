#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${R2MEM_MODEL_PATH:-/NAS/wangxy/qwen2.5-3B-Instruct}"
SERVED_MODEL_NAME="${R2MEM_SERVED_MODEL_NAME:-qwen2.5-3B-Instruct}"
PORT="${R2MEM_VLLM_PORT:-8001}"
GPU_MEMORY_UTILIZATION="${R2MEM_GPU_MEMORY_UTILIZATION:-0.70}"

vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --trust-remote-code \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len 32768 \
  --port "$PORT"
