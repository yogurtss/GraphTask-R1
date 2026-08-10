#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:=Qwen/Qwen3-4B-Instruct-2507}"
: "${MODEL_TYPE:=qwen3}"
: "${LORA_ADAPTER_PATH:?Set LORA_ADAPTER_PATH to an ms-swift SFT checkpoint}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if ! command -v swift >/dev/null 2>&1; then
  echo "ms-swift CLI not found; install the optional vLLM GRPO environment first" >&2
  exit 2
fi
if [[ ! -d "$LORA_ADAPTER_PATH" ]]; then
  echo "SFT adapter directory not found: $LORA_ADAPTER_PATH" >&2
  exit 2
fi

unset GRAPHTASK_MS_SWIFT_DATA_KIND
unset GRAPHTASK_MS_SWIFT_TRAIN_DATA
unset GRAPHTASK_MS_SWIFT_VAL_DATA
export CUDA_VISIBLE_DEVICES="${ROLLOUT_CUDA_VISIBLE_DEVICES:-0}"

swift rollout \
  --model "$MODEL_PATH" \
  --model_type "$MODEL_TYPE" \
  --adapters "$LORA_ADAPTER_PATH" \
  --external_plugins "$PROJECT_ROOT/graphtask_r1/training/ms_swift_plugin.py" \
  --agent_template hermes \
  --multi_turn_scheduler graphtask_solver \
  --max_turns "${MAX_TURNS:-8}" \
  --use_async_engine true \
  --tensor_parallel_size "${ROLLOUT_TP_SIZE:-1}" \
  --vllm_max_lora_rank "${LORA_RANK:-32}" \
  --max_model_len "${VLLM_MAX_MODEL_LEN:-4096}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.8}" \
  --port "${VLLM_SERVER_PORT:-8000}" \
  --seed "${SEED:-42}" \
  "$@"
