#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:=Qwen/Qwen3-4B-Instruct-2507}"
: "${MODEL_TYPE:=qwen3}"
: "${LORA_ADAPTER_PATH:?Set LORA_ADAPTER_PATH to an ms-swift LoRA checkpoint (SFT by default)}"
: "${TRAIN_DATA:?Set TRAIN_DATA to a Solver RL parquet file}"
: "${OUTPUT_DIR:=outputs/ms-swift-solver-grpo-cu124}"

MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-32768}"
if ! [[ "$MAX_COMPLETION_LENGTH" =~ ^[0-9]+$ ]] || (( MAX_COMPLETION_LENGTH < 1 || MAX_COMPLETION_LENGTH > 40960 )); then
  echo "MAX_COMPLETION_LENGTH must be an integer between 1 and 40960" >&2
  exit 2
fi
INTERACTION_MODE="${INTERACTION_MODE:-graphscript}"
if [[ "$INTERACTION_MODE" != "tool" && "$INTERACTION_MODE" != "graphscript" ]]; then
  echo "INTERACTION_MODE must be tool or graphscript" >&2
  exit 2
fi
VLLM_MODE="${VLLM_MODE:-server}"
if [[ "$VLLM_MODE" != "server" && "$VLLM_MODE" != "colocate" ]]; then
  echo "VLLM_MODE must be server or colocate" >&2
  exit 2
fi
USE_VLLM="${USE_VLLM:-true}"
if [[ "$USE_VLLM" != "true" && "$USE_VLLM" != "false" ]]; then
  echo "USE_VLLM must be true or false" >&2
  exit 2
fi
DEEPSPEED="${DEEPSPEED:-none}"
DEEPSPEED_ARGS=()
case "$DEEPSPEED" in
  none)
    ;;
  zero0|zero1|zero2|zero3|zero2_offload|zero3_offload)
    DEEPSPEED_ARGS=(--deepspeed "$DEEPSPEED")
    ;;
  *)
    echo "DEEPSPEED must be one of: none, zero0, zero1, zero2, zero3, zero2_offload, zero3_offload" >&2
    exit 2
    ;;
esac
RL_ALGORITHM="${RL_ALGORITHM:-grpo}"
RL_ALGORITHM_ARGS=()
case "$RL_ALGORITHM" in
  grpo)
    RL_ALGORITHM_ARGS=(
      --advantage_estimator grpo
      --scale_rewards group
      --kl_in_reward false
    )
    ;;
  reinforce_plus_plus)
    RL_ALGORITHM_ARGS=(
      --advantage_estimator reinforce_plus_plus
      --scale_rewards batch
      --kl_in_reward true
    )
    ;;
  *)
    echo "RL_ALGORITHM must be one of: grpo, reinforce_plus_plus" >&2
    exit 2
    ;;
esac
LOG_ENTROPY="${LOG_ENTROPY:-true}"
if [[ "$LOG_ENTROPY" != "true" && "$LOG_ENTROPY" != "false" ]]; then
  echo "LOG_ENTROPY must be true or false" >&2
  exit 2
fi
# A rollout is generated once per window. Logging at that boundary prevents
# ms-swift from clearing rollout metrics on an earlier optimizer-only step.
LOGGING_STEPS="${LOGGING_STEPS:-${STEPS_PER_GENERATION:-4}}"
VLLM_COLOCATE_ARGS=()
VLLM_SERVER_ARGS=()
VLLM_MODE_ARGS=()
if [[ "$USE_VLLM" == "false" ]]; then
  :
elif [[ "$VLLM_MODE" == "server" ]]; then
  VLLM_MODE_ARGS=(--vllm_mode "$VLLM_MODE")
  VLLM_SERVER_ARGS=(
    --vllm_server_host "${VLLM_SERVER_HOST:-127.0.0.1}"
    --vllm_server_port "${VLLM_SERVER_PORT:-8000}"
  )
else
  VLLM_MODE_ARGS=(--vllm_mode "$VLLM_MODE")
  VLLM_COLOCATE_ARGS=(
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.5}"
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN:-16384}"
    --sleep_level "${VLLM_SLEEP_LEVEL:-1}"
  )
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if ! command -v swift >/dev/null 2>&1; then
  echo "ms-swift CLI not found; install the optional vLLM GRPO environment first" >&2
  exit 2
fi
if ! python -c 'import math_verify' >/dev/null 2>&1; then
  echo "math_verify is required by ms-swift GRPO; install it in the training environment" >&2
  exit 2
fi
if [[ "$DEEPSPEED" != "none" ]] && ! python -c 'import deepspeed' >/dev/null 2>&1; then
  echo "deepspeed is required when DEEPSPEED=$DEEPSPEED" >&2
  exit 2
fi
if [[ ! -d "$LORA_ADAPTER_PATH" ]]; then
  echo "SFT adapter directory not found: $LORA_ADAPTER_PATH" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "Solver RL parquet not found; generate it or fix TRAIN_DATA" >&2
  exit 2
fi

export GRAPHTASK_MS_SWIFT_DATA_KIND=rl
export GRAPHTASK_MS_SWIFT_TRAIN_DATA="$TRAIN_DATA"
export INTERACTION_MODE
export RL_ALGORITHM
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-1,2,3}"
NUM_GPUS="${NUM_GPUS:-3}"
MULTI_TURN_ARGS=()
if [[ "$INTERACTION_MODE" == "tool" ]]; then
  MULTI_TURN_ARGS=(--multi_turn_scheduler graphtask_solver --max_turns "${MAX_TURNS:-8}")
fi

NPROC_PER_NODE="$NUM_GPUS" swift rlhf \
  --rlhf_type grpo \
  "${RL_ALGORITHM_ARGS[@]}" \
  --model "$MODEL_PATH" \
  --model_type "$MODEL_TYPE" \
  --adapters "$LORA_ADAPTER_PATH" \
  --train_type lora \
  --dataset graphtask-train \
  --eval_strategy no \
  --external_plugins "$PROJECT_ROOT/graphtask_r1/training/ms_swift_plugin.py" \
  --reward_funcs graphtask_score \
  --agent_template hermes \
  --loss_scale default \
  "${DEEPSPEED_ARGS[@]}" \
  --use_vllm "$USE_VLLM" \
  "${VLLM_MODE_ARGS[@]}" \
  "${VLLM_COLOCATE_ARGS[@]}" \
  "${VLLM_SERVER_ARGS[@]}" \
  "${MULTI_TURN_ARGS[@]}" \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --num_train_epochs "${EPOCHS:-1}" \
  --per_device_train_batch_size "${MICRO_BATCH_SIZE:-1}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --steps_per_generation "${STEPS_PER_GENERATION:-4}" \
  --learning_rate "${LR:-2e-6}" \
  --lora_rank "${LORA_RANK:-32}" \
  --lora_alpha "${LORA_ALPHA:-64}" \
  --target_modules all-linear \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --num_generations "${ROLLOUT_N:-4}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant":false}' \
  --dataset_num_proc "${DATASET_NUM_PROC:-1}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-1}" \
  --load_from_cache_file false \
  --split_dataset_ratio 0 \
  --save_steps "${SAVE_STEPS:-20}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --logging_steps "$LOGGING_STEPS" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --log_entropy "$LOG_ENTROPY" \
  --log_completions true \
  --report_to none \
  --seed "${SEED:-42}" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
