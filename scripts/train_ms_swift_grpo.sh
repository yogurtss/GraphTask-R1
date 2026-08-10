#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:=Qwen/Qwen3-4B-Instruct-2507}"
: "${LORA_ADAPTER_PATH:?Set LORA_ADAPTER_PATH to an ms-swift SFT checkpoint}"
: "${TRAIN_DATA:?Set TRAIN_DATA to the existing kqapro_solver_rl.parquet}"
: "${VAL_DATA:=$TRAIN_DATA}"
: "${OUTPUT_DIR:=outputs/ms-swift-solver-grpo-cu124}"

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
if [[ ! -f "$TRAIN_DATA" || ! -f "$VAL_DATA" ]]; then
  echo "Existing Solver RL parquet not found; fix TRAIN_DATA/VAL_DATA without regenerating KQA" >&2
  exit 2
fi

export GRAPHTASK_MS_SWIFT_DATA_KIND=rl
export GRAPHTASK_MS_SWIFT_TRAIN_DATA="$TRAIN_DATA"
export GRAPHTASK_MS_SWIFT_VAL_DATA="$VAL_DATA"
export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-1,2,3}"
NUM_GPUS="${NUM_GPUS:-3}"

NPROC_PER_NODE="$NUM_GPUS" swift rlhf \
  --rlhf_type grpo \
  --model "$MODEL_PATH" \
  --adapters "$LORA_ADAPTER_PATH" \
  --train_type lora \
  --dataset graphtask-train \
  --val_dataset graphtask-val \
  --external_plugins "$PROJECT_ROOT/graphtask_r1/training/ms_swift_plugin.py" \
  --reward_funcs graphtask_score \
  --agent_template hermes \
  --loss_scale default \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host "${VLLM_SERVER_HOST:-127.0.0.1}" \
  --vllm_server_port "${VLLM_SERVER_PORT:-8000}" \
  --multi_turn_scheduler graphtask_solver \
  --max_turns "${MAX_TURNS:-8}" \
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
  --max_completion_length "${MAX_COMPLETION_LENGTH:-2048}" \
  --num_generations "${ROLLOUT_N:-4}" \
  --temperature "${TEMPERATURE:-1.0}" \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant":false}' \
  --dataset_num_proc "${DATASET_NUM_PROC:-1}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-1}" \
  --load_from_cache_file false \
  --split_dataset_ratio 0 \
  --eval_steps "${EVAL_STEPS:-20}" \
  --save_steps "${SAVE_STEPS:-20}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --log_completions true \
  --report_to none \
  --seed "${SEED:-42}" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
