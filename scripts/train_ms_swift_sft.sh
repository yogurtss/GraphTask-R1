#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:=Qwen/Qwen3-4B-Instruct-2507}"
: "${MODEL_TYPE:=qwen3}"
: "${TRAIN_DATA:?Set TRAIN_DATA to the existing kqapro_sft_train.parquet}"
: "${VAL_DATA:=$TRAIN_DATA}"
: "${OUTPUT_DIR:=outputs/ms-swift-sft-qwen3-4b-cu124}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if ! command -v swift >/dev/null 2>&1; then
  echo "ms-swift CLI not found; install ms-swift==3.6.4 in the CUDA 12.4 environment" >&2
  exit 2
fi
if [[ ! -f "$TRAIN_DATA" || ! -f "$VAL_DATA" ]]; then
  echo "Existing SFT parquet not found; do not regenerate KQA, fix TRAIN_DATA/VAL_DATA" >&2
  exit 2
fi

export GRAPHTASK_MS_SWIFT_DATA_KIND=sft
export GRAPHTASK_MS_SWIFT_TRAIN_DATA="$TRAIN_DATA"
export GRAPHTASK_MS_SWIFT_VAL_DATA="$VAL_DATA"
NUM_GPUS="${NUM_GPUS:-4}"

NPROC_PER_NODE="$NUM_GPUS" swift sft \
  --model "$MODEL_PATH" \
  --model_type "$MODEL_TYPE" \
  --train_type lora \
  --dataset graphtask-train \
  --val_dataset graphtask-val \
  --external_plugins "$PROJECT_ROOT/graphtask_r1/training/ms_swift_plugin.py" \
  --agent_template hermes \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --num_train_epochs "${EPOCHS:-2}" \
  --per_device_train_batch_size "${MICRO_BATCH_SIZE:-1}" \
  --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}" \
  --learning_rate "${LR:-2e-5}" \
  --lora_rank "${LORA_RANK:-32}" \
  --lora_alpha "${LORA_ALPHA:-64}" \
  --target_modules all-linear \
  --max_length "${MAX_LENGTH:-4096}" \
  --gradient_checkpointing true \
  --gradient_checkpointing_kwargs '{"use_reentrant":false}' \
  --dataset_num_proc "${DATASET_NUM_PROC:-1}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-1}" \
  --load_from_cache_file false \
  --eval_steps "${EVAL_STEPS:-100}" \
  --save_steps "${SAVE_STEPS:-100}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-2}" \
  --logging_steps "${LOGGING_STEPS:-5}" \
  --warmup_ratio "${WARMUP_RATIO:-0.05}" \
  --save_only_model true \
  --report_to none \
  --seed "${SEED:-42}" \
  --output_dir "$OUTPUT_DIR" \
  "$@"
