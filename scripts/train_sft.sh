#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:=Qwen/Qwen3-4B-Instruct-2507}"
: "${TRAIN_DATA:?Set TRAIN_DATA to the multi-turn SFT parquet file}"
: "${VAL_DATA:=$TRAIN_DATA}"
: "${OUTPUT_DIR:=outputs/sft-qwen3-4b}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
NUM_GPUS="${NUM_GPUS:-4}"
VERL_PROFILE="${VERL_PROFILE:-cuda128}"

case "$VERL_PROFILE" in
  cuda124)
    torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
      -m verl.trainer.fsdp_sft_trainer \
      data.train_files="$TRAIN_DATA" \
      data.val_files="$VAL_DATA" \
      data.train_batch_size="${TRAIN_BATCH_SIZE:-32}" \
      data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE:-1}" \
      data.multiturn.enable=true \
      data.multiturn.messages_key=messages \
      data.max_length="${MAX_LENGTH:-4096}" \
      optim.lr="${LR:-2e-5}" \
      model.partial_pretrain="$MODEL_PATH" \
      model.strategy=fsdp2 \
      model.fsdp_config.model_dtype=bf16 \
      model.enable_gradient_checkpointing=true \
      model.lora_rank="${LORA_RANK:-32}" \
      model.lora_alpha="${LORA_ALPHA:-64}" \
      model.target_modules=all-linear \
      use_remove_padding=true \
      ulysses_sequence_parallel_size=1 \
      trainer.default_local_dir="$OUTPUT_DIR" \
      trainer.project_name=graphtask-r1 \
      trainer.experiment_name="${EXPERIMENT_NAME:-qwen3-4b-shared-sft-cu124}" \
      trainer.logger="${LOGGER:-console}" \
      trainer.seed="${SEED:-42}" \
      trainer.total_epochs="${EPOCHS:-2}" \
      trainer.n_gpus_per_node="$NUM_GPUS" \
      "$@"
    ;;
  cuda128)
    torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
      -m verl.trainer.sft_trainer \
      data.train_files="$TRAIN_DATA" \
      data.val_files="$VAL_DATA" \
      data.train_batch_size="${TRAIN_BATCH_SIZE:-32}" \
      data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE:-1}" \
      data.max_length="${MAX_LENGTH:-4096}" \
      optim.lr="${LR:-2e-5}" \
      engine=fsdp \
      model.path="$MODEL_PATH" \
      model.use_remove_padding=true \
      model.enable_gradient_checkpointing=true \
      model.lora_rank="${LORA_RANK:-32}" \
      model.lora_alpha="${LORA_ALPHA:-64}" \
      model.target_modules=all-linear \
      trainer.default_local_dir="$OUTPUT_DIR" \
      trainer.project_name=graphtask-r1 \
      trainer.experiment_name="${EXPERIMENT_NAME:-qwen3-4b-shared-sft}" \
      trainer.logger="${LOGGER:-console}" \
      trainer.seed="${SEED:-42}" \
      trainer.total_epochs="${EPOCHS:-2}" \
      trainer.n_gpus_per_node="$NUM_GPUS" \
      "$@"
    ;;
  *)
    echo "Unsupported VERL_PROFILE=$VERL_PROFILE (expected cuda124 or cuda128)" >&2
    exit 2
    ;;
esac
