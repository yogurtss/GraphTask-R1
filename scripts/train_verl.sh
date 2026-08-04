#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to Qwen/Qwen3-4B-Instruct-2507 or a local copy}"
: "${TRAIN_DATA:?Set TRAIN_DATA to the mixed-role parquet file}"
: "${VAL_DATA:=$TRAIN_DATA}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_GPUS="${NUM_GPUS:-4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-graphtask-r1-shared-lora}"
: "${OUTPUT_DIR:=outputs/verl/$EXPERIMENT_NAME}"
MODEL_ARGS=()
if [[ -n "${LORA_ADAPTER_PATH:-}" ]]; then
  MODEL_ARGS+=("actor_rollout_ref.model.lora_adapter_path=$LORA_ADAPTER_PATH")
fi

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-64}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-2048}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-2048}" \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  data.return_raw_chat=true \
  data.shuffle=false \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.lora_rank="${LORA_RANK:-32}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA:-64}" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.actor.optim.lr="${LR:-2e-6}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE:-16}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE:-1}" \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef="${KL_COEF:-0.001}" \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.ref.strategy=fsdp2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-8}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE:-1}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.6}" \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS:-8}" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$PROJECT_ROOT/configs/training/verl_tools.yaml" \
  custom_reward_function.path="$PROJECT_ROOT/src/graphtask_r1/training/verl_reward.py" \
  custom_reward_function.name=compute_score \
  trainer.use_legacy_worker_impl=disable \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.project_name=graphtask-r1 \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.nnodes="${NUM_NODES:-1}" \
  trainer.save_freq="${SAVE_FREQ:-20}" \
  trainer.test_freq="${TEST_FREQ:-20}" \
  trainer.total_epochs="${EPOCHS:-1}" \
  "${MODEL_ARGS[@]}" \
  "$@"
