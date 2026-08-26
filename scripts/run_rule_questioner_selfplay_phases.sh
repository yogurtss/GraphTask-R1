#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$PROJECT_ROOT/configs/training/selfplay_qwen3_4b_rule_questioner_large.yaml}"
OUTPUT_PATH="${2:-$PROJECT_ROOT/outputs/selfplay/rule-questioner-qwen3-4b-10k}"

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    printf '[rule-selfplay] required environment variable %s is not set\n' "$name" >&2
    return 1
  fi
}

require_file() {
  local name="$1"
  local path="${!name:-}"
  require_value "$name"
  if [[ ! -f "$path" ]]; then
    printf '[rule-selfplay] %s does not exist: %s\n' "$name" "$path" >&2
    return 1
  fi
}

require_value INITIAL_ADAPTER
if [[ ! -f "$INITIAL_ADAPTER/adapter_config.json" ]]; then
  printf '[rule-selfplay] INITIAL_ADAPTER is not a complete adapter: %s\n' \
    "$INITIAL_ADAPTER" >&2
  exit 1
fi
require_file BASE_TASKS
require_file QUESTIONER_SEEDS
require_file KQAPRO_RELATION_CATALOG
require_file GRAPHTASK_KQAPRO_DB

if [[ ! -f "$CONFIG_PATH" ]]; then
  printf '[rule-selfplay] config does not exist: %s\n' "$CONFIG_PATH" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

printf '[rule-selfplay] config: %s\n' "$CONFIG_PATH"
printf '[rule-selfplay] output: %s\n' "$OUTPUT_PATH"
printf '[rule-selfplay] questioner seeds: %s\n' "$QUESTIONER_SEEDS"
printf '[rule-selfplay] graph DB: %s\n' "$GRAPHTASK_KQAPRO_DB"

exec bash "$PROJECT_ROOT/scripts/run_selfplay_curriculum_phases.sh" \
  "$CONFIG_PATH" "$OUTPUT_PATH"
