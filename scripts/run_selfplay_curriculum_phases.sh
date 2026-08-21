#!/usr/bin/env bash
set -uo pipefail

CONFIG_PATH="${1:-configs/training/selfplay_curriculum_v3.yaml}"
OUTPUT_PATH="${2:-outputs/selfplay/curriculum-v3}"
FAILURES=0

run_phase() {
  local round_index="$1"
  local phase="$2"
  shift 2

  printf '\n[selfplay] starting round %s %s\n' "$round_index" "$phase"
  if "$@"; then
    printf '[selfplay] completed round %s %s\n' "$round_index" "$phase"
  else
    local status="$?"
    FAILURES=$((FAILURES + 1))
    printf '[selfplay] round %s %s exited with status %s; continuing to the next command\n' \
      "$round_index" "$phase" "$status" >&2
  fi
}

run_phase 1 questioner python -m graphtask_r1.cli train self-play --config "$CONFIG_PATH" --output-dir "$OUTPUT_PATH" --round-index 1 --phase questioner
run_phase 1 solver python -m graphtask_r1.cli train self-play --config "$CONFIG_PATH" --output-dir "$OUTPUT_PATH" --round-index 1 --phase solver
run_phase 2 questioner python -m graphtask_r1.cli train self-play --config "$CONFIG_PATH" --output-dir "$OUTPUT_PATH" --round-index 2 --phase questioner
run_phase 2 solver python -m graphtask_r1.cli train self-play --config "$CONFIG_PATH" --output-dir "$OUTPUT_PATH" --round-index 2 --phase solver
run_phase 3 questioner python -m graphtask_r1.cli train self-play --config "$CONFIG_PATH" --output-dir "$OUTPUT_PATH" --round-index 3 --phase questioner
run_phase 3 solver python -m graphtask_r1.cli train self-play --config "$CONFIG_PATH" --output-dir "$OUTPUT_PATH" --round-index 3 --phase solver

if ((FAILURES > 0)); then
  printf '\n[selfplay] all six commands were attempted; %s returned a non-zero status\n' \
    "$FAILURES" >&2
  exit 1
fi

printf '\n[selfplay] all six commands completed successfully\n'
