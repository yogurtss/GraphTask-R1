#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/training/selfplay_curriculum_v3.yaml}"
OUTPUT_PATH="${2:-outputs/selfplay/curriculum-v3}"

python -m graphtask_r1.cli train self-play --config "$CONFIG_PATH" --output-dir "$OUTPUT_PATH" --round-index 1 --phase questioner
python -m graphtask_r1.cli train self-play --config "$CONFIG_PATH" --output-dir "$OUTPUT_PATH" --round-index 1 --phase solver
python -m graphtask_r1.cli train self-play --config "$CONFIG_PATH" --output-dir "$OUTPUT_PATH" --round-index 2 --phase questioner
python -m graphtask_r1.cli train self-play --config "$CONFIG_PATH" --output-dir "$OUTPUT_PATH" --round-index 2 --phase solver
python -m graphtask_r1.cli train self-play --config "$CONFIG_PATH" --output-dir "$OUTPUT_PATH" --round-index 3 --phase questioner
python -m graphtask_r1.cli train self-play --config "$CONFIG_PATH" --output-dir "$OUTPUT_PATH" --round-index 3 --phase solver
