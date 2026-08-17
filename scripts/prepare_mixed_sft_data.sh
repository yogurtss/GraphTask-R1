#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Edit this block before the first run. Every value can also be overridden with
# an environment variable of the same name.
# =============================================================================
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TRAIN_TASKS="${TRAIN_TASKS:-$PROJECT_ROOT/data/processed/graph/train/tasks.parquet}"
VAL_TASKS="${VAL_TASKS:-$PROJECT_ROOT/data/processed/graph/val/tasks.parquet}"
WORK_DIR="${WORK_DIR:-$PROJECT_ROOT/outputs/sft-data}"

# A 9:1 ratio produces approximately 90% Solver + 10% Questioner rows.
# All valid Solver train rows are retained; Questioner rows are sampled to match.
SOLVER_RATIO="${SOLVER_RATIO:-9}"
QUESTIONER_RATIO="${QUESTIONER_RATIO:-1}"
# Set a positive value to ignore the ratio and request this exact Questioner count.
QUESTIONER_COUNT_OVERRIDE="${QUESTIONER_COUNT_OVERRIDE:-0}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
MODEL_TYPE="${MODEL_TYPE:-qwen3}"
TEMPLATE="${TEMPLATE:-qwen3}"
AGENT_TEMPLATE="${AGENT_TEMPLATE:-hermes}"
MAX_LENGTH="${MAX_LENGTH:-32768}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# =============================================================================

TRAIN_VIEW="$WORK_DIR/tasks/train.parquet"
VAL_VIEW="$WORK_DIR/tasks/val.parquet"
RELATION_CATALOG="$WORK_DIR/relation_catalog.json"
RAW_DIR="$WORK_DIR/exported"
PREFLIGHT_DIR="$WORK_DIR/preflight"
SOLVER_TRAIN_RAW="$RAW_DIR/solver-train.parquet"
SOLVER_VAL_RAW="$RAW_DIR/solver-val.parquet"
QUESTIONER_TRAIN_RAW="$RAW_DIR/questioner-train.parquet"
MIXED_TRAIN_RAW="$RAW_DIR/mixed-train.parquet"
MIXED_TRAIN_ACCEPTED="$PREFLIGHT_DIR/mixed-train-accepted.parquet"
SOLVER_VAL_ACCEPTED="$PREFLIGHT_DIR/solver-val-accepted.parquet"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_positive_integer() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "$name must be a positive integer; got '$value'"
}

require_non_negative_integer() {
  local name="$1"
  local value="$2"
  [[ "$value" =~ ^[0-9]+$ ]] || die "$name must be a non-negative integer; got '$value'"
}

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

[[ -f "$TRAIN_TASKS" ]] || die "TRAIN_TASKS not found: $TRAIN_TASKS"
[[ -f "$VAL_TASKS" ]] || die "VAL_TASKS not found: $VAL_TASKS"
require_positive_integer SOLVER_RATIO "$SOLVER_RATIO"
require_positive_integer QUESTIONER_RATIO "$QUESTIONER_RATIO"
require_non_negative_integer QUESTIONER_COUNT_OVERRIDE "$QUESTIONER_COUNT_OVERRIDE"
require_positive_integer MAX_LENGTH "$MAX_LENGTH"
(( MAX_LENGTH <= 40960 )) || die "MAX_LENGTH must not exceed 40960"
require_non_negative_integer SEED "$SEED"

cd "$PROJECT_ROOT"
mkdir -p "$WORK_DIR/tasks" "$RAW_DIR" "$PREFLIGHT_DIR"

echo "[1/7] Audit certified train/val tasks and create lightweight training views"
run "$PYTHON_BIN" -m graphtask_r1.cli data audit \
  --input "$TRAIN_TASKS" --kind task --deep --training-view-output "$TRAIN_VIEW"
run "$PYTHON_BIN" -m graphtask_r1.cli data audit \
  --input "$VAL_TASKS" --kind task --deep --training-view-output "$VAL_VIEW"

echo "[2/7] Build a snapshot-wide relation catalog"
run "$PYTHON_BIN" -m graphtask_r1.cli data build-relation-catalog \
  --input "$TRAIN_VIEW" --scope graph --output "$RELATION_CATALOG"

echo "[3/7] Export Solver train/val rows"
run "$PYTHON_BIN" -m graphtask_r1.cli data export-sft \
  --input "$TRAIN_VIEW" --output "$SOLVER_TRAIN_RAW" --roles solver \
  --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog "$RELATION_CATALOG" --seed "$SEED"
run "$PYTHON_BIN" -m graphtask_r1.cli data export-sft \
  --input "$VAL_VIEW" --output "$SOLVER_VAL_RAW" --roles solver \
  --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog "$RELATION_CATALOG" --seed "$SEED"

SOLVER_ROWS="$($PYTHON_BIN -c \
  'import pyarrow.parquet as pq, sys; print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)' \
  "$SOLVER_TRAIN_RAW")"
require_positive_integer SOLVER_ROWS "$SOLVER_ROWS"
if (( QUESTIONER_COUNT_OVERRIDE > 0 )); then
  QUESTIONER_COUNT="$QUESTIONER_COUNT_OVERRIDE"
else
  # Nearest integer to solver_rows * questioner_weight / solver_weight.
  QUESTIONER_COUNT=$((
    (SOLVER_ROWS * QUESTIONER_RATIO + SOLVER_RATIO / 2) / SOLVER_RATIO
  ))
fi
if (( QUESTIONER_COUNT < 1 )); then
  QUESTIONER_COUNT=1
fi
require_positive_integer QUESTIONER_COUNT "$QUESTIONER_COUNT"

echo "[4/7] Export Questioner rows (Solver=$SOLVER_ROWS, Questioner=$QUESTIONER_COUNT)"
run "$PYTHON_BIN" -m graphtask_r1.cli data export-questioner-sft \
  --input "$TRAIN_VIEW" --output "$QUESTIONER_TRAIN_RAW" \
  --count "$QUESTIONER_COUNT" --seed "$SEED" \
  --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog "$RELATION_CATALOG"

echo "[5/7] Deterministically mix role-isolated raw rows"
run "$PYTHON_BIN" -m graphtask_r1.cli data combine-sft \
  --solver-input "$SOLVER_TRAIN_RAW" --questioner-input "$QUESTIONER_TRAIN_RAW" \
  --output "$MIXED_TRAIN_RAW" --seed "$SEED"

echo "[6/7] Apply the real training template and reject any overlong/invalid row"
run "$PYTHON_BIN" scripts/preflight_ms_swift_sft.py --require-all \
  --input "$MIXED_TRAIN_RAW" \
  --accepted-output "$MIXED_TRAIN_ACCEPTED" \
  --rejected-output "$PREFLIGHT_DIR/mixed-train-rejected.parquet" \
  --summary-output "$PREFLIGHT_DIR/mixed-train-summary.json" \
  --model "$MODEL_PATH" --model-type "$MODEL_TYPE" --template "$TEMPLATE" \
  --agent-template "$AGENT_TEMPLATE" --max-length "$MAX_LENGTH"
run "$PYTHON_BIN" scripts/preflight_ms_swift_sft.py --require-all \
  --input "$SOLVER_VAL_RAW" \
  --accepted-output "$SOLVER_VAL_ACCEPTED" \
  --rejected-output "$PREFLIGHT_DIR/solver-val-rejected.parquet" \
  --summary-output "$PREFLIGHT_DIR/solver-val-summary.json" \
  --model "$MODEL_PATH" --model-type "$MODEL_TYPE" --template "$TEMPLATE" \
  --agent-template "$AGENT_TEMPLATE" --max-length "$MAX_LENGTH"

echo "[7/7] Write reusable training environment"
ENV_FILE="$WORK_DIR/sft_data.env"
{
  printf 'export SFT_TRAIN_DATA=%q\n' "$MIXED_TRAIN_ACCEPTED"
  printf 'export SFT_VAL_DATA=%q\n' "$SOLVER_VAL_ACCEPTED"
  printf 'export TRAIN_DATA=%q\n' "$MIXED_TRAIN_ACCEPTED"
  printf 'export VAL_DATA=%q\n' "$SOLVER_VAL_ACCEPTED"
  printf 'export SFT_SOLVER_RATIO=%q\n' "$SOLVER_RATIO"
  printf 'export SFT_QUESTIONER_RATIO=%q\n' "$QUESTIONER_RATIO"
  printf 'export SFT_SOLVER_ROWS=%q\n' "$SOLVER_ROWS"
  printf 'export SFT_QUESTIONER_ROWS=%q\n' "$QUESTIONER_COUNT"
  printf 'export SFT_DATA_SEED=%q\n' "$SEED"
} > "$ENV_FILE"

TOTAL_ROWS=$((SOLVER_ROWS + QUESTIONER_COUNT))
echo "SFT data preparation completed."
echo "  requested role weights: Solver=$SOLVER_RATIO Questioner=$QUESTIONER_RATIO"
echo "  actual train rows:      Solver=$SOLVER_ROWS Questioner=$QUESTIONER_COUNT Total=$TOTAL_ROWS"
echo "  train data:             $MIXED_TRAIN_ACCEPTED"
echo "  validation data:        $SOLVER_VAL_ACCEPTED"
echo "  environment:            source $ENV_FILE"
