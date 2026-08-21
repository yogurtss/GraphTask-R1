# KQAPro 训练流程

这是一份从原始 KQAPro 数据到训练的完整入口。默认流程是：

```text
数据转换 → Solver + Questioner SFT → curriculum self-play → val 选模
                              └→ 可选 Solver-only GRPO warm-up
```

KQAPro train 可进入训练；val 只用于评测和 checkpoint 选择；test 不用于训练。所有 gold answer
都由转换后的 certified program 执行产生。Solver-only GRPO 不是前置依赖。

## 1. 环境和路径

推荐 Python 3.10、PyTorch 2.6.0+cu124、`ms-swift==3.10.3`。安装方法见
[ms-swift CUDA 12.4 环境](MS_SWIFT_CUDA_12_4.md)。

```bash
export PYTHONPATH=$PWD
export KQAPRO_RAW=$PWD/data/raw/kqa_pro
export KQAPRO_DIR=$PWD/data/processed/kqapro/kqapro-v1
export KQAPRO_SMOKE_DIR=$PWD/data/processed/kqapro/kqapro-v1-smoke
export SFT_DATA_DIR=$PWD/outputs/sft-data
export KQAPRO_TRAINING=$PWD/data/training

mkdir -p "$SFT_DATA_DIR" "$KQAPRO_TRAINING"
```

`KQAPRO_RAW` 中应有 `kb.json`、`train.json`、`val.json` 和 `test.json`。如果尚未下载：

```bash
python -m pip install huggingface_hub
python -m graphtask_r1.cli data fetch --dataset kqapro --raw-dir "$PWD/data/raw"
find "$KQAPRO_RAW" -maxdepth 1 -type f
```

## 2. 数据准备

先跑 20 条 bounded smoke：

```bash
python -m graphtask_r1.cli data prepare \
  --dataset kqapro \
  --raw-dir "$KQAPRO_RAW" \
  --output-dir "$KQAPRO_SMOKE_DIR" \
  --splits train,val --limit 20 --train-sample-size 20 \
  --verification-mode full --trace-mode canonical \
  --seed 42 --workers 1
```

smoke 通过后准备正式数据。下面从 train 分层抽取 20,000 条并处理完整 val；若要完整 train，
把 `--train-sample-size` 改为 `0`。

```bash
python -m graphtask_r1.cli data prepare \
  --dataset kqapro \
  --raw-dir "$KQAPRO_RAW" \
  --output-dir "$KQAPRO_DIR" \
  --splits train,val --train-sample-size 20000 \
  --verification-mode source --trace-mode none \
  --max-witness-facts 0 --seed 42 --workers 1
```

主要产物：

```text
$KQAPRO_DIR/
├── graph.sqlite
├── train/tasks.parquet
├── val/tasks.parquet
├── train/rejections.parquet
└── val/rejections.parquet
```

同一输入、seed 和版本可重复生成相同任务。正式产物建议放 Linux 文件系统；WSL 下不要直接把
SQLite 写到 `/mnt/g`。已有匹配的 `graph.sqlite` 会复用，需要重建时才加
`--rebuild-graph`。

## 3. 生成 SFT 数据

一条脚本完成 deep audit、training view、relation catalog、Solver/Questioner 导出、1:1 混合和
真实 ms-swift template 长度预检：

```bash
export TRAIN_TASKS="$KQAPRO_DIR/train/tasks.parquet"
export VAL_TASKS="$KQAPRO_DIR/val/tasks.parquet"
export WORK_DIR="$SFT_DATA_DIR"
export GRAPH_DB_PATH="$KQAPRO_DIR/graph.sqlite"
export MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
export MODEL_TYPE=qwen3
export SOLVER_RATIO=1
export QUESTIONER_RATIO=1
export MAX_LENGTH=32768

bash scripts/prepare_mixed_sft_data.sh
```

如果只想要固定数量的 Questioner 行，可额外设置
`QUESTIONER_COUNT_OVERRIDE=2048`。脚本不会重复 SFT 行；某个角色数据不足时会同步下采样另一
角色以保持比例。

脚本最后生成 `$SFT_DATA_DIR/sft_data.env`。训练前必须 source 它：

```bash
source "$SFT_DATA_DIR/sft_data.env"

printf 'SFT_TRAIN_DATA=%s\n' "$SFT_TRAIN_DATA"
printf 'SFT_VAL_DATA=%s\n' "$SFT_VAL_DATA"
printf 'QUESTIONER_SEEDS=%s\n' "$QUESTIONER_SEEDS"

python scripts/validate_ms_swift_data.py \
  --kind sft --input "$SFT_TRAIN_DATA" "$SFT_VAL_DATA"
```

环境文件包含：

| 变量 | 内容 |
| --- | --- |
| `SFT_TRAIN_DATA` | 预检通过的 mixed Solver + Questioner SFT |
| `SFT_VAL_DATA` | 预检通过的 Solver-only val SFT |
| `TRAIN_DATA` / `VAL_DATA` | 供直接运行 shell launcher 使用的同路径别名 |
| `QUESTIONER_SEEDS` | self-play 的 Questioner seed pool |

`configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml` 读取
`SFT_TRAIN_DATA/SFT_VAL_DATA`。不要把它们手工指向 certified task 或 RL Parquet。

## 4. SFT

```bash
source "$SFT_DATA_DIR/sft_data.env"
export SFT_OUTPUT_DIR=$PWD/outputs/sft/qwen3-4b-kqapro-v03
export NUM_GPUS=4
export MAX_LENGTH=32768

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml \
  --dry-run

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml
```

直接运行 launcher 时使用 env 文件提供的 `TRAIN_DATA/VAL_DATA`：

```bash
source "$SFT_DATA_DIR/sft_data.env"
export OUTPUT_DIR=$PWD/outputs/sft/qwen3-4b-kqapro-v03
bash scripts/train_ms_swift_sft.sh
```

### SFT batch 设置

```text
global batch = NUM_GPUS × MICRO_BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS
```

默认 YAML 为 `4 × 1 × 8 = 32`。显存不足时先降低 micro batch，再相应提高 gradient
accumulation；不要依靠截断丢掉 GraphScript 尾部。

## 5. Questioner/Solver self-play

先导出 self-play 使用的 Solver RL val。这个步骤只转换格式，不会运行 GRPO：

```bash
python -m graphtask_r1.cli data export-rl \
  --input "$SFT_DATA_DIR/tasks/val.parquet" \
  --output "$KQAPRO_TRAINING/kqapro_graphscript_v03_solver_rl_val.parquet" \
  --roles solver --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog "$SFT_DATA_DIR/relation_catalog.json" --seed 42
```

然后设置 self-play 环境。注意这里会把 `VAL_DATA` 从 SFT schema 改为 RL schema：

```bash
source "$SFT_DATA_DIR/sft_data.env"

export INITIAL_ADAPTER=$PWD/outputs/sft/qwen3-4b-kqapro-v03/checkpoint-last
export BASE_TASKS="$SFT_DATA_DIR/tasks/train.parquet"
export VAL_DATA="$KQAPRO_TRAINING/kqapro_graphscript_v03_solver_rl_val.parquet"
export KQAPRO_RELATION_CATALOG="$SFT_DATA_DIR/relation_catalog.json"

python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay_curriculum_v3.yaml \
  --output-dir outputs/selfplay/curriculum-v3 \
  --dry-run

python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay_curriculum_v3.yaml \
  --output-dir outputs/selfplay/curriculum-v3
```

### 推荐：三轮拆成六个独立进程

在 GPU `Exclusive Process`、或 torch distributed 在阶段切换时出现 native pointer/free 错误的
服务器上，推荐直接运行六阶段脚本：

```bash
bash scripts/run_selfplay_curriculum_phases.sh \
  configs/training/selfplay_curriculum_v3.yaml \
  outputs/selfplay/curriculum-v3
```

脚本按照以下顺序启动六个彼此独立的顶层 Python 进程：round 1 Questioner、round 1 Solver、
round 2 Questioner、round 2 Solver、round 3 Questioner、round 3 Solver。不要并发执行这些命令，
因为它们复用同一个 archive、manifest 和服务端口。

每个成功阶段都会写入 `round_NNN/questioner_update/phase_manifest.json` 或
`round_NNN/solver_update/phase_manifest.json`。每条命令启动时都会扫描给定的输出目录，并按实际
checkpoint 和阶段 manifest 判断进度：

- 已完成的阶段自动 no-op，不会重新训练；原脚本可以安全地整体重跑。
- 只有 Questioner 目录而没有完整 Solver，表示该 round 尚未完成；下一步运行同轮 Solver。
- Solver checkpoint 已完成但 round manifest 尚未写入时，会把该轮恢复为已完成并进入下一轮。
- 未完成 checkpoint 不会被误判为成功；只有完整 adapter 和已达到 `max_steps` 的旧 checkpoint
  才能作为兼容恢复依据。

所以某条命令中断后，可以直接重新执行整个脚本，也可以复制脚本、删除已经成功的行后继续。单独
补跑阶段的命令格式为：

```bash
python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay_curriculum_v3.yaml \
  --output-dir outputs/selfplay/curriculum-v3 \
  --round-index 2 --phase solver
```

若只需要按 round 隔离而不需要拆分 Questioner/Solver，也可以首轮使用 `--one-round`，之后重复
`--resume --one-round`。恢复时仍以输出目录中的完整阶段产物为准，而不只依赖顶层
`manifest.json`。

`curriculum_v3` 将 Questioner 和 Solver adapter 分开，并按 production → grounding → frontier
逐步增加 reward 难度。详细设计和 smoke 标准见
[Curriculum v3](SELFPLAY_CURRICULUM_V3.md)。原始基线仍可分别使用
`selfplay.yaml` 和 `selfplay_frontier_v2.yaml`。

## 6. 可选 Solver-only GRPO warm-up

小模型或 SFT Solver 的 parse/execution 仍不稳定时，可在 self-play 前做一次 Solver-only
warm-up。先导出 train RL 数据；上节的 RL val 可直接复用：

```bash
python -m graphtask_r1.cli data export-rl \
  --input "$SFT_DATA_DIR/tasks/train.parquet" \
  --output "$KQAPRO_TRAINING/kqapro_graphscript_v03_grpo_train.parquet" \
  --roles solver --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog "$SFT_DATA_DIR/relation_catalog.json" --seed 42

export MS_SWIFT_SFT_ADAPTER=$PWD/outputs/sft/qwen3-4b-kqapro-v03/checkpoint-last
export SOLVER_RL_TRAIN_DATA="$KQAPRO_TRAINING/kqapro_graphscript_v03_grpo_train.parquet"
export SOLVER_RL_VAL_DATA="$KQAPRO_TRAINING/kqapro_graphscript_v03_solver_rl_val.parquet"
export SOLVER_GRPO_OUTPUT_DIR=$PWD/outputs/grpo/qwen3-4b-kqapro-v03
export NUM_GPUS=1
export VLLM_MODE=colocate
export ROLLOUT_N=2
export MAX_COMPLETION_LENGTH=4096

python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo_ms_swift_cuda124.yaml \
  --dry-run

python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo_ms_swift_cuda124.yaml
```

直接调用 `scripts/train_ms_swift_grpo.sh` 时也接受上述长变量名，并自动映射到
`LORA_ADAPTER_PATH/TRAIN_DATA/VAL_DATA/OUTPUT_DIR`。warm-up 完成后，把 self-play 的
`INITIAL_ADAPTER` 指向 GRPO checkpoint。

### GRPO batch 设置

```text
train batch      = NUM_GPUS × MICRO_BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS
generation batch = NUM_GPUS × MICRO_BATCH_SIZE × STEPS_PER_GENERATION
prompt count     = generation batch ÷ ROLLOUT_N
```

generation batch 和 eval batch 都必须能被 `ROLLOUT_N` 整除。先用单 GPU、少量数据确认
completion、reward variance 和非零 gradient，再放大训练。

## 7. 评测与产物

候选 checkpoint 始终使用同一份 KQAPro val，至少记录 answer EM/F1、GraphScript parse rate、
execution rate 和各 rejection/reward component。不要用 val 调 prompt、生成训练样本或回填
archive。

部署、单模型评测、多个 checkpoint 对比和路径 HTML 见
[KQAPro 模型评测与可视化](KQAPRO_EVAL_VIS_README.md)。

Qwen3-8B 可直接改用仓库中的
`qwen3_8b_sft_ms_swift_cuda124.yaml` 和
`qwen3_8b_solver_grpo_ms_swift_cuda124.yaml`；先 `--dry-run`，再按显存调整 batch。

发布或保存实验代码前运行：

```bash
make lint
make typecheck
make test
```
