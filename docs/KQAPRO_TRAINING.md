# KQAPro 训练流程

这是一份从原始 KQAPro 数据到训练的完整入口。默认主线针对 Qwen3-4B，使用约 5,000 条
certified Solver 源任务生成约 10,000 条 Solver/Questioner 1:1 混合 SFT，再逐轮运行和评测
self-play：

```text
5k certified train → 约 10k mixed SFT → 4B SFT → 逐轮 self-play → val 晋级
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

smoke 通过后准备正式数据。下面从 train 分层抽取 5,000 条并处理完整 val；若要完整 train，
把 `--train-sample-size` 改为 `0`。

```bash
python -m graphtask_r1.cli data prepare \
  --dataset kqapro \
  --raw-dir "$KQAPRO_RAW" \
  --output-dir "$KQAPRO_DIR" \
  --splits train,val --train-sample-size 5000 \
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

这里的 5,000 是进入认证转换的 Solver 源任务目标数，不是最终 mixed SFT 行数。少量源任务可能
因认证失败被拒绝；后续按 1:1 导出 Questioner 后，最终 mixed SFT 通常接近 10,000 行。不要通过
重复行补足整数规模。

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
export SELFPLAY_SEED_COUNT=4096

bash scripts/prepare_mixed_sft_data.sh
```

默认保留全部通过认证的 Solver，并生成相同数量的 Questioner，因此本主线最终约为 5k Solver +
5k Questioner。如果只想要固定数量的 Questioner 行，可额外设置
`QUESTIONER_COUNT_OVERRIDE=2048`。脚本不会重复 SFT 行；某个角色数据不足时会同步下采样另一
角色以保持比例。

脚本最后生成 `$SFT_DATA_DIR/sft_data.env`。训练前必须 source 它：

```bash
source "$SFT_DATA_DIR/sft_data.env"

printf 'SFT_TRAIN_DATA=%s\n' "$SFT_TRAIN_DATA"
printf 'QUESTIONER_SEEDS=%s\n' "$QUESTIONER_SEEDS"

python scripts/validate_ms_swift_data.py \
  --kind sft --input "$SFT_TRAIN_DATA"
```

环境文件包含：

| 变量 | 内容 |
| --- | --- |
| `SFT_TRAIN_DATA` | 预检通过的 mixed Solver + Questioner SFT |
| `SFT_VAL_DATA` | 预检产物；默认 SFT 训练不读取它 |
| `TRAIN_DATA` / `VAL_DATA` | shell launcher 的路径别名；默认训练只读取 `TRAIN_DATA` |
| `QUESTIONER_SEEDS` | self-play 的 Questioner seed pool |

`configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml` 只读取 `SFT_TRAIN_DATA`，并显式设置
`eval_strategy: no`。不要把训练输入手工指向 certified task 或 RL Parquet。

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

这个默认配置固定使用 2 epochs、LR `1e-5`、LoRA rank/alpha `32/64`、
`4 × 1 × 8 = 32` 的 global batch，并关闭训练期 validation。不要再额外导出 `LR` 覆盖它，
除非是在做有记录的消融实验。

直接运行 launcher 时使用 env 文件提供的 `TRAIN_DATA`；launcher 默认也是
`EVAL_STRATEGY=no`，不会读取 `VAL_DATA`：

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

self-play/GRPO 默认同样关闭训练期 validation，因此不需要导出 Solver RL val。先设置环境；
ms-swift 不保证生成 `checkpoint-last`，所以下面从 SFT 输出目录解析最近写入的真实 checkpoint，
并在训练前检查所有训练输入：

```bash
source "$SFT_DATA_DIR/sft_data.env"

export SFT_OUTPUT_DIR=$PWD/outputs/sft/qwen3-4b-kqapro-v03
SFT_ADAPTER_FILE="$(find "$SFT_OUTPUT_DIR" -type f \
  -path '*/checkpoint-*/adapter_config.json' -printf '%T@ %p\n' \
  | sort -nr | head -n 1 | cut -d ' ' -f 2-)"
test -n "$SFT_ADAPTER_FILE"
export INITIAL_ADAPTER="$(dirname "$SFT_ADAPTER_FILE")"
export BASE_TASKS="$SFT_DATA_DIR/tasks/train.parquet"
export KQAPRO_RELATION_CATALOG="$SFT_DATA_DIR/relation_catalog.json"

test -f "$INITIAL_ADAPTER/adapter_config.json"
test -f "$BASE_TASKS"
test -f "$QUESTIONER_SEEDS"
test -f "$KQAPRO_RELATION_CATALOG"

python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay_curriculum_v3.yaml \
  --output-dir outputs/selfplay/qwen3-4b-kqapro-10k \
  --dry-run

python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay_curriculum_v3.yaml \
  --output-dir outputs/selfplay/qwen3-4b-kqapro-10k \
  --one-round
```

`configs/training/selfplay_curriculum_v3.yaml` 是本 README 的默认 4B 配置，主要参数为：

| 参数 | 默认值 |
| --- | ---: |
| rounds | 3 |
| questioner_episodes / round | 1,024 |
| solver_episodes / round | 2,048 |
| opponent_samples | 4 |
| rollout_n | 4 |
| GRPO learning rate | `7e-7` |
| KL beta | `0.002` |
| base / archive / new | 0.65 / 0.10 / 0.25 |
| training-time validation | disabled |

配置内已经包含 `learning_rate`，并设置 `enable_grpo_validation: false`；self-play 不要求手动
`export LR`，也不读取 `VAL_DATA`。SFT 已经建立格式和 grounding 能力，因此三轮全部使用
frontier reward；每轮至少需要新增 128 条通过执行、难度、新颖性和目标一致性门槛的 archive
task，否则在 Solver 更新前失败，避免没有新训练信号的伪闭环。

完成第一轮并在固定 held-out eval 上晋级后，再继续下一轮：

```bash
python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay_curriculum_v3.yaml \
  --output-dir outputs/selfplay/qwen3-4b-kqapro-10k \
  --resume --one-round
```

重复一次即可完成第三轮。`--one-round` 每次只执行下一轮；不要在候选轮尚未完成评测和晋级判断时
启动下一轮。

### 可选：将三轮拆成六个独立进程

在 GPU `Exclusive Process`、或 torch distributed 在阶段切换时出现 native pointer/free 错误的
服务器上，推荐直接运行六阶段脚本：

```bash
bash scripts/run_selfplay_curriculum_phases.sh \
  configs/training/selfplay_curriculum_v3.yaml \
  outputs/selfplay/qwen3-4b-kqapro-10k
```

这个脚本适合已经决定完整跑完三轮的实验，不包含每轮之间的 held-out 晋级等待。如果采用默认的
逐轮晋级流程，应使用上面的 `--one-round` 命令。

脚本按照以下顺序启动六个彼此独立的顶层 Python 进程：round 1 Questioner、round 1 Solver、
round 2 Questioner、round 2 Solver、round 3 Questioner、round 3 Solver。不要并发执行这些命令，
因为它们复用同一个 archive、manifest 和服务端口。

脚本不会因某条命令返回非零状态而立即退出，而是记录该状态并继续尝试下一条命令；六条命令全部
尝试后，如果其中有失败，脚本整体返回非零。这样训练已经写完 checkpoint、但在 torch
distributed/native 资源清理阶段报错时，下一条命令仍会启动并从磁盘产物继续。若训练本身没有
完成，后续命令的依赖检查会拒绝跨过缺失阶段。

每个成功阶段都会写入 `round_NNN/questioner_update/phase_manifest.json` 或
`round_NNN/solver_update/phase_manifest.json`。每条命令启动时都会扫描给定的输出目录，并按实际
checkpoint 和阶段 manifest 判断进度：

- 已完成的阶段自动 no-op，不会重新训练；原脚本可以安全地整体重跑。
- 只有 Questioner 目录而没有完整 Solver，表示该 round 尚未完成；下一步运行同轮 Solver。
- Solver checkpoint 已完成但 round manifest 尚未写入时，会把该轮恢复为已完成并进入下一轮。
- 同一阶段存在 `v0`、`v1` 等多个运行目录时，先选择版本号最大的目录，再从该目录选择编号最大的
  完整 `checkpoint-xxx`；版本号和 checkpoint 编号都相同时再选择更新时间最新的一个。
- 未完成 checkpoint 不会被误判为成功；只有完整 adapter 和已达到 `max_steps` 的旧 checkpoint
  才能作为兼容恢复依据。

所以某条命令中断后，可以直接重新执行整个脚本，也可以复制脚本、删除已经成功的行后继续。单独
补跑阶段的命令格式为：

```bash
python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay_curriculum_v3.yaml \
  --output-dir outputs/selfplay/qwen3-4b-kqapro-10k \
  --round-index 2 --phase solver
```

若只需要按 round 隔离而不需要拆分 Questioner/Solver，也可以首轮使用 `--one-round`，之后重复
`--resume --one-round`。恢复时仍以输出目录中的完整阶段产物为准，而不只依赖顶层
`manifest.json`。

`curriculum_v3` 将 Questioner 和 Solver adapter 分开。本主线配置由 SFT 承担 production 和
grounding，self-play 三轮直接在 frontier 难度上训练。详细设计和 smoke 标准见
[Curriculum v3](SELFPLAY_CURRICULUM_V3.md)。原始基线仍可分别使用
`selfplay.yaml` 和 `selfplay_frontier_v2.yaml`。

## 6. 可选 Solver-only GRPO warm-up

小模型或 SFT Solver 的 parse/execution 仍不稳定时，可在 self-play 前做一次 Solver-only
warm-up。只需导出 train RL 数据：

```bash
python -m graphtask_r1.cli data export-rl \
  --input "$SFT_DATA_DIR/tasks/train.parquet" \
  --output "$KQAPRO_TRAINING/kqapro_graphscript_v03_grpo_train.parquet" \
  --roles solver --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog "$SFT_DATA_DIR/relation_catalog.json" --seed 42

export MS_SWIFT_SFT_ADAPTER="$INITIAL_ADAPTER"
export SOLVER_RL_TRAIN_DATA="$KQAPRO_TRAINING/kqapro_graphscript_v03_grpo_train.parquet"
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
`LORA_ADAPTER_PATH/TRAIN_DATA/OUTPUT_DIR`。warm-up 完成后，把 self-play 的
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

SFT 和 RL 训练期不使用 val；这里的评测是在 checkpoint 训练完成后独立运行。候选 checkpoint
始终使用同一份 KQAPro val，至少记录 answer EM/F1、GraphScript parse rate、
execution rate 和各 rejection/reward component。正式比较至少固定 1,024 条 held-out 样本；不要
用 val 调 prompt、生成训练样本或回填 archive。

SFT baseline 与某轮 self-play candidate 使用同一 input、seed、prompt 和解码设置完成评测后，
用两个 `metrics.json` 做晋级：

```bash
export SELFPLAY_OUTPUT_DIR=$PWD/outputs/selfplay/qwen3-4b-kqapro-10k
ROUND_ADAPTER_FILE="$(find "$SELFPLAY_OUTPUT_DIR/round_001/solver_update" -type f \
  -path '*/checkpoint-*/adapter_config.json' -printf '%T@ %p\n' \
  | sort -nr | head -n 1 | cut -d ' ' -f 2-)"
test -n "$ROUND_ADAPTER_FILE"
export ROUND_ADAPTER="$(dirname "$ROUND_ADAPTER_FILE")"

python -m graphtask_r1.cli evaluate kqapro-promote \
  --baseline outputs/evaluation/kqapro-sft/metrics.json \
  --candidate outputs/evaluation/kqapro-selfplay-round-001/metrics.json \
  --baseline-artifact "$INITIAL_ADAPTER" \
  --candidate-artifact "$ROUND_ADAPTER" \
  --output outputs/evaluation/promotion-round-001.json \
  --require-promotion
```

默认 gate 要求 exact match、F1 和 tool success 都不退化，且至少一项严格提升。未通过时命令返回
状态码 2，应保留上一轮已晋级 adapter，不继续下一轮。

模型服务部署、`kqapro-val`、多个 checkpoint 对比和路径 HTML 见
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
