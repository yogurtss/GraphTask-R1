# GraphTask-R1

GraphTask-R1 用同一个 Qwen3-4B + LoRA 策略训练两个角色：Questioner 在图上构造可执行任务，
Solver 通过受限图工具回答问题。所有 gold answer 都由认证程序实际执行得到，不采用模型输出
作为 gold。

当前训练主线只使用 **KQA Pro**。不需要 Freebase、Virtuoso、WebQSP、CWQ 或 GrailQA。
本 README 以服务器上限为 **Python 3.10 + CUDA 12.4** 的环境为默认流程。

## 当前支持范围

| Profile | Python / CUDA | PyTorch / verl | SFT | Solver GRPO | 自动 self-play |
| --- | --- | --- | --- | --- | --- |
| `cuda124` | 3.10 / 12.4 | 2.6.0+cu124 / v0.5.0 | 支持 | 支持，同步 SGLang | 暂不支持 |
| `cuda128` | 3.12 / 12.8 | 2.8 / v0.7.1 | 支持 | 支持，异步 SGLang | 支持 |

CUDA 12.4 的完整安装命令和固定版本见
[CUDA 12.4 环境指南](docs/CUDA_12_4_ENVIRONMENT.md)。两个 profile 必须使用不同 Conda
环境，不能在同一环境中原地升降级。

## 1. 复用已经处理好的 KQA Pro

如果下面三个文件已经存在，**不要再次运行** `data fetch`、`data prepare` 或
`--rebuild-graph`：

```text
data/processed/kqapro/kqapro-v1/graph.sqlite
data/processed/kqapro/kqapro-v1/train/tasks.parquet
data/processed/kqapro/kqapro-v1/val/tasks.parquet
```

先检查文件并设置图路径：

```bash
ls -lh \
  data/processed/kqapro/kqapro-v1/graph.sqlite \
  data/processed/kqapro/kqapro-v1/train/tasks.parquet \
  data/processed/kqapro/kqapro-v1/val/tasks.parquet

export GRAPHTASK_KQAPRO_DB=$PWD/data/processed/kqapro/kqapro-v1/graph.sqlite
```

`tasks.parquet` 中约 9k 条 accepted records 可以直接训练。数量少于 KQA Pro 原始样本是正常
现象：转换器只保留当前 DSL 能表达、可执行并通过认证的任务。

## 2. 只导出训练 Parquet，不重新处理 KQA Pro

仅当 `data/verl/` 下对应文件缺失时运行。以下命令读取现有 accepted tasks，不会重建
`graph.sqlite`，也不会修改 `data/processed/`：

```bash
python -m graphtask_r1.cli data export-sft \
  --input data/processed/kqapro/kqapro-v1/train/tasks.parquet \
  --output data/verl/kqapro_sft_train.parquet \
  --roles both

python -m graphtask_r1.cli data export-sft \
  --input data/processed/kqapro/kqapro-v1/val/tasks.parquet \
  --output data/verl/kqapro_sft_val.parquet \
  --roles both

python -m graphtask_r1.cli data export-verl \
  --input data/processed/kqapro/kqapro-v1/train/tasks.parquet \
  --output data/verl/kqapro_solver_rl.parquet \
  --roles solver

python -m graphtask_r1.cli data export-verl \
  --input data/processed/kqapro/kqapro-v1/val/tasks.parquet \
  --output data/verl/kqapro_solver_rl_val.parquet \
  --roles solver
```

完成后应有：

```text
data/verl/kqapro_sft_train.parquet
data/verl/kqapro_sft_val.parquet
data/verl/kqapro_solver_rl.parquet
data/verl/kqapro_solver_rl_val.parquet
```

## 3. Python 3.10 + CUDA 12.4 训练

确认当前环境确实加载了 CUDA 12.4 profile：

```bash
conda activate graphtask-cu124

python - <<'PY'
import torch
import verl
import sglang

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("verl:", verl.__file__)
print("sglang:", sglang.__version__)
assert torch.__version__.startswith("2.6.0")
assert torch.version.cuda == "12.4"
assert torch.cuda.is_available()
PY
```

### 3.1 双角色 SFT

```bash
export SFT_TRAIN_DATA=$PWD/data/verl/kqapro_sft_train.parquet
export SFT_VAL_DATA=$PWD/data/verl/kqapro_sft_val.parquet
export SFT_OUTPUT_DIR=$PWD/outputs/sft-qwen3-4b-cu124
export NUM_GPUS=4

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_cuda124.yaml \
  --dry-run

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_cuda124.yaml
```

该配置自动选择 verl v0.5 的 `fsdp_sft_trainer`、FSDP2、BF16、多轮 `messages` 数据和
LoRA rank/alpha 32/64。命令行环境变量优先于 YAML；例如可以用
`NUM_GPUS=1 TRAIN_BATCH_SIZE=4 EPOCHS=1` 做有界 smoke test。

### 3.2 把 SFT LoRA 合并为完整模型

verl v0.5 的 GRPO 不能直接加载 `lora_adapter_path`。SFT 结束后，把最后一个
`global_step_*` 目录导出并合并一次：

```bash
find "$SFT_OUTPUT_DIR" -maxdepth 2 -name fsdp_config.json -printf '%h\n' | sort -V

export SFT_CHECKPOINT=$SFT_OUTPUT_DIR/global_step_<最后一步>
export CUDA124_SFT_MODEL=$PWD/outputs/sft-qwen3-4b-cu124-merged

python -m graphtask_r1.training.merge_sft \
  --checkpoint "$SFT_CHECKPOINT" \
  --output "$CUDA124_SFT_MODEL" \
  --lora-alpha 64
```

合并工具会先调用 verl v0.5 的 FSDP model merger，修复它导出的 `lora_alpha: 0`，再用
PEFT 将 LoRA 写入完整 Hugging Face 权重。`--output` 必须是尚不存在的目录；Qwen3-4B 合并
需要足够的 CPU 内存和磁盘空间。

### 3.3 Solver-only GRPO

CUDA 12.4 profile 使用同步 SGLang multi-turn rollout，这样 verl v0.5 才会把每条样本的
图快照、角色和工具 session 参数传给 GraphTask 工具：

```bash
export CUDA124_SFT_MODEL=$PWD/outputs/sft-qwen3-4b-cu124-merged
export SOLVER_RL_TRAIN_DATA=$PWD/data/verl/kqapro_solver_rl.parquet
export SOLVER_RL_VAL_DATA=$PWD/data/verl/kqapro_solver_rl_val.parquet
export SOLVER_GRPO_OUTPUT_DIR=$PWD/outputs/solver-grpo-cu124
export NUM_GPUS=4

python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo_cuda124.yaml \
  --dry-run

python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo_cuda124.yaml
```

开始长训练前建议先设置较小的 `TRAIN_BATCH_SIZE`、`ROLLOUT_N`、`MAX_RESPONSE_LENGTH` 和
`EPOCHS=1`。如果显存不足，依次降低这四项，不要改用 CUDA 12.8 wheel。

### 3.4 CUDA 12.4 的边界

当前 `train self-play` orchestrator 每轮都以“base model + 上轮 adapter”的形式冻结 opponent
并继续训练 adapter；verl v0.5 不支持这条 adapter 交接接口。因此 CUDA 12.4 目前支持到
SFT + Solver GRPO，**不要**使用 `configs/training/selfplay.yaml`。自动 self-play 仍只属于
`cuda128` profile。训练脚本会在 CUDA 12.4 下拒绝直接传入 `LORA_ADAPTER_PATH`，避免静默从
错误的 base model 开始训练。

## 4. CUDA 12.8 原有流程

CUDA 12.8 环境继续使用原配置，行为未改变：

```bash
python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft.yaml --dry-run

python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo.yaml --dry-run
```

CUDA 12.8 的 self-play 说明见 [训练手册](docs/TRAINING.md)。

## 5. 只有 processed 文件不存在时才处理 KQA Pro

这是一台全新服务器的一次性流程；已有 processed 文件时跳过：

```bash
python -m graphtask_r1.cli data fetch --dataset kqapro

python -m graphtask_r1.cli data prepare \
  --dataset kqapro \
  --raw-dir data/raw/kqa_pro \
  --output-dir data/processed/kqapro/kqapro-v1 \
  --splits train,val \
  --seed 42 \
  --workers 1
```

不要为正常续训添加 `--rebuild-graph`。KQA Pro `test.json` 缺少 gold program/answer，不进入训练。

## 6. 开发验证

仓库从根目录直接运行，不需要 `pip install -e .`。GPU 栈单独安装，轻量依赖使用：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

make lint
make typecheck
make test
```

更多文档：

- [CUDA 12.4 环境指南](docs/CUDA_12_4_ENVIRONMENT.md)
- [训练手册](docs/TRAINING.md)
- [KQA Pro 数据准备与审计](docs/DATA_PREPARATION.md)
- [工具与 GraphScript 交互模式](docs/INTERACTION_MODES.md)
- [研究与实验设计](docs/RESEARCH_AND_TRAINING_GUIDE.md)
