# GraphTask-R1：ms-swift CUDA 12.4 训练

这份 README 面向以下固定环境：

| 组件 | 版本 |
| --- | --- |
| Python | `3.10` |
| CUDA runtime | `12.4` |
| PyTorch | `2.6.0+cu124` |
| ms-swift | `3.6.4` |
| vLLM | `0.8.5.post1`，仅 Solver GRPO 需要 |

该组合对应 ms-swift 官方 CUDA 12.4 / Python 3.10 / Torch 2.6 镜像。详细版本依据见
[环境指南](docs/MS_SWIFT_CUDA_12_4.md)。

## 1. 数据不会重新生成

本流程直接读取已经生成的文件：

```bash
ls -lh \
  data/processed/kqapro/kqapro-v1/graph.sqlite \
  data/verl/kqapro_sft_train.parquet \
  data/verl/kqapro_sft_val.parquet \
  data/verl/kqapro_solver_rl.parquet \
  data/verl/kqapro_solver_rl_val.parquet
```

不要执行以下命令：

```text
data fetch
data prepare
data export-sft
data export-verl
--rebuild-graph
```

`graphtask_r1/training/ms_swift_plugin.py` 会在加载时转换字段。输入 Parquet 和
`data/processed/` 不会被写入；Hugging Face `datasets` 可能生成普通 Arrow cache，但这不是
重新处理或重新导出 KQA Pro。

## 2. 创建 SFT 环境

不要在现有 verl 环境中安装 ms-swift：

```bash
conda create -n graphtask-swift-cu124 python=3.10 -y
conda activate graphtask-swift-cu124
python -m pip install --upgrade pip setuptools wheel packaging

python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install "ms-swift==3.6.4"
python -m pip install -r requirements.txt
```

只运行 SFT 时，不安装 verl、Ray、SGLang、vLLM 或 FlashAttention。训练脚本使用 PyTorch
SDPA。

验证环境：

```bash
python - <<'PY'
import torch
import swift

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("ms-swift:", swift.__version__)

assert torch.__version__.startswith("2.6.0")
assert torch.version.cuda == "12.4"
assert torch.cuda.is_available()
assert swift.__version__ == "3.6.4"
PY

python -m pip check
```

## 3. 双角色 SFT

```bash
conda activate graphtask-swift-cu124
cd /path/to/GraphTask

export GRAPHTASK_KQAPRO_DB=$PWD/data/processed/kqapro/kqapro-v1/graph.sqlite
export SFT_TRAIN_DATA=$PWD/data/verl/kqapro_sft_train.parquet
export SFT_VAL_DATA=$PWD/data/verl/kqapro_sft_val.parquet
export SFT_OUTPUT_DIR=$PWD/outputs/ms-swift-sft-qwen3-4b-cu124
export NUM_GPUS=4
```

先确认 launcher 选择正确：

```bash
python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml \
  --dry-run
```

输出中的 command 应为：

```text
bash scripts/train_ms_swift_sft.sh
```

开始训练：

```bash
python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml
```

默认设置：

- Qwen3-4B-Instruct-2507；
- LoRA rank/alpha `32/64`；
- BF16 + SDPA；
- Hermes agent template；
- micro batch size `1`，gradient accumulation `8`；
- max length `4096`。

运行时插件会：

- 把嵌套 `numpy.ndarray` 转成 JSON list；
- 把 OpenAI assistant `tool_calls` 转成 ms-swift `tool_call`；
- 把 tool messages 转成 `tool_response`；
- 注入 `graph_search`、`inspect_entity` 和 Questioner 的 `execute_program` schema。

因此遇到 `Object of type ndarray is not JSON serializable` 时，应检查是否使用了上述 config，
不要重新生成 Parquet。

## 4. 找到 SFT adapter

```bash
find "$SFT_OUTPUT_DIR" -maxdepth 2 -type d -name 'checkpoint-*' | sort -V
export MS_SWIFT_SFT_ADAPTER=/absolute/path/to/checkpoint-<最后一步>
```

ms-swift GRPO 可以直接继续训练该 adapter，不需要先合并成完整模型。

如果目前只需要 SFT，到这里即可停止。下面的 GRPO 是可选阶段。

## 5. 可选：安装 GRPO rollout 依赖

```bash
conda activate graphtask-swift-cu124
python -m pip install "vllm==0.8.5.post1"

python - <<'PY'
import torch
import vllm

print(torch.__version__, torch.version.cuda, vllm.__version__)
assert torch.__version__.startswith("2.6.0")
assert torch.version.cuda == "12.4"
PY

python -m pip check
```

不需要安装 SGLang 或 Ray。

## 6. 启动 Solver GRPO

rollout 和 trainer 不得使用同一张 GPU。下面以 4 卡服务器为例：GPU 0 运行 rollout，GPU
1/2/3 运行训练。

终端 1：

```bash
conda activate graphtask-swift-cu124
cd /path/to/GraphTask

export GRAPHTASK_KQAPRO_DB=$PWD/data/processed/kqapro/kqapro-v1/graph.sqlite
export MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
export LORA_ADAPTER_PATH=/absolute/path/to/checkpoint-<最后一步>
export ROLLOUT_CUDA_VISIBLE_DEVICES=0
export VLLM_SERVER_PORT=8000

bash scripts/rollout_ms_swift.sh
```

终端 2：

```bash
conda activate graphtask-swift-cu124
cd /path/to/GraphTask

export GRAPHTASK_KQAPRO_DB=$PWD/data/processed/kqapro/kqapro-v1/graph.sqlite
export MS_SWIFT_SFT_ADAPTER=/absolute/path/to/checkpoint-<最后一步>
export SOLVER_RL_TRAIN_DATA=$PWD/data/verl/kqapro_solver_rl.parquet
export SOLVER_RL_VAL_DATA=$PWD/data/verl/kqapro_solver_rl_val.parquet
export SOLVER_GRPO_OUTPUT_DIR=$PWD/outputs/ms-swift-solver-grpo-cu124
export TRAIN_CUDA_VISIBLE_DEVICES=1,2,3
export NUM_GPUS=3

python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo_ms_swift_cuda124.yaml \
  --dry-run

python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo_ms_swift_cuda124.yaml
```

两卡服务器使用：

```bash
export ROLLOUT_CUDA_VISIBLE_DEVICES=0
export TRAIN_CUDA_VISIBLE_DEVICES=1
export NUM_GPUS=1
```

GraphTask scheduler 为每条请求维护独立、JSON 可序列化的 session state，并在现有 KQA
SQLite 上执行有界 `graph_search` / `inspect_entity`。总 reward 交给 ms-swift；F1、EM、
precision、recall、格式错误和 rejection reason 会写入
`graphtask_reward_components` JSON 日志。

## 7. 有界 smoke test

SFT：

```bash
export NUM_GPUS=1
export MICRO_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=8
export MAX_LENGTH=2048
export EPOCHS=1
```

GRPO：

```bash
export NUM_GPUS=1
export ROLLOUT_N=2
export MAX_COMPLETION_LENGTH=512
export MICRO_BATCH_SIZE=1
export EPOCHS=1
```

确认 smoke test 能读取第一批数据、完成 forward/backward、保存 checkpoint 后，再恢复完整参数。
本仓库没有 GPU CI，因此不要跳过这一步。

## 8. 常见问题

### `ndarray is not JSON serializable`

确认 dry-run 选择的是 `scripts/train_ms_swift_sft.sh`，并检查日志是否成功导入
`graphtask_r1/training/ms_swift_plugin.py`。不要重新导出数据。

### 找不到 Parquet

只修正 `SFT_TRAIN_DATA`、`SFT_VAL_DATA`、`SOLVER_RL_TRAIN_DATA` 或
`SOLVER_RL_VAL_DATA` 的绝对路径。训练脚本会拒绝不存在的文件，不会自动生成替代文件。

### CUDA OOM

依次降低 `MAX_LENGTH`、`ROLLOUT_N`、`MAX_COMPLETION_LENGTH` 和 micro batch size。GRPO 时还
可以降低 rollout 端的 `GPU_MEMORY_UTILIZATION`。不要改装 cu126/cu128 wheel。

### 是否还能使用 verl

可以。原入口和 config 均保留，见 [verl CUDA 12.4 指南](docs/CUDA_12_4_ENVIRONMENT.md)。
verl 与 ms-swift 使用不同 Conda 环境。
