# GraphTask-R1

GraphTask-R1 使用 Qwen3-4B + LoRA 训练 Questioner 和 Solver。当前训练主线只使用
**KQA Pro**，不需要 Freebase 或 Virtuoso；所有 gold answer 都由认证程序实际执行得到。

## 当前推荐入口

服务器为 Python 3.10 + CUDA 12.4 时，推荐使用 ms-swift：

- [ms-swift CUDA 12.4 完整训练 README](README_MS_SWIFT_CUDA124.md)
- [依赖版本与环境说明](docs/MS_SWIFT_CUDA_12_4.md)

ms-swift SFT 不依赖 verl、Ray、SGLang、vLLM 或 FlashAttention。Solver GRPO 是可选阶段，
仍需单独的 vLLM rollout 服务。

原有 verl 环境继续参考 [verl CUDA 12.4 指南](docs/CUDA_12_4_ENVIRONMENT.md)。两套后端应
使用不同 Conda 环境。

## 复用现有数据

当前训练只读取下列文件：

```text
data/processed/kqapro/kqapro-v1/graph.sqlite
data/verl/kqapro_sft_train.parquet
data/verl/kqapro_sft_val.parquet
data/verl/kqapro_solver_rl.parquet
data/verl/kqapro_solver_rl_val.parquet
```

不要运行 `data fetch`、`data prepare`、`data export-sft`、`data export-verl` 或
`--rebuild-graph`。ms-swift 插件只在内存中转换现有 Parquet，不会修改源文件或重新生成 KQA
数据。

## ms-swift SFT 快速入口

完成独立 README 中的环境安装后：

```bash
export SFT_TRAIN_DATA=$PWD/data/verl/kqapro_sft_train.parquet
export SFT_VAL_DATA=$PWD/data/verl/kqapro_sft_val.parquet
export SFT_OUTPUT_DIR=$PWD/outputs/ms-swift-sft-qwen3-4b-cu124
export NUM_GPUS=4

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml \
  --dry-run

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml
```

运行时适配器会处理已有 Parquet 中的 `numpy.ndarray`、assistant `tool_calls` 和 tool
messages，因此不需要为了修复 JSON serializable 报错而重新导出数据。

## 开发检查

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

make lint
make typecheck
make test
```

其他说明：

- [数据准备说明](docs/DATA_PREPARATION.md)：仅供没有任何 processed 数据的新服务器使用
- [训练手册](docs/TRAINING.md)：原 verl/self-play 流程
- [交互模式](docs/INTERACTION_MODES.md)：工具与 GraphScript 格式
