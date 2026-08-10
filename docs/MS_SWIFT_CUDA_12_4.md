# ms-swift：Python 3.10 + CUDA 12.4 训练指南

本指南使用 `ms-swift==3.6.4`。这是 ms-swift 官方 CUDA 12.4 / Python 3.10 / Torch 2.6
镜像使用的版本。SFT 路径只依赖 PyTorch/Transformers/PEFT 等普通训练组件，
不安装 verl、Ray、SGLang、vLLM 或 FlashAttention；Solver GRPO 的 vLLM 是可选第二阶段。

## 1. 固定版本

| 组件 | 版本 |
| --- | --- |
| Python | `3.10` |
| PyTorch | `2.6.0+cu124` |
| torchvision | `0.21.0+cu124` |
| torchaudio | `2.6.0+cu124` |
| ms-swift | `3.6.4` |
| Attention（SFT/trainer） | PyTorch `sdpa` |
| vLLM（仅 GRPO） | `0.8.5.post1` |

ms-swift 3.6.4 将 Transformers 固定在 `<4.53`、PEFT 固定在 `<0.16`、TRL 固定在 `<0.20`，
与官方 cu124 镜像的 Torch 2.6 / vLLM 0.8.5 组合一致。先固定官方 cu124 wheel，再装
ms-swift，避免 pip 选择其他 CUDA runtime。

## 2. 建立 SFT 最小环境

```bash
nvidia-smi
nvcc --version

conda create -n graphtask-swift-cu124 python=3.10 -y
conda activate graphtask-swift-cu124
python -m pip install --upgrade pip setuptools wheel packaging

python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install "ms-swift==3.6.4"
python -m pip install -r requirements.txt
```

本仓库脚本显式使用 `--attn_impl sdpa`，因此 SFT 不需要编译或下载 FlashAttention。也不要在
该环境中安装 verl；两者的 Transformers、PEFT、TRL 和 rollout 约束不同。

## 3. 验证

```bash
python - <<'PY'
import torch
import swift
import transformers
import peft

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("ms-swift:", swift.__version__)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)

assert torch.__version__.startswith("2.6.0")
assert torch.version.cuda == "12.4"
assert torch.cuda.is_available()
assert swift.__version__ == "3.6.4"
PY

swift sft --help >/dev/null
python -m pip check
```

`nvidia-smi` 的 CUDA Version 表示驱动支持上限；关键验证项是
`torch.version.cuda == "12.4"` 和 `torch.cuda.is_available()`。

## 4. 直接读取现有 Parquet

```bash
export GRAPHTASK_KQAPRO_DB=$PWD/data/processed/kqapro/kqapro-v1/graph.sqlite
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

`graphtask_r1/training/ms_swift_plugin.py` 注册两个本地 dataset alias；加载器直接打开
`SFT_TRAIN_DATA` / `SFT_VAL_DATA`，在内存中做字段适配。它不会调用仓库的数据生成 CLI，也
不会写回输入 Parquet。

## 5. 可选 GRPO 环境

只有 Solver GRPO 才安装 vLLM：

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

再次检查 Torch 仍为 cu124。不要为了 GRPO 安装 SGLang 或 Ray；ms-swift 的 server 模式只需
vLLM。启动顺序、GPU 隔离和训练命令见根目录 [README](../README.md) 第 4 节。

ms-swift 3.6.4 + Torch 2.6 cu124 + vLLM 0.8.5.post1 是官方镜像组合，但本项目的自定义图工具
rollout 尚未在 GPU CI 中做端到端验证，
所以先使用 `NUM_GPUS=1`、`ROLLOUT_N=2`、较短 `MAX_COMPLETION_LENGTH` 和少量训练步进行
smoke test。SFT 路径不受这项 vLLM 边界影响。

## 6. 常见问题

### `ndarray is not JSON serializable`

确认使用的是 `qwen3_4b_sft_ms_swift_cuda124.yaml`，且日志显示导入
`graphtask_r1/training/ms_swift_plugin.py`。该插件会递归规范化 ndarray；不要重新生成 KQA 或
重新导出 Parquet。

### CUDA OOM

SFT 先设：

```bash
export NUM_GPUS=1
export MICRO_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=8
export MAX_LENGTH=2048
```

GRPO 依次降低 `ROLLOUT_N`、`MAX_COMPLETION_LENGTH`、`MICRO_BATCH_SIZE` 和 rollout 的
`GPU_MEMORY_UTILIZATION`。rollout 与 trainer 不得使用同一张 GPU。

## 官方参考

- [ms-swift CUDA 12.4 官方镜像矩阵](https://github.com/modelscope/ms-swift/blob/v3.6.4/docs/source_en/GetStarted/SWIFT-installation.md)
- [ms-swift v3.6.4 release](https://github.com/modelscope/ms-swift/releases/tag/v3.6.4)
- [ms-swift 自定义数据集](https://github.com/modelscope/ms-swift/blob/v3.6.4/docs/source_en/Customization/Custom-dataset.md)
- [ms-swift agent 数据格式](https://github.com/modelscope/ms-swift/blob/v3.6.4/docs/source_en/Instruction/Agent-support.md)
- [ms-swift 自定义 reward](https://github.com/modelscope/ms-swift/blob/v3.6.4/docs/source_en/Instruction/GRPO/DeveloperGuide/reward_function.md)
- [ms-swift 多轮 GRPO](https://github.com/modelscope/ms-swift/blob/v3.6.4/docs/source_en/Instruction/GRPO/DeveloperGuide/multi_turn.md)
- [PyTorch 2.6.0 cu124 安装表](https://pytorch.org/get-started/previous-versions/)
