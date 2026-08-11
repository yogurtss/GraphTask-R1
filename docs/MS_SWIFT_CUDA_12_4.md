# ms-swift：Python 3.10 + CUDA 12.4 环境

推荐固定 Python 3.10、PyTorch 2.6.0+cu124 和 `ms-swift==3.6.4`。SFT 使用 PyTorch SDPA；
GRPO 额外安装 vLLM。为避免 Transformers、PEFT、TRL 和 CUDA wheel 被其他项目改写，请创建
独立 Conda 环境。

## 1. 建立环境

```bash
nvidia-smi

conda create -n graphtask-swift-cu124 python=3.10 -y
conda activate graphtask-swift-cu124
python -m pip install --upgrade pip setuptools wheel packaging

python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install "ms-swift==3.6.4"
python -m pip install -r requirements.txt
```

只做 SFT 时不需要安装 vLLM。要运行 GRPO，再执行：

```bash
python -m pip install "vllm==0.8.5.post1" math_verify
```

ms-swift 3.6.4 的 GRPO trainer 会导入它的多轮 scheduler 注册表，即使 GraphScript 主线未启用
多轮模式，因此 GRPO 环境仍需补装 `math_verify`；SFT 和数据预检不需要它。

Self-play 的 frozen opponent 使用 SGLang，建议在独立服务环境中部署，并与 ms-swift actor GPU
隔离。普通 SFT 和 Solver-only GRPO 不依赖该服务。

## 2. 验证版本与 GPU

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

`nvidia-smi` 显示的是驱动支持上限；环境是否正确应以 `torch.version.cuda` 和实际
`torch.cuda.is_available()` 为准。

## 3. 验证本项目集成

```bash
export PYTHONPATH=$PWD

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml --dry-run

python scripts/preflight_ms_swift_sft.py --help
bash -n scripts/train_ms_swift_sft.sh
bash -n scripts/train_ms_swift_grpo.sh
bash -n scripts/rollout_ms_swift.sh
```

本地插件直接读取 `TRAIN_DATA` / `VAL_DATA` 指向的 Parquet，并在内存中适配字段；不会修改输入
文件，也不会隐式重新生成 KQAPro 或 KILT。

## 4. 常见问题

### `Please explicitly pass the model_type`

使用仓库内脚本或设置 `MODEL_TYPE=qwen3`。SFT、rollout 和 GRPO 均显式传入 model type。

### `ndarray is not JSON serializable`

确认 `--external_plugins` 指向 `graphtask_r1/training/ms_swift_plugin.py`。适配器会递归把
NumPy/Arrow 值转换为 JSON 原生类型。

### 样本从约 1 万降到约 3 千

先运行 `scripts/preflight_ms_swift_sft.py` 查看结构化 rejection。若主要原因为
`SFT_MAX_LENGTH_EXCEEDED`，把 `MAX_LENGTH` 与预检 `--max-length` 同时提高到 20000–40960；若
是 tool trace 过长，优先导出 GraphScript v0.2 单程序样本，并提高 KQAPro prepare 的
`--max-trace-tool-calls` 与 `--max-trace-query-results`。不得依赖静默截断，因为它可能删除 `emit`
或最终答案监督。

### CUDA OOM

先保持 `MICRO_BATCH_SIZE=1`，降低 max length 或 rollout generations，再提高 gradient
accumulation。正式 GRPO 前用 `NUM_GPUS=1`、`ROLLOUT_N=2`、短 completion 做 bounded smoke。

## 官方参考

- [ms-swift v3.6.4 release](https://github.com/modelscope/ms-swift/releases/tag/v3.6.4)
- [ms-swift 安装](https://github.com/modelscope/ms-swift/blob/v3.6.4/docs/source_en/GetStarted/SWIFT-installation.md)
- [自定义数据集](https://github.com/modelscope/ms-swift/blob/v3.6.4/docs/source_en/Customization/Custom-dataset.md)
- [Agent 数据格式](https://github.com/modelscope/ms-swift/blob/v3.6.4/docs/source_en/Instruction/Agent-support.md)
- [GRPO reward](https://github.com/modelscope/ms-swift/blob/v3.6.4/docs/source_en/Instruction/GRPO/DeveloperGuide/reward_function.md)
- [多轮 GRPO](https://github.com/modelscope/ms-swift/blob/v3.6.4/docs/source_en/Instruction/GRPO/DeveloperGuide/multi_turn.md)
- [PyTorch 2.6.0 cu124](https://pytorch.org/get-started/previous-versions/)
