# CUDA 12.4 环境安装指南

> 新安装优先使用 [ms-swift CUDA 12.4 指南](MS_SWIFT_CUDA_12_4.md)：SFT 不需要 verl、
> Ray、SGLang、vLLM 或 FlashAttention。本页保留给仍需复用原 verl v0.5 环境的服务器。

本指南面向服务器驱动最高只支持 CUDA 12.4 的情况。不要在该环境中安装本项目默认的
verl `v0.7.1` 依赖栈：该版本的官方安装文档要求 CUDA 12.8，安装脚本也使用 Torch 2.8、
SGLang 0.5.2、vLLM 0.11.0 和面向 Torch 2.8 的 FlashAttention wheel。

CUDA 12.4 使用下面这组旧版兼容环境。它来自 verl `v0.5.0` 中保留的官方 CUDA 12.4
Dockerfile 和安装脚本，而不是把当前依赖逐个降级后混装。

## 1. 固定版本

| 组件 | CUDA 12.4 环境 |
| --- | --- |
| Python | `3.10` |
| CUDA Toolkit/runtime | `12.4` |
| cuDNN | `9.8.0`；仅 FSDP 时不需要单独安装系统 cuDNN |
| PyTorch | `2.6.0+cu124` |
| torchvision | `0.21.0+cu124` |
| torchaudio | `2.6.0+cu124` |
| torchdata | `0.11.0` |
| verl | `v0.5.0` / `8fdc4d3f202f41461f4de9f42a637228e342668b` |
| SGLang | `0.4.6.post5` |
| vLLM | `0.8.5.post1` |
| FlashAttention | `2.7.4.post1`，Python 3.10 wheel |
| FlashInfer | `0.2.2.post1+cu124torch2.6` |
| TensorDict | `0.6.2` |
| Transformers | `>=4.51.0,<4.52.0` |
| Ray | `>=2.41.0,<2.45.0` |

PyTorch 官方仍提供 Torch 2.6.0 的 cu124 wheel。TorchData 官方的兼容表将 Torch 2.6.0
对应到 torchdata 0.11.0。verl 的 CUDA 12.4 Dockerfile 还特意把 Transformers 限制在
4.52 以下，避免 4.53 带来的兼容问题；不要只升级其中一个核心包。

> `nvidia-smi` 顶部的 `CUDA Version` 是驱动支持的最高 CUDA 版本，不一定是当前机器安装的
> Toolkit。先同时检查 `nvidia-smi` 和 `nvcc --version`。如果前者最高为 12.4，就继续使用
> cu124 wheel，不要安装 cu126/cu128 wheel。

## 2. 新建独立 Conda 环境

不要在现有 verl v0.7.1 环境上原地降级。下面的命令假设当前位于 GraphTask-R1 仓库根目录：

```bash
nvidia-smi
nvcc --version

conda create -n graphtask-cu124 python=3.10 -y
conda activate graphtask-cu124
python -m pip install --upgrade pip setuptools wheel packaging ninja
```

先从 PyTorch 官方 cu124 index 安装完整的 Torch 2.6 组合：

```bash
python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install torchdata==0.11.0
```

## 3. 安装 CUDA 12.4 对应的 rollout 栈

先安装 SGLang，再安装 vLLM，最后重新固定 FlashInfer。这个顺序与 verl 的 CUDA 12.4
Dockerfile 一致，可避免 vLLM 安装过程留下错误的 FlashInfer 版本。

```bash
python -m pip install --no-cache-dir \
  "sglang[all]==0.4.6.post5" \
  --find-links https://flashinfer.ai/whl/cu124/torch2.6/flashinfer-python

python -m pip install --no-cache-dir \
  vllm==0.8.5.post1 \
  tensordict==0.6.2 \
  "transformers[hf_xet]>=4.51.0,<4.52.0" \
  "ray[default]>=2.41.0,<2.45.0" \
  torch-memory-saver

python -m pip install --no-cache-dir \
  accelerate codetiming "datasets<4" dill "hydra-core<1.4" \
  "numpy<2" "pandas<3" "peft<0.16" "pyarrow>=19,<22" \
  pybind11 pylatexenc wandb
```

安装官方预编译 wheel，避免在服务器上重新编译 FlashAttention：

```bash
python -m pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

python -m pip install --no-cache-dir \
  https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl
```

上述 wheel 是 Linux x86_64 + Python 3.10。ARM、其他 Python 版本或不同 CXX11 ABI 环境不能
直接使用它们。

## 4. 安装 verl v0.5.0 和本项目依赖

把 verl 放在 GraphTask-R1 仓库之外：

```bash
git clone --depth 1 --branch v0.5.0 \
  https://github.com/verl-project/verl.git ../verl-v0.5.0

cd ../verl-v0.5.0
test "$(git rev-parse HEAD)" = "8fdc4d3f202f41461f4de9f42a637228e342668b"
python -m pip install --no-deps -e .

cd ../GraphTask
python -m pip install -r requirements.txt
```

这里故意对 verl 使用 `--no-deps`，防止 pip 再次替换已经固定的 Torch、SGLang、vLLM 和
TensorDict。verl v0.5.0 的包元数据与其 CUDA 12.4 Dockerfile 对 TensorDict 的声明不一致；
本指南选择官方 CUDA 12.4 镜像实际使用的 `0.6.2`。因此 `pip check` 可能只报告 verl 的这条
TensorDict 元数据冲突；Torch、SGLang、vLLM、FlashAttention 或 FlashInfer 的冲突不能忽略。

## 5. 验证环境

```bash
python - <<'PY'
import torch
import torchdata
import transformers
import verl
import sglang
import vllm

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("torchdata:", torchdata.__version__)
print("transformers:", transformers.__version__)
print("sglang:", sglang.__version__)
print("vllm:", vllm.__version__)
print("verl:", verl.__file__)

assert torch.__version__.startswith("2.6.0")
assert torch.version.cuda == "12.4"
assert torch.cuda.is_available()
PY

python -m graphtask_r1.cli --help
python -m pip check
```

预期至少看到 Torch `2.6.0+cu124`、Torch CUDA runtime `12.4`、SGLang `0.4.6.post5` 和
vLLM `0.8.5.post1`。如果 `torch.cuda.is_available()` 为 `False`，先修复驱动或容器的 GPU
透传，不要开始训练。

## 6. 使用仓库内的 CUDA 12.4 profile

仓库已经为 verl v0.5.0 接入独立训练入口：

- `configs/experiments/qwen3_4b_sft_cuda124.yaml` 选择旧版
  `verl.trainer.fsdp_sft_trainer`、FSDP2、BF16 和 multi-turn messages，并通过自定义 Dataset
  把 pandas 产生的嵌套 `numpy.ndarray` 转换为 JSON 原生 list；
- `graphtask_r1.training.merge_sft` 将旧版 FSDP SFT checkpoint 导出，并把 LoRA 合并为完整
  Hugging Face 模型；
- `configs/experiments/qwen3_4b_solver_grpo_cuda124.yaml` 使用合并模型和同步 SGLang 工具
  rollout；
- RL Parquet 同时写入 verl v0.5 和 v0.7 所需的 `tools_kwargs` 位置。

完整命令见仓库根目录 [README](../README.md) 的“Python 3.10 + CUDA 12.4 训练”。不要把
CUDA 12.4 profile 与 `configs/training/verl_version.yaml` 中的 v0.7.1 profile 混用，也不要在
同一个 Conda 环境中来回升级。

verl v0.5.0 无法直接通过 `actor_rollout_ref.model.lora_adapter_path` 接入上一阶段 adapter，
所以 SFT 后必须先运行合并工具。相同原因，当前 CUDA 12.4 profile 暂不支持自动 self-play；
它支持双角色 SFT 和 Solver-only GRPO。完整训练前先验证 verl v0.5.0 自带的 SGLang
multi-turn 示例，再验证本项目的 `--dry-run`、小 batch 和一轮 smoke test。

CUDA 12.4 profile 尚未在本仓库 GPU CI 中做端到端验证；当前自动检查覆盖 Python 3.10 语法、
版本化命令选择、Parquet 工具参数契约以及 LoRA 合并辅助逻辑。

若 SFT 报错 `Object of type ndarray is not JSON serializable`，说明仍在使用未接入兼容 Dataset
的旧脚本。更新仓库后直接重启 SFT 即可，已有 KQA processed 数据和 SFT Parquet 都能继续使用。

## 参考

- [PyTorch 旧版本安装表：Torch 2.6.0 cu124](https://docs.pytorch.org/get-started/previous-versions/)
- [verl v0.7.1 安装要求：CUDA 12.8](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/docs/start/install.rst)
- [verl v0.5.0 CUDA 12.4 安装脚本](https://github.com/verl-project/verl/blob/v0.5.0/scripts/install_vllm_sglang_mcore.sh)
- [verl CUDA 12.4 镜像版本说明](https://github.com/verl-project/verl/blob/v0.5.0/docker/verl0.4-cu124-torch2.6-fa2.7.4/README.md)
- [NVIDIA：`nvidia-smi` 显示驱动支持的最高 CUDA 版本](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html)
