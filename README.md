# GraphTask-R1

GraphTask-R1 训练模型把自然语言问题编译为可执行程序，再由受限执行器产生
答案。主线训练运行时统一为 **ms-swift 3.6.4**；仓库不再维护第二套训练后端、数据字段或启动
脚本。

## 实验主线

| 阶段 | 数据 | 输出 | 目的 |
| --- | --- | --- | --- |
| SFT | KQAPro train | GraphScript v0.3 | 建立完整 KoPL-to-code 能力 |
| GRPO / self-play | KQAPro train | GraphScript v0.3 | 提高 Questioner 与 Solver 的推理能力 |
| 模型选择 | KQAPro val | 执行结果 | 只用于评测与 checkpoint 选择 |

KQAPro 三个阶段使用同一个 v0.3 算子表与执行器；`val` 的 question、program 和 answer 只读，
不能进入训练数据、reward 或 self-play archive。Questioner 仍可从共享 KQAPro 图采样任意实体，
包括恰好也出现在 val 中的实体。KILT/OpenQA 保留为独立的 v0.2 passage-search 路线，暂不混入
KQAPro 的 SFT、GRPO、relation catalog 或 checkpoint。

从原始数据到 SFT、GRPO、self-play 与 val 选模的命令见
[KQAPro 训练流程](docs/KQAPRO_TRAINING.md)。完整数据边界和算子表见
[Code-first 数据契约](docs/CODE_SELF_PLAY_DATA_CONTRACT.md)。

## 目录约定

```text
data/
├── raw/                         # 官方原始文件
├── processed/                   # 图快照、TaskCertificate、trace、rejection
├── training/                    # SFT/GRPO Parquet
└── cache/                       # 外部图查询缓存

outputs/
├── preflight/                   # 长度与 schema 预检
├── sft/
├── grpo/
└── selfplay/
```

Windows 的 `G:\datasets\GraphTaskDataset` 在 WSL 中应为
`/mnt/g/datasets/GraphTaskDataset`。若 `/mnt/g` 不存在，先检查盘符并挂载：

```bash
ls /mnt
sudo mkdir -p /mnt/g
sudo mount -t drvfs G: /mnt/g
ls /mnt/g/datasets/GraphTaskDataset
```

原始数据、图数据库、Parquet、模型权重和训练输出均不提交到 Git。

## 环境

CUDA 12.4 推荐 Python 3.10、PyTorch 2.6.0+cu124 和 ms-swift 3.6.4。安装与验证命令见
[ms-swift CUDA 12.4 环境](docs/MS_SWIFT_CUDA_12_4.md)。

只做 CPU 开发检查时：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
export PYTHONPATH=$PWD
make lint
make typecheck
make test
```

## 最短可执行路径

从已生成的 KQAPro task 导出 train/val SFT：

```bash
mkdir -p data/training

for split in train val; do
  python -m graphtask_r1.cli data audit \
    --input data/processed/kqapro/kqapro-v1/$split/tasks.parquet \
    --kind task --deep \
    --training-view-output \
      data/processed/kqapro/kqapro-v1/$split/training_tasks.parquet
done

python -m graphtask_r1.cli data build-relation-catalog \
  --input \
    data/processed/kqapro/kqapro-v1/train/training_tasks.parquet \
  --scope graph \
  --output data/processed/kqapro/kqapro-v1/relation_catalog.json

for split in train val; do
  python -m graphtask_r1.cli data export-sft \
    --input data/processed/kqapro/kqapro-v1/$split/training_tasks.parquet \
    --output data/training/kqapro_graphscript_v03_sft_$split.parquet \
    --roles solver --interaction-mode graphscript --graphscript-version 0.3 \
    --relation-catalog data/processed/kqapro/kqapro-v1/relation_catalog.json
done
```

`audit --deep` 同时完成完整证书检查和轻量 training view 生成；程序执行、gold 和 trace replay 已在
`data prepare` 中完成。graph-scope catalog 固定覆盖该快照的全部 relation。

用训练时的真实模板筛出 32K 内有效样本，不启动训练：

```bash
python scripts/preflight_ms_swift_sft.py \
  --input data/training/kqapro_graphscript_v03_sft_train.parquet \
  --accepted-output outputs/preflight/accepted.parquet \
  --rejected-output outputs/preflight/rejected.parquet \
  --summary-output outputs/preflight/summary.json \
  --model Qwen/Qwen3-4B-Instruct-2507 --max-length 32768
```

确认数据后再启动 SFT；`--dry-run` 只打印实际脚本和环境变量：

```bash
export SFT_TRAIN_DATA=$PWD/outputs/preflight/accepted.parquet
export SFT_VAL_DATA=$PWD/data/training/kqapro_graphscript_v03_sft_val.parquet
export SFT_OUTPUT_DIR=$PWD/outputs/sft/qwen3-4b

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml --dry-run
```

KQAPro GRPO、frozen-opponent self-play 和 val 验证命令见 [训练手册](docs/TRAINING.md)。完整原始数据准备
见 [数据准备](docs/DATA_PREPARATION.md)，GraphScript 与显式工具模式的边界见
[交互模式](docs/INTERACTION_MODES.md)。
