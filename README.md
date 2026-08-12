# GraphTask-R1

GraphTask-R1 训练模型把自然语言问题编译为可执行的 `GraphScript v0.2`，再由受限执行器产生
答案。主线训练运行时统一为 **ms-swift 3.6.4**；仓库不再维护第二套训练后端、数据字段或启动
脚本。

## 实验主线

| 阶段 | 数据 | 输出 | 目的 |
| --- | --- | --- | --- |
| SFT | KQAPro train/val | GraphScript v0.2 | 建立 question-to-code 能力 |
| GRPO / self-play | KILT train | GraphScript v0.2 | 优化可解析、可执行和答案 reward |
| 最终验证 | HotpotQA、TriviaQA、NaturalQuestions | 执行结果 | 测量开放域迁移能力 |

三个阶段使用同一个 GraphScript schema、算子表、执行器和答案归一化逻辑。最终验证集只读，不能
进入 relation catalog、seed pool、reward 调参或 self-play archive。KILT 也可以同时用于 SFT 与
GRPO；这样可作为去掉 KQAPro domain gap 的对照实验，但默认配置保留 KQAPro 冷启动。

完整数据边界和算子表见 [Code-first 数据契约](docs/CODE_SELF_PLAY_DATA_CONTRACT.md)。

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
    data/processed/kqapro/kqapro-v1/val/training_tasks.parquet \
  --scope graph \
  --output data/processed/kqapro/kqapro-v1/relation_catalog.json

for split in train val; do
  python -m graphtask_r1.cli data export-sft \
    --input data/processed/kqapro/kqapro-v1/$split/training_tasks.parquet \
    --output data/training/kqapro_graphscript_v02_sft_$split.parquet \
    --roles solver --interaction-mode graphscript --graphscript-version 0.2 \
    --relation-catalog data/processed/kqapro/kqapro-v1/relation_catalog.json
done
```

`audit --deep` 同时完成完整证书检查和轻量 training view 生成；程序执行、gold 和 trace replay 已在
`data prepare` 中完成。graph-scope catalog 固定覆盖该快照的全部 relation。

用训练时的真实模板筛出 32K 内有效样本，不启动训练：

```bash
python scripts/preflight_ms_swift_sft.py \
  --input data/training/kqapro_graphscript_v02_sft_train.parquet \
  --accepted-output outputs/preflight/accepted.parquet \
  --rejected-output outputs/preflight/rejected.parquet \
  --summary-output outputs/preflight/summary.json \
  --model Qwen/Qwen3-4B-Instruct-2507 --max-length 32768
```

确认数据后再启动 SFT；`--dry-run` 只打印实际脚本和环境变量：

```bash
export SFT_TRAIN_DATA=$PWD/outputs/preflight/accepted.parquet
export SFT_VAL_DATA=$PWD/data/training/kqapro_graphscript_v02_sft_val.parquet
export SFT_OUTPUT_DIR=$PWD/outputs/sft/qwen3-4b

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml --dry-run
```

KILT GRPO、frozen-opponent self-play 和验证命令见 [训练手册](docs/TRAINING.md)。完整原始数据准备
见 [数据准备](docs/DATA_PREPARATION.md)，GraphScript 与显式工具模式的边界见
[交互模式](docs/INTERACTION_MODES.md)。
