# GraphTask-R1

GraphTask-R1 训练模型把自然语言问题编译为可执行程序，再由受限执行器产生
答案。主线训练运行时统一为 **ms-swift 3.10.3**；仓库不再维护第二套训练后端、数据字段或启动
脚本。

## 实验主线

| 阶段 | 数据 | 输出 | 目的 |
| --- | --- | --- | --- |
| SFT | KQAPro train | GraphScript v0.3 | 建立完整 KoPL-to-code 能力 |
| Self-play（内部使用 mixed-role GRPO） | KQAPro train | GraphScript v0.3 | 让共享模型的 Questioner 与 Solver 跨轮协同进化 |
| 可选 Solver-only GRPO | KQAPro train | GraphScript v0.3 | 在 self-play 前单独增强 Solver，或用于消融 |
| 模型选择 | KQAPro val | 执行结果 | 只用于评测与 checkpoint 选择 |

KQAPro 三个阶段使用同一个 v0.3 算子表与执行器；`val` 的 question、program 和 answer 只读，
不能进入训练数据、reward 或 self-play archive。Questioner 仍可从共享 KQAPro 图采样任意实体，
包括恰好也出现在 val 中的实体。KILT/OpenQA 保留为独立的 v0.2 passage-search 路线，暂不混入
KQAPro 的 SFT、GRPO、relation catalog 或 checkpoint。

默认主线是 **SFT → self-play → val 选模**；Solver-only GRPO 不是前置依赖。若 SFT Solver 的
GraphScript parse/execution/F1 尚不稳定，可把它作为可选 warm-up。完整命令见
[KQAPro 训练流程](docs/KQAPRO_TRAINING.md)。完整数据边界和算子表见
[Code-first 数据契约](docs/CODE_SELF_PLAY_DATA_CONTRACT.md)。

Self-play 使用同一个 ms-swift GRPO trainer，并通过
`configs/training/selfplay.yaml` 的 `rl_algorithm` 选择 advantage estimator：

```yaml
# 默认：batch 归一化，并把 KL 纳入 reward
rl_algorithm: reinforce_plus_plus

# 消融或兼容旧实验：组内归一化
# rl_algorithm: grpo
```

允许值只有 `reinforce_plus_plus` 和 `grpo`。切换算法不会改变 Questioner/Solver 数据或 reward
契约；默认推荐 REINFORCE++，需要复现实验或做对照时再切回 GRPO。该选项要求
`ms-swift==3.10.3`，完整参数、batch 整除关系和验证集设置见
[Self-play 训练配置](docs/KQAPRO_TRAINING.md#5-questionersolver-self-play)。

base direct、base tool、SFT、GRPO 的单模式部署/评测、任意多模式结果汇总和静态 HTML 路径
可视化见
[KQAPro 模型评测与路径可视化 README](docs/KQAPRO_EVAL_VIS_README.md)。
Qwen3-8B 的 SFT、GRPO 和单模型 eval 静态配置示例见
[KQAPro 训练流程的 8B 小节](docs/KQAPRO_TRAINING.md#21-qwen3-8b-配置示例)；配置检查可只运行
`--dry-run`，不会启动训练或下载权重。

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

CUDA 12.4 推荐 Python 3.10、PyTorch 2.6.0+cu124 和 ms-swift 3.10.3。安装与验证命令见
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

从已认证的 train/val task 一键生成混合角色 SFT。第一次使用时只需打开
`scripts/prepare_mixed_sft_data.sh`，修改开头的路径、模型和比例配置：

```bash
export TRAIN_TASKS=/path/to/train/tasks.parquet
export VAL_TASKS=/path/to/val/tasks.parquet
export WORK_DIR=$PWD/outputs/sft-data
export GRAPH_DB_PATH=/path/to/graph.sqlite
export SOLVER_RATIO=1
export QUESTIONER_RATIO=1
export MODEL_PATH=/path/to/model
```

```bash
bash scripts/prepare_mixed_sft_data.sh
source outputs/sft-data/sft_data.env
```

该脚本依次完成 deep audit、training view、relation catalog、双角色导出、按比例混合及真实训练模板
预检，并从 certified train tasks 生成 self-play seed pool。默认 `1:1` 表示 Solver 与 Questioner
训练曝光相同；若需要固定 Questioner 数量，设置 `QUESTIONER_COUNT_OVERRIDE`。SFT 只从
真实 explicit-root tasks 中按固定 seed 随机抽取唯一行，不重复；任一角色数据不足时，
脚本同步下采样另一角色，保持最终 Solver:Questioner 比例。metrics 同时记录精确 entity
数、terminal、路径长度、答案类型、operator
覆盖率、真实/导出 strata 占比及
`distribution_total_variation`。Self-play seed 则按真实结构分层，并将隐藏的
`source_stratum` 交给 reward（不进入模型 prompt），生成程序按 root、terminal、长度、operator 和
答案类型计算 `target_alignment`，使合成任务向真实 train 分布靠拢；该指标也会写入日志曲线。预检
默认要求所有行通过，以免过滤后静默改变比例。确认数据后再启动 SFT；`--dry-run` 只打印实际训练
脚本和环境变量：

```bash
export SFT_OUTPUT_DIR=$PWD/outputs/sft/qwen3-4b

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml --dry-run
```

直接从 SFT checkpoint 启动 frozen-opponent self-play、可选 Solver-only GRPO 和 val 验证命令见
[训练手册](docs/TRAINING.md)。完整原始数据准备
见 [数据准备](docs/DATA_PREPARATION.md)，GraphScript 与显式工具模式的边界见
[交互模式](docs/INTERACTION_MODES.md)。

## Self-play A/B 版本

原方法保持为默认的 `legacy`，继续使用 `configs/training/selfplay.yaml`。新的
`frontier_v2` 使用独立的 Solver/Questioner 更新、frontier-gated reward 和逐轮候选档案，
通过 `configs/training/selfplay_frontier_v2.yaml` 显式启用。两者可以从同一个初始 adapter
启动，但必须写入不同目录：

```bash
python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay-legacy

python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay_frontier_v2.yaml \
  --output-dir outputs/selfplay-frontier-v2
```

两套生产配置都关闭 trainer eval，每 20 个 optimizer steps 保存一次 checkpoint，最多保留
2 个。短流程 smoke 为保证跨轮 adapter 交接会每步保存；版本边界与对比产物见
[Frontier v2 A/B 说明](docs/SELFPLAY_FRONTIER_V2.md)。
