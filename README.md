# GraphTask-R1

GraphTask-R1 训练模型把自然语言问题编译为 GraphScript，再由有界图执行器生成答案。训练统一使用
`ms-swift==3.10.3`，主线为：

```text
KQAPro 数据准备 → Solver + Questioner SFT → self-play → KQAPro val 选模
                                      └→ 可选 Solver-only GRPO warm-up
```

KQAPro 的下载、转换、SFT 数据环境变量、训练和 self-play 命令全部集中在
[KQAPro 训练流程](docs/KQAPRO_TRAINING.md)。第一次运行从这份文档开始，不需要在 README、
`DATA_PREPARATION.md` 和训练手册之间来回查找。

默认主线是 **SFT → self-play → val 选模**；Solver-only GRPO 不是前置依赖，只在 Solver
冷启动不稳定时作为可选 warm-up。

## 核心约束

- KQAPro train 可用于 SFT、GRPO 和 self-play；val 只用于评测与 checkpoint 选择。
- gold answer 只由 certified program 执行产生。
- 默认交互是 GraphScript v0.3 单次程序生成；显式 function calling 是独立消融模式。
- 原始数据、Parquet、图数据库、模型权重和训练输出不提交到 Git。

## Self-play 版本

| 配置 | 用途 |
| --- | --- |
| `configs/training/selfplay.yaml` | 原始 legacy 基线 |
| `configs/training/selfplay_frontier_v2.yaml` | frontier_v2 对照 |
| `configs/training/selfplay_curriculum_v3.yaml` | 分阶段 dense reward 与独立 Q/S adapter |
| `configs/training/selfplay_qwen3_0_6b_curriculum_v3_smoke.yaml` | Toy/bounded 小模型 smoke |

不同版本可以从同一个 SFT adapter 开始，但应写入不同输出目录。

### 三轮 Self-play 分阶段运行

为避免 torchrun/NCCL 资源跨阶段残留，可把三轮 curriculum self-play 拆成六个独立进程：

```bash
bash scripts/run_selfplay_curriculum_phases.sh \
  configs/training/selfplay_curriculum_v3.yaml \
  outputs/selfplay/curriculum-v3
```

脚本依次运行每轮的 Questioner 和 Solver。每条命令都会扫描输出目录：已完成阶段自动跳过，
Questioner 完成后中断可从同轮 Solver 继续，Solver 完成后中断可从下一轮继续。因此可以直接重跑
整个脚本，也可以删除已经完成的命令后继续。阶段恢复和单命令用法见
[训练手册的 Self-play 章节](docs/TRAINING.md#5-self-play默认接在-sft-后)。

## 环境与代码检查

CUDA 12.4 环境见 [ms-swift 安装说明](docs/MS_SWIFT_CUDA_12_4.md)。只做 CPU 开发检查：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
export PYTHONPATH=$PWD
make lint
make typecheck
make test
```

## 文档入口

- [KQAPro：从原始数据到 SFT、GRPO、self-play](docs/KQAPRO_TRAINING.md)
- [Curriculum v3 设计与 A/B](docs/SELFPLAY_CURRICULUM_V3.md)
- [模型评测与路径可视化](docs/KQAPRO_EVAL_VIS_README.md)
- [GraphScript 与 tool 模式](docs/INTERACTION_MODES.md)
- [其他数据集与图快照](docs/DATA_PREPARATION.md)
- [Reward/zero-gradient 排查](docs/SELFPLAY_REWARD_DEBUG_README.md)

## 目录

```text
data/
├── raw/          # 官方原始文件
├── processed/    # graph.sqlite 与 certified tasks
└── training/     # SFT/GRPO/self-play Parquet

outputs/
├── sft-data/
├── sft/
├── grpo/
└── selfplay/
```
