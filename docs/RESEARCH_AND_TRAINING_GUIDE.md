# GraphTask-R1：研究定位、数据与训练指南

## 1. 推荐研究主张

项目不把“同一个小模型扮演 Questioner 与 Solver”本身当作创新。VisPlay、SPICE 和
SPARK 都已经采用或接近这一设定。GraphTask-R1 的主张应收敛为：

> 学习一个具有特权图访问的 Questioner，使其通过自主图游走构造可执行查询程序；
> 用程序执行与反事实图干预证明问题确实需要所声明的结构；再让同一个共享 LoRA
> 策略作为受限图工具 Solver，在其能力边界任务上持续学习。

建议的三个核心贡献是：

1. **Active executable task discovery**：Questioner 不接收预采样 gold path，而是通过
   `search / inspect / execute_program` 主动探索并输出类型化程序证书。
2. **Counterfactual structural verification**：删除 filter、intersection branch、hop 或
   witness edge 后重新执行，并搜索更低成本等价程序。它验证“结构是否必要”，而不只
   验证 path 是否存在。
3. **Asymmetric tool self-play**：Questioner 可以看局部 schema、执行候选程序但看不到
   benchmark 测试题；Solver 只能看到问题、topic entities 和有限 Search 工具，看不到
   gold program/answer。两者共享一个 backbone 和一个 LoRA。

### 与最邻近工作的边界

| 工作 | 已有核心 | 本项目不能重复声称 | 可验证差异 |
|---|---|---|---|
| Graph-R1 | 多轮 GraphRAG 工具交互 + RL | 图上多轮搜索 | 训练任务本身由主动 Questioner 发现并带程序证书 |
| Search-on-Graph-R1 | 8B Solver；gold SPARQL 脚手架 SFT + RL | 小模型学习 Search | 不依赖人工 gold SPARQL 生成每条训练轨迹 |
| GraphWalker | 随机游走合成轨迹 + 分阶段 SFT/RL | path 合成课程 | 主动程序构造、Solver-adaptive frontier、反事实必要性 |
| VisPlay / SPICE | 单策略双角色、frontier self-play | 共享模型自博弈 | 图工具信息不对称与确定性程序/图验证 |
| SPARK | 单个小模型在 KG path 上 Proposer/Solver | KG path self-play | Solver 真正访问图；程序而非预采样 path；必要性与 shortcut |

相关原始资料：

- Graph-R1: https://arxiv.org/abs/2507.21892
- VisPlay: https://arxiv.org/abs/2511.15661
- SPICE: https://arxiv.org/abs/2510.24684
- SPARK: https://arxiv.org/abs/2605.05546
- Search-on-Graph-R1: https://arxiv.org/abs/2607.18481
- GraphWalker: https://arxiv.org/abs/2603.28533
- verl agentic RL: https://github.com/verl-project/verl/blob/main/docs/start/agentic_rl.rst
- verl multi-turn tools: https://github.com/verl-project/verl/blob/main/docs/sglang_multiturn/multiturn.rst

以上论文截至 2026-08-04 的检索结果中，SPARK 是最需要直接对比的工作。论文实验必须
包含 `path-only proposer`、`no necessity`、`closed-book solver` 三个消融，否则差异很难成立。

## 2. 代码结构与数据流

```text
Questioner role prompt
  -> graph_search / inspect_entity / execute_program
  -> <task>{question, topic_entities, program}</task>
  -> deterministic verifier
  -> frozen Solver snapshot rollout K times
  -> frontier + necessity + novelty reward

Solver role prompt
  -> graph_search / inspect_entity
  -> <answer>[entity ids]</answer>
  -> executable gold answer F1

Questioner batch + Solver batch
  -> role-wise GRPO advantage normalization
  -> one shared backbone + one shared LoRA update
```

仓库中的 ToyGraph 路径可离线检查 DSL、verifier、工具和 Parquet 契约；KQA Pro SQLite 和
Freebase Virtuoso 路径负责真实数据。`verl_tools.py` 实现有生命周期的 per-trajectory tool
session；异步 reward 通过冻结 Solver 服务计算候选级 frontier；round orchestrator 管理
共享 adapter、archive、mixed-role 数据和恢复。

## 3. 模型选择

固定首轮模型为 `Qwen/Qwen3-4B-Instruct-2507 + LoRA rank 32`。该 checkpoint 仅使用
non-thinking 模式，避免多轮工具历史中的 reasoning 清理问题；工具格式采用 Hermes 风格。

建议从 4×A100/H100 开始：TP=1，每卡 rollout worker，SGLang async rollout。显存不足时
先降低 `TRAIN_BATCH_SIZE`、`MAX_RESPONSE_LENGTH` 和 `ROLLOUT_N`，再启用 parameter / optimizer
offload。不要先降低图验证强度。

## 4. 服务器环境

CUDA 12.8 profile 将 verl 固定为 `v0.7.1` /
`bec9ef74768dd201881cd4e54cd0385e87caae27`。最高只支持 CUDA 12.4 的服务器使用仓库内的
Python 3.10 / verl v0.5.0 profile 和[独立 CUDA 12.4 兼容环境](CUDA_12_4_ENVIRONMENT.md)，
不要在默认环境中原地降级 Torch。CUDA 12.4 当前支持 SFT 和 Solver-only GRPO，自动
self-play 仍要求 CUDA 12.8：

PyTorch、verl、SGLang、Ray、FlashAttention 与 CUDA 栈由服务器环境独立安装和维护。verl
源码可位于服务器任意位置，不需要放入本仓库的 `third_party/`；当前 Python 环境能够
`import verl` 即可。

GraphTask-R1 采用 clone 后直接运行的根目录包布局，不需要安装本项目。服务器缺少轻量运行依赖
时才需要在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -c "import torch, verl, sglang; print(torch.__version__, verl.__file__, sglang.__file__)"
```

先运行该 commit 自带的 Qwen3-4B multi-turn 示例，确认 CUDA、Ray、SGLang、tool calling
和 LoRA 能协同，再运行本项目。升级 verl 时必须重新校验 Hydra 与数据 schema。

## 5. 数据下载

### 5.1 KQA Pro：开发与 Questioner 冷启动

KQA Pro 含约 12 万问题、KoPL 程序与 SPARQL，适合先做程序映射和语言化 SFT。官方论文与
代码：https://arxiv.org/abs/2007.03875 和 https://github.com/shijx12/KQAPro_Baselines 。

```bash
pip install huggingface_hub
./scripts/download_data.sh kqapro
```

第一轮只抽一个按 entity/relation 隔离的 mini split。不要让 dev/test 问题文本或逻辑形式
进入 Questioner 的候选池。

### 5.2 Freebase + WebQSP / CWQ / GrailQA：主实验

- WebQSP 提供问题与 Freebase SPARQL；Freebase 设置说明可从 GrailQA 资源页进入。
- ComplexWebQuestions v1.1： https://www.tau-nlp.sites.tau.ac.il/compwebq
- GrailQA： https://dki-lab.github.io/GrailQA/ ，含 64,331 个问题及 IID、compositional、
  zero-shot 划分。

```bash
./scripts/download_data.sh webqsp
./scripts/download_data.sh cwq
./scripts/download_data.sh grailqa
```

Freebase dump 很大，建议使用本地 Virtuoso；将 snapshot/version hash 写入每个样本的
`extra_info.graph_snapshot`。下载后记录原始 URL、许可、SHA-256 和日期。仓库不会自动
接受第三方镜像或绕过数据许可。

## 6. 分阶段训练

### Stage A：确定性证书管线

```bash
python -m graphtask_r1.cli e2e mini-pipeline \
  --graph toy --num-programs 1000 --seed 42 --output-dir outputs/toy
```

检查 `replay_accuracy=1.0`、`unrecoverable_errors=0`，并人工看被 `SHORTCUT_FOUND`、
`REDUNDANT_CONDITION`、`ANSWER_LEAK` 拒绝的样本。

### Stage B：双角色 SFT 冷启动

从 KQA Pro 程序构造两类数据：

- Questioner：局部图状态与工具历史 -> `<task>...</task>`；
- Solver：问题与逐步工具观察 -> 下一次 tool call 或 `<answer>...</answer>`。

两类样本混在同一 SFT dataset 中，使用一个模型和一个 adapter。role prompt 与工具权限不同，
权重对象不能复制。先让格式有效率与 canonical trace replay 稳定，再进入 RL。

### Stage C：Solver-only GRPO

先把 `--roles solver` 导出并训练，验证真实 Search 工具闭环：

```bash
python -m graphtask_r1.cli data export-verl \
  --input outputs/toy/tasks.parquet \
  --output outputs/verl/solver.parquet --roles solver

MODEL_PATH=/models/Qwen3-4B-Instruct-2507 \
TRAIN_DATA=$PWD/outputs/verl/solver.parquet \
NUM_GPUS=4 EXPERIMENT_NAME=solver-grpo \
bash scripts/train_verl.sh
```

### Stage D：共享 LoRA self-play

每轮保存冻结 opponent snapshot。当前策略生成 Questioner candidates，经 verifier 后由上轮
snapshot 以 Solver 角色采样 K 次；把 `opponent_pass_rate` 写回 Questioner reward context。
Solver batch 混合 base/archive/new 任务。随后把两个角色的 prompts 合并为一个 Parquet，
在一次 GRPO job 中更新同一个 LoRA：

```bash
python -m graphtask_r1.cli data export-verl \
  --input outputs/toy/tasks.parquet \
  --output outputs/verl/mixed.parquet --roles both

MODEL_PATH=/checkpoints/shared_sft \
TRAIN_DATA=$PWD/outputs/verl/mixed.parquet \
ROLLOUT_N=8 LR=2e-6 LORA_RANK=32 NUM_GPUS=4 \
EXPERIMENT_NAME=selfplay-round-01 \
bash scripts/train_verl.sh
```

当前实现由异步 reward 调用独立冻结 Solver 服务，对每个有效候选执行 K 次真实 graph-tool
rollout；不存在固定 `opponent_pass_rate`。外层 orchestrator 使用 2 卡 actor + 2 卡 frozen
opponent，管理 snapshot、archive 混样和下一轮 verl job。

## 7. 建议首轮超参数

| 参数 | 初值 |
|---|---:|
| backbone | Qwen3-4B-Instruct-2507 |
| LoRA rank / alpha | 32 / 64 |
| rollout N | 8 |
| learning rate | 2e-6 |
| max turns | 8 |
| target frontier pass rate | 0.5 |
| Questioner / Solver loss weight | 0.35 / 0.65 |
| base / archive / new | 0.35 / 0.35 / 0.30 |
| max program cost | 4.5 |

Questioner 与 Solver 的 advantage 必须分角色标准化；若直接对混合 batch 标准化，两个 reward
尺度会相互污染。若当前 verl commit 没有 role-group normalization hook，第一版可采用交替
optimizer step，但继续写入同一个 adapter，并控制两角色 token/step 预算。

## 8. 必做实验与停止条件

最小主表至少包含：random walk、path proposer、active program、active program + necessity、
full self-play。必做消融：无 necessity、无 shortcut、闭卷 Solver、独立 LoRA、无 archive、
固定 hop curriculum。

扩展到 Freebase 长训练前必须满足：

- accepted trace replay 100%；
- Questioner executable rate 稳定；
- necessity-filtered 数据的 shortcut rate 显著更低；
- 相同 token/rollout 预算下，至少一个 held-out compositional bucket 优于 random-walk；
- 外部冻结 Solver 仍能回答，排除共享模型私有协议；
- Questioner/Solver 格式有效率不连续两轮下降。
