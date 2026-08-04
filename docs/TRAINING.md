# Qwen3-4B + verl 训练手册

项目固定 verl `v0.7.1`，commit：
`bec9ef74768dd201881cd4e54cd0385e87caae27`。默认模型是
`Qwen/Qwen3-4B-Instruct-2507`，非 thinking 模式，共享 LoRA rank/alpha 为 32/64。

## 1. 环境

```bash
conda create -n graphtask python=3.11 -y
conda activate graphtask

git clone --recursive https://github.com/verl-project/verl.git third_party/verl
cd third_party/verl
git checkout bec9ef74768dd201881cd4e54cd0385e87caae27
pip install -e .
cd ../..

pip install -e '.[dev,training]'
pip install 'sglang[all]' flash-attn --no-build-isolation
```

先运行 verl 自带的 Qwen3-4B multi-turn 示例，确认 CUDA、Ray、SGLang 和 tool-call template
正常。项目只支持上述固定提交；升级 verl 后必须重新检查 Hydra 字段和 Parquet contract。

## 2. 双角色 SFT

按数据文档生成：

```text
data/verl/kqapro_sft_train.parquet
data/verl/kqapro_sft_val.parquet
```

然后：

```bash
export SFT_TRAIN_DATA=$PWD/data/verl/kqapro_sft_train.parquet
export SFT_VAL_DATA=$PWD/data/verl/kqapro_sft_val.parquet
export SFT_OUTPUT_DIR=$PWD/outputs/sft-qwen3-4b

graphtask-r1 train sft \
  --config configs/experiments/qwen3_4b_sft.yaml --dry-run
graphtask-r1 train sft \
  --config configs/experiments/qwen3_4b_sft.yaml
```

SFT 数据使用 verl `messages` contract。Questioner 学习输出结构化 TaskProposal；Solver 学习
真实 `graph_search`/`inspect_entity` 调用和 `<answer>`。两种角色写入同一 adapter。

## 3. Solver-only GRPO

```bash
graphtask-r1 data export-verl \
  --input data/processed/kqapro/kqapro-v1/train/tasks.parquet \
  --output data/verl/kqapro_solver_rl.parquet --roles solver

export SFT_ADAPTER=/absolute/path/to/sft/lora_adapter
export SOLVER_RL_TRAIN_DATA=$PWD/data/verl/kqapro_solver_rl.parquet
export SOLVER_RL_VAL_DATA=$PWD/data/verl/kqapro_solver_rl_val.parquet
export SOLVER_GRPO_OUTPUT_DIR=$PWD/outputs/solver-grpo

graphtask-r1 train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo.yaml --dry-run
graphtask-r1 train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo.yaml
```

先确认格式有效率、工具成功率和 held-out KQA F1，再进入 Freebase self-play。

## 4. 真实 mixed-role self-play

默认 GPU 布局：

```text
GPU 0–1  verl actor/rollout/ref，共享 LoRA mixed-role GRPO
GPU 2–3  上一轮冻结 LoRA 的 SGLang Solver，DP=2、TP=1
CPU      GraphTask opponent/archive service、Virtuoso 和 reward workers
```

准备环境变量：

```bash
export INITIAL_ADAPTER=/absolute/path/to/solver_grpo_or_sft_adapter
export BASE_TASKS=$PWD/data/processed/kqapro/kqapro-v1/train/tasks.parquet
export VAL_DATA=$PWD/data/verl/kqapro_solver_rl_val.parquet
export QUESTIONER_SEEDS=$PWD/data/verl/freebase_questioner_seeds.parquet
export FREEBASE_ENDPOINT=http://127.0.0.1:8890/sparql
export GRAPHTASK_GRAPH_CACHE=$PWD/data/cache/freebase.sqlite
```

先检查完整进程计划：

```bash
graphtask-r1 train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay-qwen3-4b --dry-run
```

确认路径后启动：

```bash
graphtask-r1 train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay-qwen3-4b
```

每轮使用上轮 adapter 启动冻结 Solver。Questioner 的每个有效候选通过异步 `/evaluate`
接口执行 8 次真实工具 rollout；其 pass rate、结构必要性和 novelty 共同形成 reward。接受任务由
单写者 SQLite archive 幂等保存，并在下一轮进入 Solver batch。基础设施请求重试后仍失败会
中止该 job，不会伪装成 Solver 答错。

恢复：

```bash
graphtask-r1 train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay-qwen3-4b --resume
```

resume 会核对 config hash，并从最后完成 round 的 adapter 继续。不要修改已有 round 目录。

## 5. Benchmark 评测

用待评测 adapter 启动冻结 Solver SGLang 和 GraphTask service，命令形态与每轮 `plan.json`
中的 `sglang`、`opponent` 一致。服务健康后：

```bash
graphtask-r1 evaluate benchmark \
  --input data/processed/grailqa/dev/examples.parquet \
  --output-dir outputs/eval/grailqa-dev \
  --solver-url http://127.0.0.1:18080 \
  --snapshot freebase-v1 --samples 1
```

输出包括 entity F1、exact match、工具调用数、延迟、逐 split 指标和逐样本 predictions。
论文主表至少比较 base/SFT、static synthetic、solver-only GRPO 和完整 self-play；这些实验必须
使用相同 token、rollout 和图调用预算。

## 6. 故障排查和降配顺序

- SGLang 启动失败：先检查 adapter 路径、模型缓存和 `/health`，再检查 `round_*/logs/`。
- Virtuoso timeout：不要把 timeout 当负样本；检查 endpoint、索引和 cache 权限。
- 显存不足：先降低 batch、response length、rollout N，再启用 optimizer/parameter offload。
- Questioner 全部无效：回到 SFT，检查 TaskProposal JSON、禁止 `all_entities` 和 topic root 一致性。
- 角色坍缩：检查两角色格式率和 reward 分布，再调整 0.35/0.65 权重。

本仓库只验证 CPU 逻辑与命令契约；训练是否取得提升必须由服务器实验和独立 benchmark 证明。
