# Qwen3-4B + verl 训练手册

项目固定 verl `v0.7.1`，commit：
`bec9ef74768dd201881cd4e54cd0385e87caae27`。默认模型是
`Qwen/Qwen3-4B-Instruct-2507`，非 thinking 模式，共享 LoRA rank/alpha 为 32/64。

## 1. 环境

PyTorch、verl、SGLang、Ray、FlashAttention 和 CUDA 栈均由服务器环境独立管理，本仓库不会
安装或升级它们。verl 的源码目录可以放在服务器任意位置，不需要复制或 clone 到本仓库的
`third_party/` 下；只需确保当前 Python 环境可以 `import verl`。项目验证版本为 verl `v0.7.1`
及上述 commit。

clone GraphTask-R1 后进入仓库根目录。服务器若还缺少本项目的轻量运行依赖，再执行：

```bash
python -m pip install -r requirements.txt
```

GraphTask-R1 本身无需安装。可在训练前检查当前服务器环境是否可见：

```bash
python -c "import torch, verl, sglang; print(torch.__version__, verl.__file__, sglang.__file__)"
```

先运行 verl 自带的 Qwen3-4B multi-turn 示例，确认 CUDA、Ray、SGLang 和 tool-call template
正常。升级 verl 后必须重新检查 Hydra 字段和 Parquet contract。

## 2. 双角色 SFT

如果以下三个 processed 文件已经存在，不要再次执行 `data prepare`：

```text
data/processed/kqapro/kqapro-v1/graph.sqlite
data/processed/kqapro/kqapro-v1/train/tasks.parquet
data/processed/kqapro/kqapro-v1/val/tasks.parquet
```

SFT 报告缺少 `kqapro_sft_train.parquet` 时，只需从现有 accepted tasks 导出训练 Parquet。导出器
只读 `tasks.parquet` 和 `graph.sqlite`，不会重新处理或覆盖 KQA Pro：

```bash
export GRAPHTASK_KQAPRO_DB=$PWD/data/processed/kqapro/kqapro-v1/graph.sqlite

python -m graphtask_r1.cli data export-sft \
  --input data/processed/kqapro/kqapro-v1/train/tasks.parquet \
  --output data/verl/kqapro_sft_train.parquet --roles both

python -m graphtask_r1.cli data export-sft \
  --input data/processed/kqapro/kqapro-v1/val/tasks.parquet \
  --output data/verl/kqapro_sft_val.parquet --roles both
```

导出后应得到：

```text
data/verl/kqapro_sft_train.parquet
data/verl/kqapro_sft_val.parquet
```

然后：

```bash
export SFT_TRAIN_DATA=$PWD/data/verl/kqapro_sft_train.parquet
export SFT_VAL_DATA=$PWD/data/verl/kqapro_sft_val.parquet
export SFT_OUTPUT_DIR=$PWD/outputs/sft-qwen3-4b

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft.yaml --dry-run
python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft.yaml
```

SFT 数据使用 verl `messages` contract。Questioner 学习输出结构化 TaskProposal；Solver 学习
真实 `graph_search`/`inspect_entity` 调用和 `<answer>`。两种角色写入同一 adapter。

## 3. Solver-only GRPO

```bash
python -m graphtask_r1.cli data export-verl \
  --input data/processed/kqapro/kqapro-v1/train/tasks.parquet \
  --output data/verl/kqapro_solver_rl.parquet --roles solver

python -m graphtask_r1.cli data export-verl \
  --input data/processed/kqapro/kqapro-v1/val/tasks.parquet \
  --output data/verl/kqapro_solver_rl_val.parquet --roles solver

export SFT_ADAPTER=/absolute/path/to/sft/lora_adapter
export SOLVER_RL_TRAIN_DATA=$PWD/data/verl/kqapro_solver_rl.parquet
export SOLVER_RL_VAL_DATA=$PWD/data/verl/kqapro_solver_rl_val.parquet
export SOLVER_GRPO_OUTPUT_DIR=$PWD/outputs/solver-grpo

python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo.yaml --dry-run
python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo.yaml
```

先确认格式有效率、工具成功率和 held-out KQA F1，再进入 KQA Pro SQLite self-play。

## 4. KQA Pro mixed-role self-play

默认路径完全使用已经由 `kb.json` 构建的 KQA Pro SQLite 图，不需要下载 Freebase、启动
Virtuoso 或设置 `FREEBASE_ENDPOINT`。accepted KQA 任务作为不可变 base pool，Questioner 从
同一图的独立实体 seeds 出发生成新任务；认证通过的任务写入 archive，并在后续轮次混入 Solver
batch。

默认 GPU 布局：

```text
GPU 0–1  verl actor/rollout/ref，共享 LoRA mixed-role GRPO
GPU 2–3  上一轮冻结 LoRA 的 SGLang Solver，DP=2、TP=1
CPU      GraphTask opponent/archive service、KQA Pro SQLite 和 reward workers
```

先导出 KQA Pro seeds。当前 orchestrator 每轮读取 seed Parquet 的全部行，因此先让 seed 数量与
默认的 `solver_episodes: 256` 对齐；短轮稳定后再提高到 512 或 1024。

```bash
export GRAPHTASK_KQAPRO_DB=$PWD/data/processed/kqapro/kqapro-v1/graph.sqlite

python -m graphtask_r1.cli data sample-seeds \
  --snapshot kqapro-v1 \
  --count 256 --pool-limit 100000 --seed 42 \
  --output data/verl/kqapro_questioner_seeds.parquet
```

准备训练环境变量：

```bash
export INITIAL_ADAPTER=/absolute/path/to/solver_grpo_or_sft_adapter
export BASE_TASKS=$PWD/data/processed/kqapro/kqapro-v1/train/tasks.parquet
export VAL_DATA=$PWD/data/verl/kqapro_solver_rl_val.parquet
export QUESTIONER_SEEDS=$PWD/data/verl/kqapro_questioner_seeds.parquet
```

先检查完整进程计划：

```bash
python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay-kqapro --dry-run
```

确认路径后启动：

```bash
python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay-kqapro
```

每轮使用上轮 adapter 启动冻结 Solver。Questioner 的每个有效候选通过异步 `/evaluate`
接口执行 8 次真实工具 rollout；其 pass rate、结构必要性和 novelty 共同形成 reward。接受任务由
单写者 SQLite archive 幂等保存，并在下一轮进入 Solver batch。基础设施请求重试后仍失败会
中止该 job，不会伪装成 Solver 答错。

恢复：

```bash
python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay-kqapro --resume
```

resume 会核对 config hash，并从最后完成 round 的 adapter 继续。不要修改已有 round 目录。

## 5. 验证与可选 Freebase benchmark

默认实验使用 `kqapro_solver_rl_val.parquet` 监控 held-out KQA 指标。WebQSP、CWQ 和 GrailQA
的问题文件本身不能替代知识图，它们仍需要 Freebase backend；没有 Freebase 时不要运行下面的
可选 benchmark，也不要把缺失的图查询当作模型错误。

用待评测 adapter 启动冻结 Solver SGLang 和 GraphTask service，命令形态与每轮 `plan.json`
中的 `sglang`、`opponent` 一致。以后准备好 Freebase 服务后可以运行：

```bash
python -m graphtask_r1.cli evaluate benchmark \
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
- KQA 图打开失败：检查 `GRAPHTASK_KQAPRO_DB` 是否指向已生成的只读 `graph.sqlite`。
- 可选 Freebase 路径出现 Virtuoso timeout：不要把 timeout 当负样本；检查 endpoint、索引和 cache 权限。
- 显存不足：先降低 batch、response length、rollout N，再启用 optimizer/parameter offload。
- Questioner 全部无效：回到 SFT，检查 TaskProposal JSON、禁止 `all_entities` 和 topic root 一致性。
- 角色坍缩：检查两角色格式率和 reward 分布，再调整 0.35/0.65 权重。

本仓库只验证 CPU 逻辑与命令契约；训练是否取得提升必须由服务器实验和独立 benchmark 证明。
