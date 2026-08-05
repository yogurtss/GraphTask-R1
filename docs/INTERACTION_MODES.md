# Tool use 与 GraphScript 双模式

GraphTask-R1 现在支持两条端到端交互路径，同时保留原有 `tool/tool` 为默认行为：

| 模式 | Questioner | Solver | verl agent |
|---|---|---|---|
| `tool` | 多轮 `graph_search / inspect_entity / execute_program` | 多轮图搜索后提交 `<answer>` | `tool_agent` |
| `graphscript` | 一次提交受限 GraphScript | 一次提交受限 GraphScript，执行结果即答案 | `single_turn_agent` |

GraphScript 不是 Python。v0.1 只接受严格 JSON，形状固定为
`start -> follow -> follow -> require_unique -> emit`，禁止文件、网络、shell、循环及任意代码执行。
它会编译到现有 typed `Program`，gold answer、verifier、necessity/shortcut 检查、reward、archive
和 graph backend 均继续复用现有实现。

## 公平比较约束

- 首轮只比较 `tool/tool` 与 `graphscript/graphscript` 两个端到端系统；结果不能归因到单一角色。
- 两组分别训练一个共享 Questioner/Solver LoRA，使用相同 base checkpoint、样本、seed、训练步数、
  frozen-opponent 采样数和 relation catalog。
- 两组均使用 `graphscript_v0_1` task profile：单 topic、两跳 chain、唯一实体答案、无已知 shortcut。
- relation catalog 是按 graph snapshot 划分的全局训练目录，不包含某个 episode 的可达路径或答案。
- `edge_visits` 定义为后端返回的 primitive triples。GraphScript 执行有硬预算；tool rollout 同时记录
  每次搜索返回的 triples，verl 日志中的总量用于效率比较。
- comparison profile 中，tool Questioner 将总 edge budget 静态分成搜索与候选执行两半，并限制一次
  `execute_program`；tool Solver 的全部预算用于搜索。这样无需跨 tool 实例的全局状态也能保证硬上限。
- 主报告需同时给出 optimizer steps、实际训练 token、LLM calls、edge visits 和 wall time；不能仅按
  LLM call 数或仅按 step 数宣称公平。

## 1. 从现有 KQA Pro 任务选出共同子集

```bash
graphtask-r1 data select-interaction-tasks \
  --input data/processed/kqapro/kqapro-v1/train/tasks.parquet \
  --output data/processed/kqapro/kqapro-v1/train/interaction_tasks.parquet

graphtask-r1 data select-interaction-tasks \
  --input data/processed/kqapro/kqapro-v1/val/tasks.parquet \
  --output data/processed/kqapro/kqapro-v1/val/interaction_tasks.parquet

graphtask-r1 data build-relation-catalog \
  --input data/processed/kqapro/kqapro-v1/train/interaction_tasks.parquet \
  --output data/processed/kqapro/kqapro-v1/relation_catalog.json
```

命令会保留结构化 rejection records 和 metrics。不要用 val/test 任务扩充训练 relation catalog。
筛选会在任务对应的 graph backend 上同时执行认证 Program 与有界 GraphScript；只有 seed、gold 和
有界结果一致，且满足默认 `max_follow_limit=100`、`max_edge_visits=200` 的任务才会进入共同子集。
输入必须属于同一 graph snapshot；如实验预算不同，使用同名命令行参数显式覆盖默认值。

## 2. 导出两组 SFT 数据

```bash
graphtask-r1 data export-sft \
  --input data/processed/kqapro/kqapro-v1/train/interaction_tasks.parquet \
  --output data/verl/kqapro_tool_sft.parquet \
  --interaction-mode tool \
  --relation-catalog data/processed/kqapro/kqapro-v1/relation_catalog.json

graphtask-r1 data export-sft \
  --input data/processed/kqapro/kqapro-v1/train/interaction_tasks.parquet \
  --output data/verl/kqapro_graphscript_sft.parquet \
  --interaction-mode graphscript \
  --relation-catalog data/processed/kqapro/kqapro-v1/relation_catalog.json
```

分别训练并保存两个 SFT adapter。两组必须从同一个 base checkpoint 开始；不要让一个 arm 从另一个
arm 的 adapter 继续训练。

## 3. 构建 Freebase relation catalog 与 seeds

Freebase catalog 必须来自训练侧任务。可从已有 legacy tool self-play archive 导出，再构建 catalog：

```bash
graphtask-r1 data export-archive \
  --archive outputs/selfplay-qwen3-4b/archive.sqlite \
  --output data/processed/freebase_train_tasks.parquet

graphtask-r1 data build-relation-catalog \
  --input data/processed/freebase_train_tasks.parquet \
  --output data/processed/freebase_relation_catalog.json \
  --snapshot freebase-v1
```

随后为两个 arm 用相同 seed、denylist 和 catalog 分别导出 seed rows：

```bash
graphtask-r1 data sample-seeds --snapshot freebase-v1 --seed 42 \
  --exclude data/processed/freebase_heldout_entities.json \
  --relation-catalog data/processed/freebase_relation_catalog.json \
  --interaction-mode tool --output data/verl/freebase_tool_seeds.parquet

graphtask-r1 data sample-seeds --snapshot freebase-v1 --seed 42 \
  --exclude data/processed/freebase_heldout_entities.json \
  --relation-catalog data/processed/freebase_relation_catalog.json \
  --interaction-mode graphscript --output data/verl/freebase_graphscript_seeds.parquet
```

## 4. 运行两个独立 self-play arm

Tool comparison arm 使用 `configs/training/selfplay_tool_compare.yaml`；GraphScript arm 使用
`configs/training/selfplay_graphscript.yaml`。两者必须使用不同的 `INITIAL_ADAPTER`、
`QUESTIONER_SEEDS` 和 output directory，但环境变量 `BASE_TASKS`、`VAL_DATA`、模型、seed 与预算一致。

```bash
export RELATION_CATALOG=$PWD/data/processed/freebase_relation_catalog.json
export KQAPRO_RELATION_CATALOG=$PWD/data/processed/kqapro/kqapro-v1/relation_catalog.json

graphtask-r1 train self-play \
  --config configs/training/selfplay_tool_compare.yaml \
  --output-dir outputs/interaction-tool --dry-run

graphtask-r1 train self-play \
  --config configs/training/selfplay_graphscript.yaml \
  --output-dir outputs/interaction-graphscript --dry-run
```

先检查两个 dry-run plan 的 model、round、数据量、预算和 GPU 布局完全一致，再分别启动真实训练。
现有 `configs/training/selfplay.yaml` 继续使用 `interaction_mode: tool` 和 `program_profile: full`，
不受对比 profile 限制。

## 验收顺序

1. ToyGraph：解析、预算、唯一性、tool/program 执行等价及 replay 全部通过。
2. KQA Pro SFT：GraphScript parse/schema/executable rate 和 tool-call success rate 达标。
3. Frozen Solver 校准：两个模式的 pass-rate 分布都不能集中在 0 或 1。
4. Freebase smoke：固定 seeds、bounded limits，各跑一个短 round，确认 reward 可重算且 resume 连续。
5. 完整比较：至少 3 个训练 seed；报告 valid/frontier rate、frontier per 1K edge visits、token、calls、
   latency 和 Standard/High-Branching 分桶。
