# 交互模式与统一程序接口

KQAPro 主实验只使用 `GraphScript v0.3` 作为模型输出；KILT passage-search 单独保留 v0.2。
显式工具调用保留为消融模式，但它通过同一个
ms-swift 数据加载器、图后端和 reward 运行，不拥有独立数据 schema 或训练入口。

| 模式 | 模型行为 | ms-swift 调度 | 用途 |
| --- | --- | --- | --- |
| `graphscript` | 一次生成完整 JSON 程序，执行器返回答案 | 单轮 | SFT、GRPO、self-play、最终验证 |
| `tool` | 多轮调用 `graph_search`、`inspect_entity`、`text_search` | `graphtask_solver` | 消融与兼容性检查 |

## KQAPro GraphScript v0.3

输入只保证自然语言问题，不保证 topic entity。程序用 `resolve_entity` 或 `all_entities` 建立
入口：

```text
all_entities, resolve_entity, follow, intersect, union, filter_type, filter_literal,
filter_qualifier, count, query_attribute, query_attribute_under_condition,
query_attribute_qualifier, query_relation, query_relation_qualifier, verify,
select_between, select_among, emit
```

约束如下：

- 至多 64 个操作，SSA handle 为 `h0..h63`；
- `follow` 有显式方向、relation 和返回上限；
- 所有 materialization 受 `max_follow_limit`、`max_edge_visits` 和
  `max_returned_entities` 限制；
- 最后一个操作必须为 `emit`；
- 模型不得输出自由文本答案，gold 只能来自执行器；
- SFT、GRPO 和 benchmark 使用同一个 parser、schema、operator set 与 executor。

v0.2 的 `search_passage`/`passage_pages` 仅属于 KILT；v0.1 只用于固定两跳消融。

## 数据字段

训练文件保存 GraphTask 中立字段：

- SFT：`messages`、`role`、`task_id`、`interaction_mode`、
  `graphscript_version`、`operator_set`；
- RL：`prompt`、`reward_model`、`extra_info`、`uid`；
- `extra_info` 保存 graph snapshot、角色、限制、relation catalog ID、operator set 和 trace 上下文。

GraphScript 行加载后不附加 tool schema。`tool` 行则由
`graphtask_r1/training/ms_swift_data.py` 动态生成 ms-swift `tools` 字段，源 Parquet 不保存某个
训练框架的 agent/session 私有字段。

## SFT 一键导出

```bash
export TRAIN_TASKS=/path/to/train/tasks.parquet
export VAL_TASKS=/path/to/val/tasks.parquet
export SOLVER_RATIO=9
export QUESTIONER_RATIO=1
export MODEL_PATH=/path/to/model
bash scripts/prepare_mixed_sft_data.sh
```

脚本固定生成 GraphScript SFT，并一次完成角色隔离导出、按实际 Solver 行数配比、确定性混合和模板
预检。底层独立导出仍可用于排障。RL 导出继续使用对应 CLI：

```bash
python -m graphtask_r1.cli data export-rl \
  --input data/processed/kqapro/kqapro-v1/train/tasks.parquet \
  --output data/training/kqapro_graphscript_v03_rl.parquet \
  --roles solver --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog data/processed/kqapro/kqapro-v1/relation_catalog.json
```

脚本调用的 `export-questioner-sft` 是独立、定量且 snapshot-neutral 的入口。自然语言 prompt
不强调数据集名或内部版本名；`"version":"0.3"` 只作为机器解析字段保留。Questioner 的 fixed
seed root 与 Solver 问题编译仍是两个角色合约，预检前不要把两类原始行直接混在一起。

若运行工具消融，应使用底层命令将 `--interaction-mode` 改为 `tool`，并在启动脚本中显式设置
`INTERACTION_MODE=tool`。不要把两种模式混入同一个训练 split。
