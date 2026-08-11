# 交互模式与统一程序接口

主实验只使用 `GraphScript v0.2` 作为模型输出。显式工具调用保留为消融模式，但它通过同一个
ms-swift 数据加载器、图后端和 reward 运行，不拥有独立数据 schema 或训练入口。

| 模式 | 模型行为 | ms-swift 调度 | 用途 |
| --- | --- | --- | --- |
| `graphscript` | 一次生成完整 JSON 程序，执行器返回答案 | 单轮 | SFT、GRPO、self-play、最终验证 |
| `tool` | 多轮调用 `graph_search`、`inspect_entity`、`text_search` | `graphtask_solver` | 消融与兼容性检查 |

## GraphScript v0.2

输入只保证自然语言问题，不保证 topic entity。程序可用 `resolve_entity` 或 `search_passage` 建立
入口，再使用统一算子：

```text
start, all_entities, resolve_entity, search_passage, passage_pages,
follow, intersect, union, filter_type, filter_literal, count,
query_attribute, query_relation, select_between, select_among,
require_unique, emit
```

约束如下：

- 至多 64 个操作，SSA handle 为 `h0..h63`；
- `follow` 有显式方向、relation 和返回上限；
- 所有 materialization 受 `max_follow_limit`、`max_edge_visits` 和
  `max_returned_entities` 限制；
- 最后一个操作必须为 `emit`；
- 模型不得输出自由文本答案，gold 只能来自执行器；
- SFT、GRPO 和 benchmark 使用同一个 parser、schema、operator set 与 executor。

v0.1 只用于固定两跳的历史公平对比，不混入主实验。

## 数据字段

训练文件保存 GraphTask 中立字段：

- SFT：`messages`、`role`、`task_id`、`interaction_mode`、
  `graphscript_version`、`operator_set`；
- RL：`prompt`、`reward_model`、`extra_info`、`uid`；
- `extra_info` 保存 graph snapshot、角色、限制、relation catalog ID、operator set 和 trace 上下文。

GraphScript 行加载后不附加 tool schema。`tool` 行则由
`graphtask_r1/training/ms_swift_data.py` 动态生成 ms-swift `tools` 字段，源 Parquet 不保存某个
训练框架的 agent/session 私有字段。

## 导出示例

```bash
python -m graphtask_r1.cli data export-sft \
  --input data/processed/kqapro/kqapro-v1/train/tasks.parquet \
  --output data/training/kqapro_graphscript_v02_sft_train.parquet \
  --roles solver --interaction-mode graphscript --graphscript-version 0.2 \
  --relation-catalog data/processed/kqapro/kqapro-v1/relation_catalog.json

python -m graphtask_r1.cli data export-rl \
  --input data/processed/kqapro/kqapro-v1/train/tasks.parquet \
  --output data/training/kqapro_graphscript_v02_rl.parquet \
  --roles solver --interaction-mode graphscript --graphscript-version 0.2 \
  --relation-catalog data/processed/kqapro/kqapro-v1/relation_catalog.json
```

若运行工具消融，将两条命令的 `--interaction-mode` 改为 `tool`，并在启动脚本中显式设置
`INTERACTION_MODE=tool`。不要把两种模式混入同一个训练 split。
