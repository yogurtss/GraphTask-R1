# Code-first self-play 数据契约

当前主实验只使用 KQAPro `GraphScript v0.3`：模型输入自然语言问题，输出一个有界、可执行、
带类型的 JSON 程序，答案只能由执行器产生。KILT/OpenQA 使用独立 v0.2 passage-search profile，
不共享 SFT、GRPO、relation catalog、seed pool 或 checkpoint。

| 阶段 | 数据 | 模型输入 | 模型输出 | 数据用途 |
|---|---|---|---|---|
| SFT | KQAPro train | question + KQAPro catalog | GraphScript v0.3 | code 冷启动 |
| GRPO/self-play | KQAPro train | 同 SFT | GraphScript v0.3 | Questioner/Solver reward |
| 内部选择 | KQAPro val | 同 SFT | GraphScript v0.3 | checkpoint 选择，只读 |

SSP test 不得用于 relation catalog、entity seed、问题生成、reward 调参或训练数据回填。这里仅借用
[CoEvoKG](https://arxiv.org/abs/2608.01904) 的 train-only source pool / external SSP evaluation
边界；不使用它的自由文本 solver、chain proposer、path reward 或 evidence write-back。

## KQAPro v0.3 算子集

每条 KQAPro SFT/GRPO 行都显式记录 `graphscript_version=0.3` 和同一个 `operator_set`：

```text
all_entities, resolve_entity, follow, intersect, union, filter_type, filter_literal,
filter_qualifier, count, query_attribute, query_attribute_under_condition,
query_attribute_qualifier, query_relation, query_relation_qualifier, verify,
select_between, select_among, emit
```

`resolve_entity`/`all_entities` 解决入口，五个 qualifier/verify 算子覆盖原先缺失的 11 个 KoPL
函数。SQLite 快照必须保存 relation/attribute 的稳定 fact ID 和 qualifier，不能只保存扁平 triples。
程序最多 64 个操作，handle 为 `h0..h63`；relation 与 qualifier key 必须来自 KQAPro catalog。
v0.2 的 `search_passage`/`passage_pages` 只属于 KILT，v0.1 只用于两跳消融。

## 数据生成

KQAPro 导入默认把 trace call budget 提高到 32、compact query result budget 提高到 1024；可在
20K–40K token 环境下继续显式调整：

```bash
python -m graphtask_r1.cli data prepare --dataset kqapro \
  --raw-dir /mnt/g/datasets/GraphTaskDataset/raw/kqa_pro \
  --output-dir /mnt/g/datasets/GraphTaskDataset/processed/kqapro/kqapro-v1 \
  --max-trace-tool-calls 32 --max-trace-query-results 1024

python -m graphtask_r1.cli data export-sft \
  --input /mnt/g/datasets/GraphTaskDataset/processed/kqapro/kqapro-v1/train/tasks.parquet \
  --output /mnt/g/datasets/GraphTaskDataset/training/kqapro_graphscript_v03_sft.parquet \
  --roles solver --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog /mnt/g/datasets/GraphTaskDataset/processed/kqapro/kqapro-v1/relation_catalog.json
```

KILT bootstrap 与 v0.2 代码仍保留，但不在当前默认训练链运行；启用前需要单独完成 KILT SFT 与
reader/passage 算子设计。

## 验证

ms-swift SFT 预检仍按训练时模板和 token policy 切分 valid/overlong/invalid；推荐先在 32K 检查，
必要时升到 40K：

```bash
conda run -n ms-swift-debug python scripts/preflight_ms_swift_sft.py \
  --input /mnt/g/datasets/GraphTaskDataset/training/kqapro_graphscript_v03_sft.parquet \
  --accepted-output outputs/preflight-kqapro-v03/accepted.parquet \
  --rejected-output outputs/preflight-kqapro-v03/rejected.parquet \
  --summary-output outputs/preflight-kqapro-v03/metrics.json \
  --model Qwen/Qwen3-4B-Instruct-2507 --max-length 32768
```

最终 SSP 评测除了 EM/F1，还必须报告 `program_parse_rate`、`program_execution_rate`、
`mean_program_operators` 和 `mean_passage_searches`，从而区分“代码格式失败”“执行失败”和“执行成功但
语义错误”。冻结 solver 服务及 ms-swift GRPO 的 completion 默认 32768，允许范围为 1–40960。
