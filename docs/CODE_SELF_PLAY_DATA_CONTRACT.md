# Code-first self-play 数据契约

主实验统一使用 `GraphScript v0.2`：模型输入自然语言问题，输出一个有界、可执行、带类型的 JSON
程序；答案只能由执行器产生。KQAPro、KILT 和最终 OpenQA 评测不再各自维护一套算子或依赖隐式
topic entity。

| 阶段 | 数据 | 模型输入 | 模型输出 | 数据用途 |
|---|---|---|---|---|
| SFT | KQAPro train | question + 当前图的 relation catalog | GraphScript v0.2 | code 冷启动 |
| GRPO | KILT train | question + KILT relation catalog | GraphScript v0.2 | self-play/code reward |
| 内部选择 | KILT val | 同 GRPO | GraphScript v0.2 | checkpoint 选择，不作为最终结论 |
| 最终评测 | HotpotQA、TriviaQA、NaturalQuestions SSP test | question + KILT catalog | GraphScript v0.2 | 只读外部评测 |

SSP test 不得用于 relation catalog、entity seed、问题生成、reward 调参或训练数据回填。这里仅借用
[CoEvoKG](https://arxiv.org/abs/2608.01904) 的 train-only source pool / external SSP evaluation
边界；不使用它的自由文本 solver、chain proposer、path reward 或 evidence write-back。

## 统一算子集

每条 SFT/GRPO 行都显式记录 `graphscript_version=0.2` 和同一个 `operator_set`：

```text
start, all_entities, resolve_entity, search_passage, passage_pages, follow,
intersect, union, filter_type, filter_literal, count,
query_attribute, query_relation, select_between, select_among,
require_unique, emit
```

`resolve_entity` 和 `search_passage` 解决无 topic 的入口问题；`passage_pages` 把检索命中的证据页变成
可继续图遍历的 entity handle。其余算子覆盖当前 typed Program IR。程序最多 64 个操作，handle 为
`h0..h63`；relation 必须来自当前 snapshot 的训练侧 catalog。v0.1 仅保留给严格两跳公平对比，不应
混入主实验。

当前 passage 路径能确定性回答“答案是 KILT 页面实体/标题”的样本。对答案仅存在于正文 span、且
不是页面实体的样本，v0.2 会记为可解析但不可正确作答；不要把 `passage_pages` 的页面标题伪装成
正文答案。后续如增加 reader/extract 算子，SFT、GRPO、val 必须同时切换版本并使用同一执行器。

## 数据生成

KQAPro 导入默认把 trace call budget 提高到 32、compact query result budget 提高到 1024；可在
20K–40K token 环境下继续显式调整：

```bash
python -m graphtask_r1.cli data prepare --dataset kqapro \
  --raw-dir /mnt/g/datasets/GraphTaskDataset/kqapro/raw \
  --output-dir /mnt/g/datasets/GraphTaskDataset/processed/kqapro/kqapro-v1 \
  --max-trace-tool-calls 32 --max-trace-query-results 1024

python -m graphtask_r1.cli data export-sft \
  --input /mnt/g/datasets/GraphTaskDataset/processed/kqapro/kqapro-v1/train/tasks.parquet \
  --output /mnt/g/datasets/GraphTaskDataset/training/kqapro_graphscript_v02_sft.parquet \
  --roles solver --interaction-mode graphscript --graphscript-version 0.2 \
  --relation-catalog /mnt/g/datasets/GraphTaskDataset/processed/kqapro/kqapro-v1/relation_catalog.json
```

KILT bootstrap 的默认输出已经是 v0.2 code GRPO：

```bash
export GRAPHTASK_KILT_DB=/mnt/g/datasets/GraphTaskDataset/processed/kilt/kilt-2019-08-01-v1/graph.sqlite
python -m graphtask_r1.cli data bootstrap-kilt-grpo \
  --output-dir /mnt/g/datasets/GraphTaskDataset/processed/kilt/kilt-certified-grpo-v2 \
  --count 20608 --pool-limit 1000000 --val-ratio 0.02484472 --seed 42 \
  --interaction-mode graphscript --graphscript-version 0.2
```

## 验证

ms-swift SFT 预检仍按训练时模板和 token policy 切分 valid/overlong/invalid；推荐先在 32K 检查，
必要时升到 40K：

```bash
conda run -n ms-swift-debug python scripts/preflight_ms_swift_sft.py \
  --input /mnt/g/datasets/GraphTaskDataset/training/kqapro_graphscript_v02_sft.parquet \
  --accepted-output outputs/preflight-kqapro-v02/accepted.parquet \
  --rejected-output outputs/preflight-kqapro-v02/rejected.parquet \
  --summary-output outputs/preflight-kqapro-v02/metrics.json \
  --model Qwen/Qwen3-4B-Instruct-2507 --max-length 32768
```

最终 SSP 评测除了 EM/F1，还必须报告 `program_parse_rate`、`program_execution_rate`、
`mean_program_operators` 和 `mean_passage_searches`，从而区分“代码格式失败”“执行失败”和“执行成功但
语义错误”。冻结 solver 服务及 ms-swift GRPO 的 completion 默认 32768，允许范围为 1–40960。
