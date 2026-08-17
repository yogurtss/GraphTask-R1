# 数据集与图快照准备

本文档描述从官方原始文件到 GraphTask-R1 训练 Parquet 的完整过程。不要把 `data/`、模型权重、
Virtuoso 数据库或 token 提交到 Git。

如果 `data/processed/kqapro/kqapro-v1/graph.sqlite` 以及 `train/val/tasks.parquet` 已经存在，说明
KQA Pro 转换已经完成。此时不要再次运行 `data prepare`；直接跳到 2.3 节，用
`data export-sft` 重新生成 Solver SFT，并用独立的 `data export-questioner-sft --count N` 生成
精确 N 条 Questioner SFT。prompt 合约变化后两种角色都应重导出，不要混用旧 Parquet；这些导出
过程不会修改 processed 数据。

所有耗时数据命令默认向 stderr 输出 `INFO` 级进度日志，包含当前阶段、完成数、总数、百分比、
耗时以及 accepted/rejected 等计数；最终 JSON 仍单独写到 stdout。需要安静运行时，将全局参数放在
命令组之前：`python -m graphtask_r1.cli --log-level WARNING data prepare ...`。

`data prepare` 支持逐记录并发，但默认使用 1 个 worker。多个线程会竞争同一 SQLite 文件的
磁盘页缓存，而且 Python 侧的校验和序列化受 GIL 影响，实际经常比串行更慢。服务器上应先用
相同的 `--limit` 比较，再按实测设置 `--workers N`；不建议直接按 CPU 核数配置。
KQA 的每个 worker 使用独立的只读 SQLite 连接，结果仍按原始 index 排序，因此相同输入、seed
与版本会产生相同任务、trace 和 rejection 顺序。`kb.json -> graph.sqlite` 是单写者构建阶段，
不会并发写库。若现有 `graph.sqlite` 的源文件 hash、转换器版本和 snapshot 均匹配，prepare 会
直接复用；需要强制重建时添加 `--rebuild-graph`。

KQAPro 原始 JSON array、任务 Parquet、audit、relation catalog、SFT/RL export 和 ms-swift
preflight 均按 bounded batch 流式读写。新生成的数据直接使用 `audit --deep`；只有迁移旧版、每条
内联数万 witness facts 的文件时，才去掉 `--deep` 使用快速检查。

## 1. 目录和不可变性

```text
data/
├── raw/                         # 官方原始文件，只读保留
│   ├── kqa_pro/
│   ├── webqsp/
│   ├── complexwebquestions/
│   ├── grailqa/
│   └── freebase_setup/
├── processed/
│   ├── kqapro/kqapro-v1/
│   ├── webqsp/
│   ├── cwq/
│   └── grailqa/
├── training/                    # SFT/GRPO/self-play Parquet
└── cache/                       # 外部图查询缓存（含 Virtuoso）
```

每次转换都会记录原文件 SHA-256、转换器版本、split、配置和统计。原文件更新后应创建新的
snapshot 目录，不要覆盖已用于实验的 snapshot。

## 2. KQA Pro

官方来源：[KQAPro_Baselines](https://github.com/shijx12/KQAPro_Baselines)。官方仓库将数据集
标为 CC BY-SA 4.0；使用 Hugging Face 镜像下载时仍应按官方数据许可归因。

### 2.1 下载和文件检查

```bash
python -m pip install huggingface_hub
python -m graphtask_r1.cli data fetch --dataset kqapro --raw-dir data/raw

find data/raw/kqa_pro -maxdepth 2 -type f
```

准备器要求：

```text
data/raw/kqa_pro/kb.json
data/raw/kqa_pro/train.json
data/raw/kqa_pro/val.json
data/raw/kqa_pro/test.json
```

若镜像多包了一层目录，将这四个文件移动到上述位置即可。不要用 `test.json` 生成训练任务；
官方 test split 不包含可用于训练的 gold program/answer。

### 2.2 转换

先用小样本检查环境：

```bash
python -m graphtask_r1.cli data prepare --dataset kqapro \
  --raw-dir data/raw/kqa_pro \
  --output-dir data/processed/kqapro/kqapro-v1-smoke \
  --splits train,val --limit 20 --train-sample-size 20 \
  --verification-mode full --trace-mode canonical --seed 42 --workers 1
```

确认无系统性错误后，分层抽取 20,000 条 train，并处理完整 val：

```bash
python -m graphtask_r1.cli data prepare --dataset kqapro \
  --raw-dir data/raw/kqa_pro \
  --output-dir data/processed/kqapro/kqapro-v1 \
  --splits train,val --train-sample-size 20000 \
  --verification-mode source --trace-mode none \
  --seed 42 --workers 1 --max-witness-facts 0
```

在 WSL 中应将 `--output-dir` 放在 Linux ext4（例如仓库内的 `data/processed`），不要直接写
`/mnt/g` 的 DrvFS/9p；完成后再复制归档到 G 盘。真实 KQAPro 小样本上 `--workers 2` 比 1 略快，
但 4 个线程会出现 SQLite 争用和更高峰值内存，因此先用 `--limit 100 --workers 1/2` 实测后再定，
不要盲目增加线程。

转换器执行以下步骤：

1. 将 `kb.json` 构建为带 subject/object/type/attribute/fact/qualifier 索引的 `graph.sqlite`；
2. 将 `Find/FindAll/Relate/And/Or/FilterConcept/Filter*/Count/What`，以及
   `QueryAttr/QueryRelation/SelectBetween/SelectAmong`，以及全部 `QFilter*`、qualifier query 和
   `Verify*` 映射到 typed DSL；
3. 对 train 按 operator set、terminal operator 和程序长度桶做可回放的分层抽样；
4. 重新执行 DSL 产生 gold answer；
5. 将实体 ID 对应标签与原 KQA answer 对账；
6. source verification 不套用 Questioner 的 shortcut/answer-leak 门；只有显式 full verification
   才运行必要性和 shortcut 检查；
7. canonical trace 默认关闭，只在显式 `--trace-mode canonical` 时编译和 replay；
8. SFT 默认不内联 witness，不再用无关邻域填满 5 万条边；
9. 接受样本以 bounded row group 写入 `tasks.parquet`/`traces.parquet`，其他样本写入
   `rejections.parquet`。

默认 `--max-witness-facts 0` 会显式设置 `witness_complete=false` 和
`generation.witness_omitted=true`。这不影响 gold 或 source verification；它们都由完整程序执行
产生。若归档实验需要 inline witness，可设置正整数；超过上限的记录会设置
`generation.witness_truncated=true`。提高该上限不会增加可训练样本数，只会增加图查询、I/O 和
存储，通常不应为 SFT 开启。

KQAPro 官方 27 个 KoPL 函数均有确定性映射；qualifier 依附于稳定 fact ID，避免将同一实体的
不同 relation/attribute statement 错配。gold 仍只由认证程序执行产生。

默认 `--train-sample-size 20000`；传 `0` 才处理完整 train。抽样详情写入
`train/sampling.json`，包含 source/selected rows、strata 配额、operator coverage、seed 与 sampler
version。val 不抽样。默认 `--verification-mode source --trace-mode none` 是 SFT 推荐路径；小规模
诊断才使用 `--verification-mode full --trace-mode canonical`。

### 2.3 产物审计

只执行下面这组必要步骤：

```bash
for split in train val; do
  python -m graphtask_r1.cli data audit \
    --input data/processed/kqapro/kqapro-v1/$split/tasks.parquet \
    --kind task --deep \
    --training-view-output \
      data/processed/kqapro/kqapro-v1/$split/training_tasks.parquet
done

python -m graphtask_r1.cli data build-relation-catalog \
  --input \
    data/processed/kqapro/kqapro-v1/train/training_tasks.parquet \
  --scope graph \
  --output data/processed/kqapro/kqapro-v1/relation_catalog.json

for split in train val; do
  python -m graphtask_r1.cli data export-sft \
    --input data/processed/kqapro/kqapro-v1/$split/training_tasks.parquet \
    --output data/training/kqapro_graphscript_v03_solver_sft_$split.parquet \
    --roles solver --interaction-mode graphscript --graphscript-version 0.3 \
    --relation-catalog data/processed/kqapro/kqapro-v1/relation_catalog.json
done

python -m graphtask_r1.cli data export-questioner-sft \
  --input data/processed/kqapro/kqapro-v1/train/training_tasks.parquet \
  --output data/training/kqapro_graphscript_v03_questioner_sft_train.parquet \
  --count 2048 --seed 42 --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog data/processed/kqapro/kqapro-v1/relation_catalog.json
```

`audit --deep` 同时检查完整 `TaskCertificate` 并生成不含 witness 的 training view；它不重复执行
program，因为 gold 和 source answer 对账已由 `data prepare` 完成。canonical trace 只在 bounded
diagnostic 中 replay。`--scope graph` 从 `graph.sqlite` 生成固定的完整 relation allowlist，不依赖
小样本覆盖率。

`export-questioner-sft` 是 snapshot-neutral 的独立入口：它只要求 certified task、对应
`GraphBackend` 和 relation catalog。它用实例级 RNG 对单 seed tasks 做确定性 reservoir sampling，
输出行数必须等于 `--count`，不足时直接失败。Questioner prompt 仅加入实体 label、type 和有界局部
relation IDs，不写入邻居实体或答案；根操作固定按 ID 解析。预检后使用 `data combine-sft` 将角色
隔离的 accepted 文件确定性混合，详见 [训练手册](TRAINING.md#2-sft-长度预检)。

必须检查 `metrics.json` 中的接受率和各 reason code。`SOURCE_ANSWER_MISMATCH`、
`INCOMPLETE_SLICE` 或 `TRACE_REPLAY_MISMATCH` 大量出现时不要训练。

## 3. KILT、HotpotQA 与 TriviaQA（CoEvoKG 数据协议对齐）

这条路线只对齐知识源和最终测试分布，不采用 CoEvoKG 的 chain-conditioned proposer、LLM
quality gate、path reward 或 evidence write-back。GraphTask 的训练 gold 仍只能由认证程序执行产生；
SSP 问题不会被转换成 `TaskCertificate`，也不会进入 self-play archive。

当前 Windows 数据盘统一布局如下（WSL 路径为 `/mnt/g/datasets/GraphTaskDataset`）：

```text
G:\datasets\GraphTaskDataset\
├── kilt_knowledgesource.json
├── ssp\ce7a0dfbc862f923ad1668a471c409b2e023b73f\test.jsonl
├── hotpotqa\ssp-test\test\examples.parquet
└── triviaqa\ssp-test\test\examples.parquet
```

SSP 固定 revision 为 `ce7a0dfbc862f923ad1668a471c409b2e023b73f`，原始 `test.jsonl`
SHA-256 为 `871c7b7cdec2e090e8597ef26a9a973a46aad0830bb1e016679dddd748462f50`。
适配器会验证该 release 的 bucket 数量；HotpotQA 和 TriviaQA 应各为 500 题。原始 SSP 中的
MuSiQue 不属于 CoEvoKG 最终六数据集，默认会排除。

### 3.1 KILT bounded smoke build

先只读取 100 页，验证 hyperlink 图与 SQLite FTS5 passage index：

```bash
python -m graphtask_r1.cli data prepare --dataset kilt \
  --raw-dir /mnt/g/datasets/GraphTaskDataset/kilt_knowledgesource.json \
  --output-dir data/processed/kilt/kilt-2019-08-01-smoke \
  --limit 100 --workers 1
```

确认 `metrics.json`、`rejections.parquet`、图遍历和文本检索后再做全量构建。当前原文件为
37,318,876,722 字节，SHA-256 是
`f966d6f09c4ff91656db5c56c384f136b0c495c7083c043586b8cb1033c389a5`；根目录
`checksums.sha256` 同时固定了 KILT 与 SSP。全量构建会生成页面节点、`wikipedia_link` 边、
category type、Wikidata ID 属性和 passage FTS，应预留至少 150 GB。若只需要 hyperlink 图，
可加 `--no-text-index` 降低空间占用。

```bash
python -m graphtask_r1.cli data prepare --dataset kilt \
  --raw-dir /mnt/g/datasets/GraphTaskDataset/kilt_knowledgesource.json \
  --output-dir /mnt/g/datasets/GraphTaskDataset/kilt-2019-08-01-v1 \
  --workers 1

export GRAPHTASK_KILT_DB=/mnt/g/datasets/GraphTaskDataset/kilt-2019-08-01-v1/graph.sqlite
python -m graphtask_r1.cli graph preflight --snapshot kilt-2019-08-01-v1 --limit 5
```

KILT importer 是单写者、流式且原子替换；运行时图 backend 只读且 instance-scoped。没有
`wikipedia_id` 的 anchor 不会被猜测性匹配，而是按页面聚合保存
`MISSING_ANCHOR_TARGET` reason code。

### 3.2 HotpotQA 与 TriviaQA 测试集

若需重新下载固定 SSP release：

```bash
python -m graphtask_r1.cli data fetch --dataset ssp \
  --raw-dir /mnt/g/datasets/GraphTaskDataset
```

这两个 SSP test bucket 都是 open-domain 输入：共 1,000 题且 `topic_entity_ids` 全为空。
正式测试必须使用带 FTS5 passage index 的完整 KILT snapshot，不能使用
`--no-text-index` 构建物。Solver 可以先调用 bounded `text_search` 获得页面 ID 和 passage，
再使用 `graph_search`/`inspect_entity` 补充多跳证据；答案仍按 SSP alias EM/F1 评测。

分别生成三个最终测试集：

```bash
python -m graphtask_r1.cli data prepare --dataset ssp \
  --raw-dir /mnt/g/datasets/GraphTaskDataset/ssp/ce7a0dfbc862f923ad1668a471c409b2e023b73f \
  --output-dir /mnt/g/datasets/GraphTaskDataset/hotpotqa/ssp-test \
  --include-datasets hotpotqa --workers 2

python -m graphtask_r1.cli data prepare --dataset ssp \
  --raw-dir /mnt/g/datasets/GraphTaskDataset/ssp/ce7a0dfbc862f923ad1668a471c409b2e023b73f \
  --output-dir /mnt/g/datasets/GraphTaskDataset/triviaqa/ssp-test \
  --include-datasets triviaqa --workers 2

python -m graphtask_r1.cli data prepare --dataset ssp \
  --raw-dir /mnt/g/datasets/GraphTaskDataset/ssp/ce7a0dfbc862f923ad1668a471c409b2e023b73f \
  --output-dir /mnt/g/datasets/GraphTaskDataset/nq/ssp-test \
  --include-datasets nq --workers 2
```

三个适配后数据集允许空 topic entity，并把答案 alias 保存为等价类；评测采用与
Search-R1/CoEvoKG 相同的 lowercase、去标点、去冠词、空白归一化 EM。KILT task 格式和
HotpotQA/TriviaQA 官方格式也由相同 adapter 支持，但只有上述固定 SSP release 可以声明
`CoEvoKG test parity`。

### 3.3 从 KILT 图生成认证 GRPO 训练集

KILT knowledge source 不能直接作为 GRPO prompt。`bootstrap-kilt-grpo` 从只读 KILT 图按显式
seed 采样 `hop1`、`hop2`、`type_filter` 和 `count` 程序，执行完整 GraphTask 认证，并只把
通过的 `TaskCertificate` 导出到统一的 GraphScript v0.2 Solver GRPO contract。模型只看到问题并
输出可执行 code，不输出自由文本答案。所有 gold answer 来自程序执行；
SSP 的问题和答案不会被读取。

先在 smoke 图上验证：

```bash
export GRAPHTASK_KILT_DB=$PWD/data/processed/kilt/kilt-2019-08-01-smoke/graph.sqlite

python -m graphtask_r1.cli data bootstrap-kilt-grpo \
  --output-dir data/processed/kilt/kilt-grpo-smoke \
  --count 64 --pool-limit 100 --max-attempts 3200 \
  --val-ratio 0.125 --seed 42

python -m graphtask_r1.cli data audit \
  --input data/processed/kilt/kilt-grpo-smoke/train/tasks.parquet --kind task
```

产物包括：

```text
kilt-grpo-smoke/
├── train/tasks.parquet
├── train/traces.parquet
├── train/solver_grpo.parquet
├── val/tasks.parquet
├── val/traces.parquet
├── val/solver_grpo.parquet
├── rejections.parquet
├── relation_catalog.json
├── metrics.json
└── manifest.json
```

正式数据若希望得到 20,096 train + 512 validation，可请求总计 20,608 个 accepted tasks：

```bash
export GRAPHTASK_KILT_DB=/mnt/g/datasets/GraphTaskDataset/kilt-2019-08-01-v1/graph.sqlite

python -m graphtask_r1.cli data bootstrap-kilt-grpo \
  --output-dir /mnt/g/datasets/GraphTaskDataset/processed/kilt/kilt-certified-grpo-v2 \
  --count 20608 --pool-limit 1000000 \
  --val-ratio 0.02484472 --seed 42 \
  --interaction-mode graphscript --graphscript-version 0.2
```

达到 `max_attempts` 仍不足 requested count 时命令会失败，同时保留 `metrics.json` 和结构化
`rejections.parquet`，不会静默输出不完整训练集。正式运行前必须检查 accepted count、各 program
family 接受数、rejection reason 分布和 canonical trace replay。

## 4. Freebase 与 Virtuoso

主实验使用 GrailQA 推荐的 [Freebase-Setup](https://github.com/dki-lab/Freebase-Setup)。Freebase
数据和 Virtuoso 索引通常需要数百 GB；开始前根据官方 dump 的实际大小预留充足磁盘。

```bash
python -m graphtask_r1.cli data fetch --dataset freebase --raw-dir data/raw
cd data/raw/freebase_setup
```

之后严格执行该仓库当前 README 中的 dump 下载、Virtuoso 安装、加载和索引步骤。项目不会自动
接受未知第三方镜像，也不会绕过下载许可。服务就绪后：

```bash
export FREEBASE_ENDPOINT=http://127.0.0.1:8890/sparql
export GRAPHTASK_GRAPH_TIMEOUT=20
export GRAPHTASK_GRAPH_RETRIES=2
export GRAPHTASK_GRAPH_CACHE=$PWD/data/cache/freebase.sqlite

python -m graphtask_r1.cli graph preflight --snapshot freebase-v1 --limit 5
```

快照在数据中只记录 `freebase-v1`；endpoint 和凭证只通过环境变量传入。

## 5. WebQSP、CWQ 与 GrailQA

- WebQSP：使用 Microsoft/Freebase-Setup 指向的官方文件。
- ComplexWebQuestions v1.1：<https://www.tau-nlp.sites.tau.ac.il/compwebq>
- GrailQA：<https://dki-lab.github.io/GrailQA/>，CC BY-SA 4.0。

下载脚本对需要点击许可或人工下载的数据只打印官方地址。把 JSON/JSONL 放到对应目录后运行：

```bash
python -m graphtask_r1.cli data prepare --dataset webqsp \
  --raw-dir data/raw/webqsp --output-dir data/processed/webqsp --workers 1

python -m graphtask_r1.cli data prepare --dataset cwq \
  --raw-dir data/raw/complexwebquestions --output-dir data/processed/cwq --workers 1

python -m graphtask_r1.cli data prepare --dataset grailqa \
  --raw-dir data/raw/grailqa --output-dir data/processed/grailqa --workers 1
```

适配器保留原始 SPARQL/逻辑形式，统一 gold entity ID、问题、split 和 topic entities。主评测
使用 gold topic entities，避免把实体链接误差混入图推理结果。

### 5.1 防止 self-play 使用 held-out seeds

```bash
python -m graphtask_r1.cli data merge-denylists \
  --inputs \
    data/processed/webqsp/heldout_topic_entities.json \
    data/processed/cwq/heldout_topic_entities.json \
    data/processed/grailqa/heldout_topic_entities.json \
  --output data/processed/freebase_heldout_entities.json

python -m graphtask_r1.cli data sample-seeds --snapshot freebase-v1 \
  --exclude data/processed/freebase_heldout_entities.json \
  --count 4096 --pool-limit 100000 --seed 42 \
  --output data/training/freebase_questioner_seeds.parquet
```

采样器还会过滤度数小于 2、度数大于 100 和常见 metadata 实体。benchmark 的 dev/test 问题、
逻辑形式和 topic entities 不得进入 Questioner prompt 或 archive。

## 6. 训练前质量门

进入 GPU 训练前应全部满足：

- `data audit` 通过且没有 duplicate ID；
- bounded diagnostic 的 accepted canonical trace replay 为 100%；
- gold answer 全部来自程序执行；
- `SOURCE_ANSWER_MISMATCH` 和 `INCOMPLETE_SLICE` 已人工抽查；
- Freebase endpoint 无持续 timeout，缓存目录可写；
- 三个 benchmark 的 held-out denylist 已合并；
- SFT/GRPO Parquet 中不存在 test 问题文本或逻辑形式。
