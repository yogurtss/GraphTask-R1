# 其他数据集与图快照

KQAPro 的下载、转换、SFT 数据导出和训练已统一到
[KQAPro 训练流程](KQAPRO_TRAINING.md)，本页不再重复。本文只保留 KILT、SSP、Freebase、
WebQSP、CWQ 和 GrailQA 的数据说明。

原始文件只读保留；转换产物记录 source hash、converter version、split、配置和 seed。原始数据、
图数据库、训练 Parquet 和访问 token 不提交到 Git。

## 1. 目录

```text
data/
├── raw/          # 官方原始文件
├── processed/    # 图快照与 certified tasks
├── training/     # SFT/GRPO/self-play Parquet
└── cache/        # 外部图查询缓存
```

## 2. KILT、HotpotQA 与 TriviaQA（CoEvoKG 数据协议对齐）

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

### 2.1 KILT bounded smoke build

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

### 2.2 HotpotQA 与 TriviaQA 测试集

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

### 2.3 从 KILT 图生成认证 GRPO 训练集

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

## 3. Freebase 与 Virtuoso

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

## 4. WebQSP、CWQ 与 GrailQA

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

### 4.1 防止 self-play 使用 held-out seeds

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

## 5. 训练前质量门

进入 GPU 训练前应全部满足：

- `data audit` 通过且没有 duplicate ID；
- bounded diagnostic 的 accepted canonical trace replay 为 100%；
- gold answer 全部来自程序执行；
- `SOURCE_ANSWER_MISMATCH` 和 `INCOMPLETE_SLICE` 已人工抽查；
- Freebase endpoint 无持续 timeout，缓存目录可写；
- 三个 benchmark 的 held-out denylist 已合并；
- SFT/GRPO Parquet 中不存在 test 问题文本或逻辑形式。
