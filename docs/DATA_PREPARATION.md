# 数据集与图快照准备

本文档描述从官方原始文件到 GraphTask-R1 训练 Parquet 的完整过程。不要把 `data/`、模型权重、
Virtuoso 数据库或 token 提交到 Git。

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
├── cache/                       # Virtuoso 查询缓存
└── verl/                        # SFT/GRPO/self-play Parquet
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
  --output-dir data/processed/kqapro/kqapro-v1 \
  --splits train,val --limit 100 --seed 42 --workers 1
```

确认无系统性错误后，删除这个临时输出目录并运行全量转换：

```bash
python -m graphtask_r1.cli data prepare --dataset kqapro \
  --raw-dir data/raw/kqa_pro \
  --output-dir data/processed/kqapro/kqapro-v1 \
  --splits train,val --seed 42 --workers 1
```

转换器执行以下步骤：

1. 将 `kb.json` 构建为带 subject/object/type/attribute 索引的 `graph.sqlite`；
2. 将 `Find/FindAll/Relate/And/Or/FilterConcept/Filter*/Count/What` 映射到 core DSL；
3. 重新执行 DSL 产生 gold answer；
4. 将实体 ID 对应标签与原 KQA answer 对账；
5. 运行局部物化、必要性、shortcut、answer leakage 和 canonical trace replay；
6. 接受样本写入 `tasks.parquet`/`traces.parquet`，其他样本写入 `rejections.parquet`。

Qualifier、属性查询、Verify、极值选择等不属于首版研究核心的 KoPL 操作不会被猜测性转换，
而是保留原程序并标记 `UNSUPPORTED_KOPL_OPERATOR`。

### 2.3 产物审计

```bash
python -m graphtask_r1.cli data audit \
  --input data/processed/kqapro/kqapro-v1/train/tasks.parquet --kind task

python -m graphtask_r1.cli data export-sft \
  --input data/processed/kqapro/kqapro-v1/train/tasks.parquet \
  --output data/verl/kqapro_sft_train.parquet --roles both

python -m graphtask_r1.cli data export-sft \
  --input data/processed/kqapro/kqapro-v1/val/tasks.parquet \
  --output data/verl/kqapro_sft_val.parquet --roles both
```

必须检查 `metrics.json` 中的接受率和各 reason code。`SOURCE_ANSWER_MISMATCH`、
`INCOMPLETE_SLICE` 或 `TRACE_REPLAY_MISMATCH` 大量出现时不要训练。

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
  --output data/verl/freebase_questioner_seeds.parquet
```

采样器还会过滤度数小于 2、度数大于 100 和常见 metadata 实体。benchmark 的 dev/test 问题、
逻辑形式和 topic entities 不得进入 Questioner prompt 或 archive。

## 5. 训练前质量门

进入 GPU 训练前应全部满足：

- `data audit` 通过且没有 duplicate ID；
- accepted canonical trace replay 为 100%；
- gold answer 全部来自程序执行；
- `SOURCE_ANSWER_MISMATCH` 和 `INCOMPLETE_SLICE` 已人工抽查；
- Freebase endpoint 无持续 timeout，缓存目录可写；
- 三个 benchmark 的 held-out denylist 已合并；
- SFT/GRPO Parquet 中不存在 test 问题文本或逻辑形式。
