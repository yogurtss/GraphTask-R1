# GraphTask-R1 对齐 CoEvoKG 数据集的最小改造调研报告

> 实现状态（2026-08-11）：SSP 固定 revision importer、HotpotQA/TriviaQA 各 500 题测试集、
> alias-aware normalized EM、KILT hyperlink/FTS5 流式 importer、SQLite snapshot factory 和
> bounded integration tests 已完成。35GB KILT 已通过 100 页 smoke build；全量图/FTS 构建尚未
> 启动，因此当前可声明 SSP test parity，不能声明 KILT exact artifact parity。
> 当前 KILT 原文件为 37,318,876,722 字节，SHA-256 为
> `f966d6f09c4ff91656db5c56c384f136b0c495c7083c043586b8cb1033c389a5`。

> 调研日期：2026-08-11
>
> 范围：只讨论数据、下载、适配与实验协议，不实现代码
>
> 论文证据读本：[coevokg_reader/paper.md](coevokg_reader/paper.md)
>
> 前序综合报告：[COEVOKG_DATASET_RESEARCH_REPORT.md](COEVOKG_DATASET_RESEARCH_REPORT.md)

## 1. 结论先行

### 1.1 “完全一样”目前只能做到一部分

| 对齐对象 | 能否精确复现 | 结论 |
|---|---|---|
| KILT Wikipedia 底库 | 可以 | 使用官方 2019-08-01 KILT knowledge source，固定原始文件哈希 |
| 最终测试题 | 可以 | 固定 `Quark-LLM/SSP` revision，下载 `test.jsonl`，排除 MuSiQue，得到论文使用的 2,625 题 |
| 测试集组成与数量 | 可以 | NQ、TriviaQA、PopQA、HotpotQA、2WikiMultiHopQA 各 500，Bamboogle 125 |
| E5 retriever 型号 | 可以 | `intfloat/e5-base-v2`；top-3、最多 8 turns 可按论文设置 |
| 正式训练 source benchmark 清单 | **不可以** | 论文与公开仓库没有发布完整名单、版本、比例和样本 ID |
| 正式 seed/fallback 与 validation 样本 | **不可以** | 只公开数据格式与上限，没有正式文件或 manifest |
| 正式初始 chain pool | **不可以** | 只公开构造程序，没有发布实验所用 chain 文件、随机种子、输入 manifest 和哈希 |
| 正式 passage corpus / FAISS index | **不可以逐文件复现** | 模型和 KILT 来源已知，但正式切段、索引文件与哈希未发布 |

因此，当前最严谨的目标应命名为：

> **CoEvoKG public-protocol parity（公开协议对齐）**，而不是 exact artifact reproduction（逐样本、逐文件完全复现）。

若作者后续发布正式 source manifest、seed/validation parquet、chain pool、corpus 与 index 哈希，才能升级为真正的 exact reproduction。测试侧则现在就能精确对齐。

### 1.2 推荐路线

推荐采用“**相同数据环境，不同训练方法**”路线：

1. 使用相同的 KILT 2019-08-01 Wikipedia 快照；
2. 使用相同的 SSP 2,625 题作为一次性最终测试；
3. 用 source-train QA 的答案与 provenance 只决定 GraphTask 的 seed 分布；
4. 仍由 GraphTask 的 Questioner 生成 `Program`，由固定图快照执行程序得到 gold，并做 shortcut、反事实必要性与泄漏检查；
5. 不引入 CoEvoKG 的 chain-conditioned proposer、LLM quality gate 作为 gold 判定、path-support reward 或 evidence write-back。

这样可以最大限度对齐知识源和评测分布，同时保住 GraphTask-R1 的核心方法边界。

## 2. CoEvoKG 实际使用了哪些数据

### 2.1 训练侧

论文与公开代码共同确认训练时有三类输入：

| 数据输入 | 用途 | 已确认来源 |
|---|---|---|
| chain pool | 给 proposer 提供 2–3 hop 任务素材 | source benchmark 的 **train split** 答案定位到 KILT 页面，再沿 Wikipedia hyperlink 构造链 |
| seed/fallback QA | 生成候选全部被拒时补齐 Solver batch | 从 source benchmark 的训练数据中划出 |
| train-derived validation | 选择 checkpoint | 从训练来源中另行划出，排除最终 SSP 题 |

论文明确说链池来自 KILT/Wikipedia 上的离线 2–3 hop walk（[S001](coevokg_reader/paper.md#S001)、[S002](coevokg_reader/paper.md#S002)），fallback 来自固定人工 QA 集（[S003](coevokg_reader/paper.md#S003)）。公开代码进一步表明：它先读取 train-only source manifest，再用训练题 gold answer 定位 KILT 页面，并从该页面做随机游走。

但以下关键内容**原文未明确说明，仓库也未发布**：

- 正式训练到底包含哪些 source benchmark；
- 每个来源采用哪个 revision、多少样本、什么比例；
- seed pool、validation pool 和 chain anchor pool 的逐条 ID；
- 正式 chain pool 的大小、去重结果、随机种子与 SHA-256；
- 正式 KILT passage 切分规则、FAISS index 文件与哈希。

公开仓库中存在 HotpotQA、2Wiki、MuSiQue、NQ、通用 QA、Bamboogle 等 loader，只能证明程序支持这些格式，**不能据此断言正式实验使用了全部 loader**。

### 2.2 测试侧

最终测试来自公开 SSP release，而不是直接使用 KILT 的 dev/test 文件：

| 数据集 | SSP `data_source` | 数量 |
|---|---|---:|
| Natural Questions | `searchR1_nq` | 500 |
| TriviaQA | `searchR1_triviaqa` | 500 |
| PopQA | `searchR1_popqa` | 500 |
| HotpotQA | `searchR1_hotpotqa` | 500 |
| 2WikiMultiHopQA | `searchR1_2wikimultihopqa` | 500 |
| Bamboogle | `searchR1_bamboogle` | 125 |
| **论文合计** |  | **2,625** |

SSP 文件还包含 `searchR1_musique` 500 题，因此原始 `test.jsonl` 共 3,125 行；CoEvoKG 没有把 MuSiQue 纳入最终六数据集结果。论文说明这些 SSP 子集只用于最终报告，训练验证与 fallback 来自训练划分（[S004](coevokg_reader/paper.md#S004)、[S005](coevokg_reader/paper.md#S005)）。

## 3. “相同数据、不同方法”的边界

### 3.1 可以共用的内容

- KILT 2019-08-01 Wikipedia 页面、段落、标题、Wikipedia ID 与 hyperlink；
- 同一 source-train QA 池及其答案/provenance（在作者 manifest 未公开前只能使用我们声明的近似 manifest）；
- 同一 SSP 最终测试文件和过滤规则；
- 若要做严格环境对比，可共用 `intfloat/e5-base-v2`、top-3 和 8-turn 上限；
- 相同的训练样本上限 20,096、validation 上限 512、314 个优化步，用于等预算实验。

### 3.2 不应照搬的内容

| CoEvoKG 组件 | GraphTask-R1 的处理 |
|---|---|
| 按一条现成 entity chain 让 proposer 改写问题 | 不使用；Questioner 从 seed 主动探索并提出可执行 `Program` |
| frozen LLM quality gate 决定候选是否可用 | 不作为 gold 权威；只允许做辅助文本质量检查 |
| gold 由链尾答案锚点提供 | 不使用；gold 必须执行认证程序得到 |
| Solver path-support 奖励 | 不直接复制；保留 answer reward 与现有成本项，可另行研究轨迹—证书一致性 |
| 正确轨迹写回图并成为后续链 | 不复制；继续使用 accepted task archive，底层认证图保持固定、可回放 |
| GRPO + REINFORCE++ 的具体双角色更新 | 不因换数据而改变现有训练算法 |

### 3.3 对“完全一样”的操作性定义

建议给实验命名并分层，避免论文表述过度：

| 层级 | 要求 | 推荐命名 |
|---|---|---|
| L0 测试对齐 | 同一 SSP revision、同一 2,625 IDs、同一归一化 EM | `SSP-test parity` |
| L1 语料对齐 | L0 + 同一 KILT 2019-08-01 原始文件哈希 | `KILT-corpus parity` |
| L2 协议对齐 | L1 + train-only 近似 source manifest、20,096/512、314 steps | `public-protocol parity` |
| L3 产物对齐 | L2 + 作者正式 manifest、chain、corpus、index 和所有哈希 | `exact artifact parity`；当前做不到 |

本项目当前应以 L2 为目标，并在结果表脚注明“formal CoEvoKG training manifest was not publicly released”。

## 4. 最小代码改造设计（数据适配边界已实现）

### 4.1 为什么不能只把 JSON 路径换掉

现有 GraphTask-R1 的训练主线消费 `TaskCertificate`，它要求 `Program`、SPARQL、执行所得 `gold_answers`、`witness_facts` 和验证摘要；`base_tasks` 也被直接解析为 `TaskCertificate`。KILT/SSP 是自然语言 QA，并不自带 GraphTask 可执行程序。

此外，现有 Solver 工具按 entity ID 做 `graph_search`；CoEvoKG/SSP 的 Solver 从问题文本发出检索 query。许多 SSP 问题没有可直接交给现有工具的 topic entity ID。因而简单改 `data_path` 会同时破坏：

- 认证 gold 的来源；
- Questioner seed 的语义；
- Solver 的首跳入口；
- 自然语言答案及 alias 的评测；
- 训练轨迹与最终 SSP 评测的一致性。

### 4.2 推荐的最小改造：新增适配层，不改核心算法

建议把改动限制在六个边界内：

| 边界 | 规划改造 | 尽量不动的部分 |
|---|---|---|
| 数据下载/manifest | 新增 KILT、source QA、SSP 下载与 manifest 生成命令；每条记录保存 revision、raw ID、split、SHA-256 | archive 与训练循环 |
| KILT 图导入 | 将 KILT 页面作为节点、inline anchor 作为有向边、段落作为节点证据，导入 SQLite；snapshot 命名为如 `kilt-2019-08-01-v1` | `GraphBackend` 协议、程序执行器、overlay 机制 |
| seed 采样 | 支持从 source-train QA 的 answer/provenance 页面采样 seed；映射失败时保留结构化 rejection reason | Questioner 的主动生成逻辑 |
| 检索适配 | 增加 KILT passage 检索 backend，并提供 query-text `search` 模式；若只做 KG 实验可不开启 | Questioner/Solver 的训练算法与奖励定义 |
| benchmark 适配 | 允许 NQ、TriviaQA、PopQA、HotpotQA、2Wiki、Bamboogle；支持空 topic entity、字符串答案、alias 与 normalized EM | 现有 WebQSP/CWQ/GrailQA 转换器 |
| 配置与审计 | 新增独立 `kilt_ssp` 配置、denylist、近重复/实体重叠审计 | 现有 Freebase/KQA Pro 配置与默认行为 |

这不是一个“重写项目”的方案。核心 schema 中 `TaskCertificate`、`Program`、certifier、necessity、shortcut、challenger/solver reward、archive、自博弈轮次都保持不变。

### 4.3 训练数据如何进入 GraphTask

推荐数据流如下：

```text
source benchmark train QA
        │
        ├─ answer/provenance → KILT page ID → questioner seed distribution
        │                                      │
        │                                      ▼
        │                           GraphTask Questioner explores
        │                                      │
        │                                      ▼
        │                           Program + certified execution
        │                                      │
        │                                      ▼
        └─────────────────────────── accepted TaskCertificate

KILT 2019-08-01 paragraphs ──→ optional E5 search backend ──→ Solver observations
SSP 2,625 held-out questions ───────────────────────────────→ final evaluation only
```

关键点是：source QA 决定“从哪里开始采样”，不决定新任务 gold。新任务答案仍只能来自认证程序执行。这既利用了 CoEvoKG 的数据分布，又没有复制它的 chain-to-question 方法。

### 4.4 文件级影响预估

当前实现保持“新增为主、少量扩展”：

| 位置 | 预计变化 |
|---|---|
| `graphtask_r1/data/` | 新增 KILT/SSP importer、source manifest、answer-to-page 映射与审计 |
| `graphtask_r1/graph/` | 复用 SQLite backend；新增 KILT 导入器，必要时只给 entity info 增加 passage sidecar |
| `graphtask_r1/graph/factory.py` | 增加一个 `kilt-2019-08-01-v1` snapshot 分支 |
| `graphtask_r1/schema/task.py` | 扩展 benchmark dataset 类型；训练证书结构不变 |
| `graphtask_r1/evaluation/` | 新增 SSP normalized EM、alias 归一化、macro/micro 指标 |
| `graphtask_r1/training/` 与工具配置 | 仅在需要与 CoEvoKG 共用文本检索环境时增加 `retrieval` interaction mode |
| `configs/` | 新增独立配置，不修改现有默认实验 |
| `tests/` | importer、映射、固定随机种子、denylist、SSP 过滤、backend 与端到端小样本测试 |

若目标只是先完成 L0 测试对齐，最小工作只涉及 SSP importer、benchmark schema、text-search evaluation adapter 和指标；若目标是 L2 训练对齐，才需要 KILT 图导入和 answer-anchored seed sampler。

## 5. KILT 与相关数据的官方下载方法

以下命令只是操作规范，**本轮没有执行下载**。建议不要把大文件提交到 Git；仓库只保存 manifest、脚本、哈希和小型 fixture。

### 5.1 推荐目录

```text
data/
├── raw/
│   ├── kilt/2019-08-01/
│   ├── kilt_tasks/<revision>/
│   ├── popqa/<revision>/
│   ├── 2wikimultihopqa/<revision>/
│   └── ssp/ce7a0df.../
├── processed/
│   └── kilt/kilt-2019-08-01-v1/
│       ├── graph.sqlite
│       ├── passages.jsonl
│       ├── faiss.index
│       └── manifests/
└── manifests/
    ├── source_train.jsonl
    ├── train_validation.jsonl
    ├── ssp_coevokg_2625.jsonl
    └── checksums.sha256
```

### 5.2 KILT Wikipedia knowledge source：主推荐方式

KILT 官方仓库给出的语料是 2019-08-01 Wikipedia，5,903,530 个页面，原始 JSON 约 34.76 GiB；Hugging Face 卡片显示下载后约 37.32 GB。官方原始文件是最清楚的 snapshot 锚点：

```bash
mkdir -p data/raw/kilt/2019-08-01
cd data/raw/kilt/2019-08-01
wget -c http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json
sha256sum kilt_knowledgesource.json > kilt_knowledgesource.json.sha256
```

KILT 官方没有在 README 同时给出一个可比对的 SHA-256，因此这里的哈希用于固定本项目实际收到的字节。首次下载后应把哈希、字节数、下载日期与 URL 写入 manifest。

如果要运行 CoEvoKG 的公开 chain preprocessing 进行外部复核，它默认使用 MongoDB：

```bash
mongoimport \
  --db kilt \
  --collection knowledgesource \
  --file data/raw/kilt/2019-08-01/kilt_knowledgesource.json
```

GraphTask-R1 正式路线不需要把 MongoDB 引入训练依赖；更小的维护面是一次性导入本项目已有 SQLite 图格式，运行时只读。

官方来源：[KILT GitHub README](https://github.com/facebookresearch/KILT#kilt-knowledge-source)、[Hugging Face KILT Wikipedia](https://huggingface.co/datasets/facebook/kilt_wikipedia)。KILT GitHub 已于 2023-10-31 归档为只读，因此一定要保存 revision 与本地哈希。

### 5.3 KILT 任务数据

KILT 官方提供统一下载脚本：

```bash
git clone https://github.com/facebookresearch/KILT.git third_party/KILT
cd third_party/KILT
git rev-parse HEAD
mkdir -p data
python scripts/download_all_kilt_data.py
python scripts/get_triviaqa_input.py
```

其中与 CoEvoKG 最终评测同名、且 KILT 官方提供 train split 的开放域 QA 有：

| KILT task | train 数量 | 备注 |
|---|---:|---|
| Natural Questions | 87,372 | KILT 格式含 answer/provenance |
| HotpotQA | 88,869 | KILT open-domain 映射版 |
| TriviaQA | 61,844 | 需运行 `get_triviaqa_input.py` 关联 question |

也可以从官方 [KILT 数据目录](https://github.com/facebookresearch/KILT#kilt-data-catalogue) 逐文件下载，或使用 [facebook/kilt_tasks](https://huggingface.co/datasets/facebook/kilt_tasks)。不过要特别强调：**下载这三个 train split 不等于已经还原 CoEvoKG 正式训练集**；作者没有确认正式 source manifest 就是这三者。

### 5.4 PopQA

PopQA 不属于原始 KILT task catalogue。官方仓库提供 `data/popQA.tsv`，也给出 Hugging Face 方式：

```bash
git clone https://github.com/AlexTMallen/adaptive-retrieval.git third_party/adaptive-retrieval
sha256sum third_party/adaptive-retrieval/data/popQA.tsv
```

或：

```python
import datasets

popqa = datasets.load_dataset("akariasai/PopQA")["test"]
```

官方来源：[Adaptive Retrieval / PopQA](https://github.com/AlexTMallen/adaptive-retrieval#popqa)。PopQA 官方发布主要是约 14k 题的 test 数据，并没有可直接等同于 CoEvoKG“source benchmark train split”的标准训练划分，因此不要自行把它切成 train 后宣称与论文相同。

### 5.5 2WikiMultiHopQA

使用作者官方仓库 README 中的 Dropbox dataset 链接，优先选择带 `evidences_id`、`answer_id` 和 `id_aliases.json` 的更新版：

```bash
git clone https://github.com/Alab-NII/2wikimultihop.git third_party/2wikimultihop
git -C third_party/2wikimultihop rev-parse HEAD
```

数据压缩包应从 [2WikiMultiHopQA 官方仓库](https://github.com/Alab-NII/2wikimultihop#new-update-april-7-2021) 的 “Here” 链接下载；解压后记录 `train.json`、`dev.json`、`test.json`、`id_aliases.json` 的逐文件哈希。不要依赖二次转载链接来声称 exact reproduction。

2Wiki 的 Wikipedia/Wikidata ID 与 KILT 2019-08-01 页面并非天然一一对应，导入时必须输出 `mapped`、`ambiguous`、`missing` 三类结构化结果，不能用模糊标题匹配静默兜底。

### 5.6 SSP 最终测试集：可精确复现

Hugging Face 数据集页面当前存在 schema cast error，因此不建议对整个仓库直接调用 `load_dataset()`。应固定页面披露的 revision，单独下载 `test.jsonl`：

```bash
python -m pip install "huggingface_hub>=0.24"
hf download Quark-LLM/SSP test.jsonl \
  --repo-type dataset \
  --revision ce7a0dfbc862f923ad1668a471c409b2e023b73f \
  --local-dir data/raw/ssp/ce7a0dfbc862f923ad1668a471c409b2e023b73f

sha256sum \
  data/raw/ssp/ce7a0dfbc862f923ad1668a471c409b2e023b73f/test.jsonl
```

过滤出 CoEvoKG 的六组，并核对数量：

```bash
mkdir -p data/manifests
jq -c 'select(.data_source != "searchR1_musique")' \
  data/raw/ssp/ce7a0dfbc862f923ad1668a471c409b2e023b73f/test.jsonl \
  > data/manifests/ssp_coevokg_2625.jsonl

wc -l data/manifests/ssp_coevokg_2625.jsonl
jq -r '.data_source' data/manifests/ssp_coevokg_2625.jsonl | sort | uniq -c
```

预期总数是 2,625；六个 bucket 必须分别为 500、500、500、500、500、125。官方页面与已知问题：[Quark-LLM/SSP](https://huggingface.co/datasets/Quark-LLM/SSP)。

### 5.7 E5 retriever

若要做与 CoEvoKG 相同的文本检索环境，再下载 E5；如果只先构建 KILT hyperlink 图，可以暂缓：

```bash
hf download intfloat/e5-base-v2 \
  --repo-type model \
  --local-dir data/models/intfloat/e5-base-v2
```

固定 Hugging Face revision，并在 passage 编码时遵守 E5 的 query/passage 输入约定。论文配置为每次 top-3、Solver 最多 8 turns（[S006](coevokg_reader/paper.md#S006)）。模型卡：[intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)。

### 5.8 容量规划

- KILT 官方原始 JSON：约 35–37 GB；
- MongoDB 或 SQLite 导入、索引、passage corpus、临时文件和 FAISS 会显著放大空间；
- 工程上建议至少预留 **150 GB**，若同时保留 Mongo、SQLite、embedding 中间结果和多版索引，建议 **250–300 GB**。

后两项是工程容量估算，不是论文报告值。正式下载前先检查磁盘，不应在训练节点临时盘上无 manifest 地重复构建。

## 6. 数据构建与隔离协议

### 6.1 我们需要发布的 source manifest

由于作者 manifest 缺失，我们自己的近似版必须逐条可审计：

```json
{
  "source_dataset": "nq",
  "source_revision": "<pinned revision>",
  "source_split": "train",
  "source_id": "...",
  "raw_sha256": "...",
  "question_sha256": "...",
  "answer_strings": ["..."],
  "kilt_wikipedia_ids": ["..."],
  "mapping_status": "mapped",
  "usage": "questioner_seed",
  "sampling_seed": 42
}
```

必须同时发布：每个来源的 raw count、过滤原因计数、映射成功率、抽样前后数量、sampling seed、最终 sample IDs 和文件 SHA-256。未映射、歧义映射和被 denylist 拒绝的记录都要保留结构化原因。

### 6.2 split 规则

建议按以下顺序执行：

1. 先固定 SSP 2,625 的 raw IDs、问题文本哈希、答案 alias、涉及标题/实体；
2. 构建 test denylist：精确 ID、规范化文本、近重复文本、答案—实体组合；
3. 只从 source benchmark 原始 train split 选择候选；
4. 先划出 512 条 train-derived validation，再构建 Questioner seed pool；
5. validation 不进入 Questioner seed、base task、archive 或训练 rollout；
6. SSP 不参与 checkpoint selection；所有超参数冻结后只跑最终一次；
7. 报告 exact overlap、near-duplicate、answer overlap、entity overlap 四种审计，而不是只检查 question ID。

### 6.3 两套评测必须并存

只换成 SSP 会丢失 GraphTask 的方法优势证据，因此建议保留两条互补评测线：

- **SSP open-domain QA**：回答“在相同 KILT/Wikipedia 数据环境下，是否提升 NQ/Hotpot 等搜索问答”；
- **GraphTask certified OOD**：回答“是否真正学会可执行组合推理、必要性和抗 shortcut”。

前者主要报告 normalized EM、macro/micro、调用数、延迟；后者继续报告 answer F1/EM、程序族、跳数、necessity、shortcut 与工具成本。不能用 SSP 的答案 EM 取代 certified evaluation。

## 7. 建议的实验阶段

### 阶段 A：只对齐测试，成本最低

- 下载并固定 SSP 2,625；
- 使用现有模型加一个 KILT/E5 evaluation adapter；
- 先得到当前 Freebase/KQA Pro 训练模型在 SSP 上的零样本基线；
- 不改训练数据，用来判断主要瓶颈是知识源/检索工具不匹配还是推理能力。

### 阶段 B：KILT-corpus parity

- 构建 KILT hyperlink SQLite 图与 passage corpus；
- 仍用 GraphTask 自己的随机/结构约束 seed 做训练；
- 对比 `Freebase/KQA Pro → SSP` 与 `KILT → SSP`，测量知识源对齐收益。

### 阶段 C：public-protocol parity

- 加入 source-train answer/provenance 引导的 seed 分布；
- 训练上限 20,096、validation 512、314 steps；
- 最终 SSP 只跑一次；
- 与阶段 B 的差值隔离“训练题分布对齐”收益，而不是把全部提升归功于方法。

### 阶段 D：作者产物发布后再做 exact parity

- 按作者正式 manifest 重建；
- 比对逐文件 SHA-256、sample IDs 与来源比例；
- 再使用“exact dataset”措辞。

## 8. 验收标准

在进入完整训练前，数据层至少应满足：

| 项目 | 验收条件 |
|---|---|
| KILT raw | revision、URL、字节数、SHA-256 已记录；页面数 5,903,530 |
| 图导入 | 节点/边/段落计数稳定；同一 seed 两次构建哈希一致 |
| source manifest | 全部为原始 train split；逐源计数、ID 和拒绝原因齐全 |
| validation | 512；与 train、archive、SSP 均无 ID/文本重叠 |
| SSP | 固定 revision；原始 3,125；过滤后 2,625；bucket 计数精确 |
| gold | 训练生成任务只由认证程序执行得到，不从 source QA 直接拷贝 |
| 可回放 | 所有 RNG 显式 seed；环境状态 JSON 可序列化；外部检索有 timeout、retry、cache、trace ID |
| 测试 | ToyGraph/小型 KILT fixture 先通过，再跑大语料；新增行为有 unit + integration test |

## 9. 风险与决策

### 9.1 主要风险

1. **训练集身份不可复现**：这是上游公开材料缺失，不应由我们猜一个清单后隐藏。
2. **同语料收益被误写成推理收益**：KILT 训练 + KILT 检索 + Wikipedia benchmark 的提升可能主要来自知识源对齐。
3. **自然语言答案评测不等于实体 ID 评测**：必须显式处理 alias、冠词、标点、大小写、日期和数字格式。
4. **超链接不是类型化 KG 关系**：若所有边都简化为 `wiki_link`，GraphTask 的程序结构会变弱；应保留 anchor 文本、段落上下文，并明确这是 document graph。
5. **评测工具不匹配**：只给 SSP 模型 entity-ID graph search，会低估它；只给文本搜索又与当前训练工具不一致。
6. **数据污染**：source train 与 SSP 虽 split 不同，仍可能共享实体、答案、事实和近重复问法。

### 9.2 最终建议

建议批准的方案是：

> 先完成 L0（SSP exact test parity），再完成 L2（KILT public-protocol parity）；数据层对齐 CoEvoKG，训练方法继续保持 GraphTask-R1。正式报告中明确标注训练 manifest 缺失，不使用“训练数据完全相同”的表述。

这个顺序的代码改动最少，也能回答最重要的因果问题：

- 仅换评测环境，现有模型能否迁移？
- 换成 KILT 知识源后，提升有多少来自语料对齐？
- 再加入 source-train 引导 seed 后，提升有多少来自问题分布对齐？
- 在这些条件相同后，GraphTask 的认证任务生成与反事实必要性是否仍带来额外增益？

## 10. 主要依据

- CoEvoKG 论文定向证据读本：[coevokg_reader/paper.md](coevokg_reader/paper.md)
- [CoEvoKG 官方仓库](https://github.com/lazzy1225/CoEvoKG)
- [KILT 官方仓库与数据目录](https://github.com/facebookresearch/KILT)
- [KILT Wikipedia on Hugging Face](https://huggingface.co/datasets/facebook/kilt_wikipedia)
- [KILT tasks on Hugging Face](https://huggingface.co/datasets/facebook/kilt_tasks)
- [SSP test release](https://huggingface.co/datasets/Quark-LLM/SSP)
- [E5-base-v2 model card](https://huggingface.co/intfloat/e5-base-v2)
- [PopQA 官方仓库](https://github.com/AlexTMallen/adaptive-retrieval)
- [2WikiMultiHopQA 官方仓库](https://github.com/Alab-NII/2wikimultihop)
