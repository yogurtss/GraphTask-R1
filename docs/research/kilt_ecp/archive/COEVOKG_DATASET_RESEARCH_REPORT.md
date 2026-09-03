# CoEvoKG 调研及 GraphTask-R1 数据方案改进建议

> 调研日期：2026-08-11
>
> 调研对象：CoEvoKG v1（arXiv:2608.01904，2026-08-03）及其公开代码
>
> 重点：训练数据集、验证集、测试集的选择与隔离；对 GraphTask-R1 的可迁移改进
>
> 证据口径：优先采用论文、作者代码仓库和公开评测数据；无法由公开材料确认的内容明确标为“未披露”或“推断”

## 1. 结论先行

CoEvoKG 与 GraphTask-R1 的共同母题确实很接近：二者都让任务生成者和搜索求解器共同演化，都利用图结构约束生成任务，并用当前 Solver 的成功率定义难度前沿。真正需要拉开差异的地方不是“KG + self-play”，而是验证强度和训练数据来源：

1. **CoEvoKG 是“路径支撑型”方法**：由 KG 实体链生成问题，以答案正确和搜索路径被图证据支持作为奖励；成功轨迹经验证、去重后写回图记忆。
2. **GraphTask-R1 应坚持“程序证书 + 反事实必要性”定位**：gold answer 由认证程序执行产生；删除 hop、filter、intersection branch 或 witness edge 后重执行，验证题目是否真的需要该结构。这比“路径存在/被支持”更强。
3. **CoEvoKG 最值得直接吸收的是持久化证据写回、seed fallback 监控、难度带课程和等预算实验协议**，而不是其 answer-anchored random walk 数据构造。
4. **GraphTask-R1 当前最大的实证短板是测试集尚未落到当前训练主线**。仓库当前实际训练只使用 KQA Pro train/val；WebQSP、CWQ、GrailQA 仍是计划中的主评测。若论文现在开跑，KQA Pro val 既选模型又报结果会导致明显的评测偏乐观。
5. **建议把数据协议固定为三层**：KQA Pro 仅做 SFT/工程冷启动；Freebase train-side 图区域做 self-play；WebQSP/CWQ/GrailQA 的官方 held-out split 做一次性最终评测，并增加严格的 entity/composition OOD 子集。

一句话判断：**思想邻近，但 GraphTask-R1 仍有清楚的可发表差异；前提是把“程序执行与反事实必要性”做成主证据，并补齐独立、不可反复调参的测试协议。**

## 2. 术语表

| 统一术语 | 含义 | 本报告用法 |
|---|---|---|
| proposer / Questioner | 生成训练问题的策略 | 谈 CoEvoKG 时用 proposer，谈本项目时用 Questioner |
| solver / Solver | 使用搜索或图工具回答问题的策略 | 保留 Solver |
| graph memory | 带节点文本、边关系和写回证据的动态链池 | 指 CoEvoKG 的可演化外部记忆 |
| certified program | 可在固定图快照上执行并产生 gold answer 的程序 | 指 GraphTask-R1 的类型化程序证书 |
| path support | 搜索轨迹中的相邻实体是否被图证据支持 | 不等同于结构必要性 |
| structural necessity | 删除结构组件后答案是否变化、是否存在低成本 shortcut | GraphTask-R1 的核心验证信号 |
| held-out | 不参与训练、fallback、验证、生成或图链构造的样本 | 最终报告集 |

## 3. CoEvoKG 方法概览

CoEvoKG 将同一份 KG 链池同时用作：

- 生成可验证多跳问题的任务源；
- 验证 Solver 搜索路径的证据源；
- 接收成功搜索证据写回的持久化记忆。

每一轮的主要流程为：

```text
当前图记忆 G_t
  -> 采样 2–3 hop 实体链
  -> proposer 每条链生成 M 个候选题
  -> deterministic leakage filter + frozen LLM quality gate
  -> 保留 top-k 候选；失败槽位由训练侧 seed QA fallback
  -> Solver 每题做 G 次、多至 8 轮的 Wikipedia 搜索
  -> answer correctness × (1 + beta × path support)
  -> proposer 根据 Solver 成功率获得 bell-shaped frontier reward
  -> 正确且路径受支持的证据经验证、去重后写回 G_{t+1}
```

其三个递进组件在 Qwen2.5-7B 上的宏平均消融增益为：KG task generation `+1.5`，path/difficulty reward `+1.1`，write-back `+0.6`。这说明写回的效果为正但不是主要增益来源；最大的单项收益仍来自受 KG 约束的任务生成。

### 3.1 与 GraphTask-R1 的关键相同点和差异

| 维度 | CoEvoKG | GraphTask-R1 | 判断 |
|---|---|---|---|
| 双角色 | proposer + solver 联合训练 | Questioner + Solver，共享 backbone/LoRA | 高度相似，不能单独作为创新 |
| 任务来源 | 预物化 KG 实体链 | Questioner 主动图探索并构造程序 | 本项目更主动 |
| gold 来源 | 链中指定实体/训练题 answer anchor | 认证程序在固定图快照上实际执行 | 本项目可验证性更强 |
| 质量门 | deterministic leakage + LLM verifier | schema、可执行性、answer leakage、shortcut、反事实必要性 | 本项目应强调无需 LLM 决定 gold |
| Solver | 多轮文本搜索 | 有限 KG Search/GraphScript 工具 | 任务环境不同 |
| 过程奖励 | answer-correct 后乘 path support | answer F1 + frontier + necessity + novelty | 可加入轨迹—证书一致性，但不应替代 necessity |
| 持久记忆 | 成功证据写回节点/边，供后续轮次复用 | accepted task archive，未把 Solver 新证据变成图 overlay | CoEvoKG 可直接启发改进 |
| 隔离 | 训练 split 链池；SSP held-out 只最终报告 | 已规划 held-out entity denylist 与多维污染控制 | 本项目规划更严格，但尚需产物化 |

## 4. CoEvoKG 的训练数据选择

### 4.1 训练数据实际上由三部分组成

| 输入 | 作用 | 来源和选择 | 是否动态变化 |
|---|---|---|---|
| `CHAIN_DATA_NAS` | proposer 的任务模板、path support 证据库 | 由 source benchmark **train split** 的答案在 KILT/Wikipedia 中定位，再做 answer-anchored random walk；清洗后仅保留 train 记录 | 是，训练中写回新证据/新链 |
| `DATA_PATH` | 当某条链没有候选题通过质量门时的 seed/fallback QA | source benchmark 的训练划分，预先过滤 | 否；使用率随训练下降 |
| `TEST_DATA_PATH` | 训练时选 checkpoint 的验证集 | 从 source benchmark 训练数据中另行切出的 validation pool | 否，公开配置默认最多读取 512 条 |

必须注意，论文正文“offline random walks over a Wikipedia-based KG”容易让人理解成纯图无监督采样；公开代码给出的更精确过程是：

1. 从一个只允许 `split=train` 的 manifest 中按各数据集池大小比例抽样；
2. 读取问题及其 gold answer；
3. 用 gold answer 搜索对应 KILT Wikipedia 页面；
4. 从答案页面开始随机游走，再反转路径，使答案位于链尾；
5. 清洗时删除原始 QA 字段，仅保留实体序列、节点文本和关系标签；
6. 训练时从该链生成新的多跳问题。

因此，CoEvoKG 的主要生成题不是复述原训练问题，但**它使用了训练题的 gold answer 来锚定链池分布**。这是一种弱监督/远程监督式的 task source，不宜描述成完全不依赖人工 QA。

### 4.2 图与检索语料选择

- 图源和检索语料都来自 KILT Wikipedia knowledge source；节点是 Wikipedia article，边来自页面 inline anchor hyperlink。
- 初始池使用 2–3 hop 链；训练配置中 proposer 也限制为 2–3 hop。
- E5 dense retriever 在同一 Wikipedia/KILT 语料上返回 top-3 passage；Solver 最多搜索 8 轮。
- 这种选择使“生成图、路径验证图、检索语料”高度对齐，优点是训练稳定、证据可追踪；缺点是不能证明对另一知识源、另一快照或结构化 KG 的迁移。

### 4.3 规模与训练预算

公开配置和正文可以对齐出以下数字：

| 项目 | 数值 |
|---|---:|
| 训练样本上限 | 20,096 |
| 训练验证样本上限 | 512 |
| train batch size | 128 |
| 训练 epoch | 2 |
| 总优化步数 | 314 |
| 每题 Solver rollout group size | 8 |
| proposer 每链候选数 | 4 |
| 质量门后 top-k | 2 |
| 质量阈值 | 0.45 |
| 初始候选链数/step | 32 |
| 新写回链上限 | 50,000 |

`20,096 / 128 × 2 = 314`，与论文所述所有方法统一 314 optimization steps 一致。等优化步数是好做法，但论文仍应同时报告实际 rollout token、检索调用和 verifier API 成本；不同方法每一步的候选生成与验证开销并不相同。

### 4.4 训练集选择中公开材料没有说明清楚的部分

下列信息在论文、README 和当前公开仓库中均无法完整复原：

- 构建正式训练 chain pool 时究竟纳入了哪些 source benchmark；
- 每个来源的原始版本、revision、原始行数、抽样后行数和 sample ID；
- seed/fallback pool 与 chain anchor pool 是否完全同源、是否共享实例；
- train-derived validation 的逐数据集构成和抽样 seed；
- 正式实验使用的 chain pool 规模、去重前后规模及来源占比；
- 用于生成正式数据的 manifest 文件和 SHA-256。

代码提供了 HotpotQA、2Wiki 类格式、MuSiQue、NQ、通用 QA 和 Bamboogle loader，但 loader 的存在不能证明正式实验全部使用了这些数据。因此，本报告不把 loader 列表当作正式训练集名单。

这是 CoEvoKG 数据可复现性中最明显的缺口。GraphTask-R1 应主动把这一项做成优势：训练与测试都发布不可变 manifest、来源 revision、split、sample ID 和哈希。

## 5. CoEvoKG 的测试数据选择

### 5.1 最终测试集构成

论文最终报告六个 open-domain QA benchmark，来自 `Quark-LLM/SSP` 公开测试文件中的留出子集：

| 类型 | 数据集 | 样本数 | 主要能力 |
|---|---|---:|---|
| 单跳/开放域 | Natural Questions (NQ) | 500 | 真实用户问题、事实检索 |
| 单跳/开放域 | TriviaQA | 500 | trivia 型实体事实 |
| 单跳/长尾 | PopQA | 500 | 实体流行度/长尾事实鲁棒性 |
| 多跳 | HotpotQA | 500 | 两文档组合推理 |
| 多跳 | 2WikiMultiHopQA | 500 | 跨 Wikipedia 页面多跳推理 |
| 小型困难集 | Bamboogle | 125 | 搜索引擎不易直接命中的手工多跳题 |
| **合计** |  | **2,625** |  |

SSP 的公开 `test.jsonl` 实际还有 500 条 MuSiQue，因此文件总计 3,125 条；CoEvoKG 明确只选上述六组，排除了 MuSiQue。这个选择本身可以接受，但论文未解释为什么排除 MuSiQue。

### 5.2 测试协议

- 六个 SSP 子集不参与训练、validation、chain construction、proposer generation 或 fallback。
- 每种方法使用同一 E5/KILT retrieval stack、每次 top-3、最多 8 turns。
- 训练中用 train-derived validation 选最高 accuracy checkpoint，然后只在 SSP held-out 上评测一次。
- 推理采用 greedy decoding；主要指标是答案归一化后的 exact-match accuracy。
- 主表报告六个数据集的等权宏平均。

### 5.3 选择的优点

1. 同时覆盖单跳、长尾事实和多跳，能显示训练收益是否只来自路径长度。
2. 使用与 Search-R1、SSP 一致的公开子集，使基线预算和题目对齐更容易。
3. 最终集不参与 checkpoint selection，协议上比直接在 benchmark dev 上反复调参更干净。
4. 三个不同大小的 backbone 使用同一测试矩阵，能观察收益是否依赖模型规模。

### 5.4 选择的局限和潜在偏差

1. **并非严格跨数据集泛化。**训练 chain anchor 和 fallback 来自 source benchmark 的训练划分，而最终测试来自多个同名 benchmark 的留出题。即使实例不重合，实体、事实、问法和数据集风格仍可能重合。
2. **并非严格图 OOD。**训练图、检索库和 path verifier 都基于 KILT/Wikipedia；测试也主要依赖 Wikipedia 事实。结果证明的是同知识源内的搜索泛化，而不是跨 KG/跨快照泛化。
3. **Bamboogle 被宏平均放大。**它只有 125 题，却与每个 500 题数据集各占宏平均的 `1/6`。一个 Bamboogle 样本会改变该列 0.8 个百分点，而 500 题集每题只改变 0.2 个百分点。
4. **宏平均让收益看起来略高于按题目数加权的结果。**由主表和公开样本数计算：

   | Backbone | 论文宏平均增益 | 2,625 题 micro 增益 |
   |---|---:|---:|
   | Qwen2.5-3B | +11.2 | +10.0 |
   | Qwen2.5-7B | +10.1 | +10.0 |
   | Llama-3.1-8B | +11.6 | +10.8 |

   这不推翻结论，但主表最好同时给 macro 和 micro，并给各数据集置信区间。
5. **评测只看答案 EM。**它不能直接确认搜索路径正确、检索成本合理或证据足以支持答案。论文训练时强调 path support，最终测试却没有报告 trajectory faithfulness 或 citation/evidence recall。
6. **缺少训练—测试污染审计产物。**作者声明 SSP held-out 未被使用，但没有发布 source manifest、ID denylist、文本近重复报告或实体重叠矩阵，第三方难以复核。

## 6. GraphTask-R1 当前数据方案审计

这里必须区分“当前可运行主线”和“研究计划中的主实验”，否则容易高估完成度。

### 6.1 当前可运行主线

当前 README 明确规定：

- 图快照：KQA Pro `kb.json` 转换得到的 `kqapro-v1/graph.sqlite`；
- SFT train：KQA Pro train 中通过程序映射和认证的任务；
- SFT validation：KQA Pro val 中通过认证的任务；
- Solver GRPO train：同一 KQA Pro train 的 Solver role 导出；
- Solver GRPO validation：同一 KQA Pro val 的 Solver role 导出；
- KQA self-play：base pool 来自 KQA Pro train，Questioner 从独立实体 seeds 出发，Solver batch 按 `base/archive/new = 0.35/0.35/0.30` 混合；
- 当前没有在主训练 README 中落地一个独立于 KQA Pro val 的最终 test set。

优点是 gold answer 只由 certified program 执行产生，且转换过程检查 source answer、一致性、shortcut、answer leakage、反事实必要性和 trace replay。这个训练标签质量强于 CoEvoKG 的 answer string + LLM quality gate。

主要问题是：

- 同一个 KQA Pro val 很可能承担早停、选 checkpoint、调超参和阶段验收；它不应再作为最终论文主结果。
- KQA Pro 冷启动数据与未来 Freebase 主实验存在 schema、实体 ID、关系和问题风格迁移。
- 当前主线尚不能支持“提升独立人工 KGQA benchmark”的核心论文结论。

### 6.2 已规划但尚需产物化的主评测

项目计划使用 Freebase + WebQSP/CWQ/GrailQA：

- WebQSP：标准 Freebase KGQA；
- CWQ：复杂组合问题；
- GrailQA：IID、compositional、zero-shot 三类泛化；
- dev/test topic entities 合并为 self-play seed denylist；
- 训练侧只能抽取 train split 的 seeds、relation whitelist、程序模板统计和 verbalization examples；
- 最终同时报告原 benchmark gold、当前图重执行 denotation、entity-set F1 和人工审计。

这套规划总体优于 CoEvoKG 的公开隔离协议，但目前存在两个工程风险：

1. benchmark adapter 按文件名推断 split；若下载目录混入备份、派生文件或命名异常文件，可能被标成 `unknown` 后仍写入 processed data。应改成显式 manifest，不再依赖文件名猜 split。
2. 现有 denylist 只聚合 dev/test `topic_entity_ids`。它无法阻止文本近重复、答案实体重合、程序签名重合、关系组合重合，也不能证明 archive 中没有通过中间节点触及 held-out 题的关键证据。

## 7. 建议的数据集重构方案

### 7.1 推荐的三层数据角色

| 层 | 数据 | 允许用途 | 禁止用途 |
|---|---|---|---|
| A：冷启动 | KQA Pro train | DSL/GraphScript SFT、格式和工具轨迹冷启动 | 作为最终主结论；使用 KQA Pro test 反向生成训练数据 |
| B：主训练 | Freebase 图 + benchmark train-side seeds/statistics + 纯图采样 seeds | Questioner self-play、Solver RL、archive/memory | 读取 WebQSP/CWQ/GrailQA dev/test question、LF、SPARQL、answers |
| C：最终评测 | WebQSP test、CWQ test、GrailQA dev buckets/官方 test | 一次性最终报告 | checkpoint selection、reward、fallback、relation mining |

另从层 B 的 train split 固定切出两个互不重叠的集合：

- `train_validation`：每 N 步做早停和 checkpoint selection；
- `development_audit`：只用于人工抽查、错误分析和 prompt 调试，不参与自动选模型。

### 7.2 必须生成的不可变 manifest

建议新增 `data/manifests/<experiment_id>/`，至少保存：

```text
source_manifest.json
train_ids.jsonl
train_validation_ids.jsonl
development_audit_ids.jsonl
final_test_ids.jsonl
heldout_topic_entities.json
heldout_answer_entities.json
text_dedup_report.json
program_signature_overlap.json
entity_relation_overlap.json
graph_snapshot_manifest.json
```

每份 manifest 应记录：官方 URL、许可、下载日期、revision/commit、文件 SHA-256、split、原始行数、过滤理由计数、最终 sample IDs、随机种子、转换器版本和图快照 hash。发布论文时把 manifest 与配置一并归档。

### 7.3 四种隔离应真正执行，而不只写在计划里

1. **实例隔离**：官方 ID 不重合；所有派生任务保留 source lineage。
2. **文本隔离**：规范化 exact hash、MinHash/n-gram、embedding 相似度三层审计；接近 held-out question 的生成题进入结构化 rejection，而非静默删除。
3. **实体隔离**：默认设置禁止 held-out topic entity 成为 Questioner seed 或目标答案；额外报告严格 unseen-entity 子集。中间实体重合只审计和分桶，不建议全部禁止，否则可能破坏 Freebase 连通性。
4. **结构隔离**：按 canonical program signature、relation tuple 和 composition family 分桶。GrailQA zero-shot/compositional 的官方定义优先，不要用自行重切分覆盖官方可比性。

### 7.4 测试集建议

主表保持 KGQA 场景一致，不建议直接照搬 CoEvoKG 的六个 open-domain 文本搜索集：

| 主表列 | 建议指标 |
|---|---|
| WebQSP test | entity F1、Hits@1、search calls |
| CWQ test | entity F1、EM、search calls |
| GrailQA IID | entity F1、exact match |
| GrailQA compositional | entity F1、exact match |
| GrailQA zero-shot | entity F1、exact match |
| structural-hard set | necessity pass、shortcut rate、intervention drop |

补充表再报告：KQA Pro unseen-program/unseen-entity、GraphWalkerBench 或 KGQAGen/Dynamic-KGQA。这样可同时证明标准 benchmark utility 和生成任务的结构泛化。

所有主结果至少同时给：

- macro average 和按题目数加权 micro；
- 三个显式训练种子；
- paired bootstrap 95% CI；
- 每题 search calls、edge visits、tokens 和 wall time；
- 原 benchmark gold 与固定 Freebase snapshot 上重新执行的 denotation 两套口径；
- 失败原因分解，而不只给总 F1。

## 8. 从 CoEvoKG 吸收哪些改进

### P0：把当前 archive 升级为“可验证证据 overlay”，但不要污染认证图

CoEvoKG 的 write-back 很适合迁移，但 GraphTask-R1 不应直接改写产生 gold 的基础 KG。建议使用两层图：

```text
immutable certified KG snapshot
  + append-only evidence overlay
      - source trajectory id
      - source checkpoint/round
      - retrieved fact or path
      - verifier result
      - confidence and timestamp
      - dedup key
      - replay hash
```

- certified program 只在 immutable base graph 上产生 gold；
- overlay 只用于 Questioner 采样优先级、Solver 检索提示、novelty 和 path alignment；
- 新证据若要晋升为可执行事实，必须经过独立 schema/type/checksum 认证并产生新 graph snapshot ID；
- 写回失败、冲突、重复、过期都保留结构化 reason code。

这样既吸收 CoEvoKG 的持续知识积累，又不破坏本项目“gold 可重放”的核心可信性。

### P0：把 train/val/test 隔离变成代码强约束

- `data prepare` 改为只接受显式 manifest；出现 `unknown` split 直接失败。
- `sample-seeds` 同时接收 topic、answer、text hash 和 program-signature denylist。
- archive writer 在提交时再次运行 contamination gate，防止生成阶段的绕过。
- 最终 evaluation 命令检查 checkpoint manifest 中从未加载 final-test hashes。

### P1：引入 CoEvoKG 式 difficulty band curriculum

当前目标 pass rate 固定为 0.5。建议比较：

- fixed frontier：`p*=0.50`；
- annealed frontier：`0.60 -> 0.40`；
- operator-conditioned frontier：intersection/count/filter 分别维护成功率目标；
- uncertainty-aware frontier：用 Beta posterior/置信区间替代 8 次 rollout 的点估计。

只有 8 次 rollout 时，`p_s` 只能取 0、0.125、…、1，点估计噪声较大。可以令 reward 依赖 posterior mean，并对置信区间过宽的候选降低权重。

### P1：加入 Solver 轨迹与 certified witness 的一致性奖励

在答案正确之后再计算：

- Solver 访问的实体/边是否覆盖 certified witness；
- 是否走了已知 shortcut；
- path precision/recall；
- 证据是否足以执行同一 denotation。

推荐形式：

```text
R_solver = R_answer × (1 + beta × R_witness_alignment)
           - lambda_cost × normalized_tool_cost
```

`R_witness_alignment` 只能作为正确答案后的附加项，避免奖励“路径看似合理但答案错误”。它是 CoEvoKG path support 的程序化增强版。

### P1：保留并显式记录 seed fallback

当 Questioner 某个 batch 没有合格候选时，从固定 train-side certified base pool 回填，并记录：

- fallback rate；
- 按 operator、hop、round 的 fallback rate；
- fallback 题与生成题的成功率、结构复杂度和训练占比。

若 fallback 长期高于 30% 或连续两轮不降，应视为 Questioner/quality gate 未学会，而不是继续扩大训练。

### P2：严格等预算对比

除相同 optimization steps 外，至少匹配或报告：

- accepted tasks；
- proposer tokens；
- Solver rollout tokens；
- graph/search calls；
- verifier calls；
- wall-clock GPU hours。

CoEvoKG 的“相同 314 步”值得采用，但 GraphTask-R1 的反事实执行成本更高，只匹配步数不足以回答“收益来自方法还是更多验证计算”。

## 9. 建议的最小实验矩阵

| 实验 | 训练任务来源 | 记忆 | 验证 | 目的 |
|---|---|---|---|---|
| E0 | KQA Pro static | 无 | executable only | 当前冷启动基线 |
| E1 | Freebase random walk | archive | path exists | 对齐 CoEvoKG/path proposer |
| E2 | active program | archive | executable | 测主动程序构造 |
| E3 | active program | archive | executable + necessity | 测反事实必要性 |
| E4 | active program | evidence overlay | executable + necessity + witness alignment | 完整方法 |
| E5 | E4，但固定 difficulty | evidence overlay | 同 E4 | 测 frontier curriculum |
| E6 | E4，但不使用 benchmark-train QA answer anchor | evidence overlay | 同 E4 | 量化对人工 QA seed 的依赖 |

每个实验固定相同 accepted task 数、Solver rollout 数和最大图调用预算。主结果只看 WebQSP/CWQ/GrailQA；自生成任务上的成功率只能作为诊断指标。

最关键的新增消融是 E6：它能直接回答 GraphTask-R1 相对 CoEvoKG 的数据优势——提升是否来自真正的 active graph discovery，而不是用 benchmark train answers 把生成分布预先锚到测试任务附近。

## 10. 预期论文叙事调整

建议把最近工作边界写成：

> CoEvoKG demonstrates that KG chains can jointly ground self-generated search tasks and retain verified evidence across training rounds. GraphTask-R1 addresses a different limitation: path-supported tasks may still admit shortcuts or fail to require the claimed composition. We therefore replace chain-conditioned task synthesis with active executable program construction and certify structural necessity through counterfactual graph interventions, while retaining an append-only evidence memory for curriculum evolution.

中文要点：

- 不争“第一个 KG self-play”或“第一个共进化图记忆”；
- 明确承认 CoEvoKG 已覆盖 KG chain generation、frontier difficulty 和 evidence write-back；
- 主创新集中到 active program construction、program-executed gold、counterfactual necessity 和 asymmetric graph tools；
- 数据实验突出不依赖 benchmark-train gold answer anchor 的版本；
- 评测突出 GrailQA compositional/zero-shot 与 structural-hard intervention set。

## 11. 证据强度与待确认项

| 结论 | 证据强度 | 依据 |
|---|---|---|
| 六个最终 benchmark 与样本数 | 高 | 论文正文 + SSP `test.jsonl` 实际计数 |
| SSP 测试文件另含 500 MuSiQue、CoEvoKG 排除它 | 高 | SSP 文件实际计数 + 论文六项列表 |
| 正式训练预算为 20,096、512、314 steps | 高 | 公开配置 + 正文 |
| chain pool 由 train QA answer 锚定 KILT 随机游走 | 高 | 作者公开预处理代码 |
| 正式训练具体用了哪些 source benchmark | 低/未披露 | 正式 manifest 和数据未发布，不能从 loader 推断 |
| held-out 完全无语义/实体污染 | 未验证 | 有作者声明，但缺少可复核的 overlap audit |
| GraphTask-R1 当前只跑 KQA Pro 主线 | 高 | 当前 README、训练配置和数据手册 |
| WebQSP/CWQ/GrailQA 已完成主结果 | 否 | 目前是计划和 adapter，仓库内无结果产物 |

## 12. 参考资料

1. Li, Z. et al. *CoEvoKG: Co-Evolving Knowledge Graphs with Self-Evolving Search Agents*. arXiv:2608.01904 (2026). <https://arxiv.org/abs/2608.01904>
2. CoEvoKG official repository, inspected at commit `384fa6839dc406faa3f334ef72e06fb0dfd51924`. <https://github.com/lazzy1225/CoEvoKG>
3. Quark-LLM/SSP public dataset. <https://huggingface.co/datasets/Quark-LLM/SSP>
4. Petroni, F. et al. *KILT: a Benchmark for Knowledge Intensive Language Tasks*. <https://github.com/facebookresearch/KILT>
5. Kwiatkowski, T. et al. *Natural Questions: a Benchmark for Question Answering Research*. <https://aclanthology.org/Q19-1026/>
6. Joshi, M. et al. *TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension*. <https://aclanthology.org/P17-1147/>
7. Mallen, A. et al. *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric Memories*. <https://aclanthology.org/2023.acl-long.546/>
8. Yang, Z. et al. *HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering*. <https://aclanthology.org/D18-1259/>
9. Ho, X. et al. *Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps*. <https://aclanthology.org/2020.coling-main.580/>
10. Press, O. et al. *Measuring and Narrowing the Compositionality Gap in Language Models*. <https://aclanthology.org/2023.findings-emnlp.378/>
11. Shi, J. et al. *KQA Pro: A Dataset with Explicit Compositional Programs for Complex Question Answering over Knowledge Base*. <https://aclanthology.org/2022.acl-long.422/>
12. Gu, Y. et al. *Beyond I.I.D.: Three Levels of Generalization for Question Answering on Knowledge Bases*. <https://aclanthology.org/2021.webnlg-1.23/>

## 13. 建议下一步

1. 先实现显式 split manifest 和多维 contamination audit；这是投入最低、对论文可信度提升最大的改动。
2. 再把 archive 扩展为 append-only evidence overlay，保持 certified KG 不可变。
3. 用 ToyGraph/KQA Pro 做 `no-memory / archive / evidence-overlay` 三组单元与小规模集成实验。
4. 通过工程 gate 后再在 Freebase 上跑 E1–E4，并冻结最终测试脚本和 test hashes。
5. 首轮主表优先完成 WebQSP、CWQ、GrailQA 三套评测，不要先扩到 CoEvoKG 的 open-domain 六数据集。
