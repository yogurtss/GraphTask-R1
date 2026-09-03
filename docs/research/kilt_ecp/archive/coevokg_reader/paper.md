# CoEvoKG 数据与评测证据读本（定向节选）

> Source: arXiv:2608.01904v1
>
> Version read: 2026-08-03 arXiv v1, 9-page PDF and experimental HTML
>
> Scope: 仅收录本次调研所需的训练数据、验证集与测试集段落；不是全文翻译
>
> Full report: [../COEVOKG_DATASET_RESEARCH_REPORT.md](../COEVOKG_DATASET_RESEARCH_REPORT.md)

## 索引

- S001–S002：初始图链池与链构造
- S003：seed fallback
- S004–S005：最终 benchmark 与隔离
- S006–S007：训练、检索与 checkpoint 选择

## 术语表

| Canonical term | 中文 | 说明 |
|---|---|---|
| evidence chain | 证据链 | 实体节点、关系标签和节点 passage 的组合 |
| chain pool | 链池 | 训练时供 proposer 采样的当前图记忆 |
| seed pool | 种子题池 | 生成题未通过质量门时的固定人工 QA fallback |
| held-out subset | 留出子集 | 只用于最终报告的 SSP benchmark 子集 |
| path support | 路径支撑 | Solver 轨迹中的相邻实体关系是否得到图证据支持 |

<a id="S001"></a>
**Source:** p.3, Section 3.1

**Original:** Chains are drawn from a pool populated by offline random walks over a Wikipedia-based knowledge graph (the KILT knowledge source). The initial pool is denoted by G0; path support verification and evidence write-back operate on the current pool Gt.

**中文:** 证据链来自一个通过离线随机游走构建的链池，其底层是基于 Wikipedia 的知识图，即 KILT knowledge source。初始池记为 G0；路径支撑验证和证据写回都作用于当前轮次的链池 Gt。

<a id="S002"></a>
**Source:** p.4, Section 3.2, “Evidence chain pool”

**Original:** The authors materialize 2–3 hop walks offline. Each walk yields article-level entities, typed natural-language relation labels, and an article passage at every node. Terminal entities likely to leak into questions are filtered, and at least two relational hops are required.

**中文:** 作者离线物化 2–3 跳随机游走。每条链包含文章级实体、自然语言关系标签，以及每个节点对应的文章段落。可能在问题中直接泄漏的终点实体会被过滤，并要求至少两个关系跳，以排除单跳捷径。

<a id="S003"></a>
**Source:** p.4, Section 3.2, “Seed fallback”

**Original:** If a chain produces no candidate that passes the quality gate, its empty solver slot is filled with a verified seed question from a fixed human-written QA set. The fallback receives no proposer difficulty signal. Its rate decreases from about 0.40 to about 0.25.

**中文:** 若某条链没有生成任何通过质量门的候选题，则用固定人工 QA 集中的已验证种子题填补 Solver batch 的空槽。该 fallback 样本不向 proposer 提供难度信号。训练过程中 fallback 比例约从 0.40 降至 0.25。

<a id="S004"></a>
**Source:** p.6, Section 4.1, “Benchmarks”

**Original:** Evaluation covers NQ, TriviaQA, and PopQA as single-hop open-domain QA, and HotpotQA, 2WikiMultiHopQA, and Bamboogle as multi-hop reasoning benchmarks. The final subsets come from the public SSP release: 500 questions per benchmark and 125 for Bamboogle.

**中文:** 评测使用 NQ、TriviaQA、PopQA 三个单跳开放域问答集，以及 HotpotQA、2WikiMultiHopQA、Bamboogle 三个多跳推理集。最终子集来自 SSP 的公开发布：前五个数据集各 500 题，Bamboogle 为 125 题。

<a id="S005"></a>
**Source:** p.6, Section 4.1, “Benchmarks”

**Original:** SSP evaluation subsets are used only for reporting. Training-time validation and seed fallback are split from source benchmark training data and exclude the SSP evaluation examples. The metric is normalized exact-match accuracy and its macro average.

**中文:** SSP 评测子集仅用于最终报告。训练时的验证集和 seed fallback 都从来源 benchmark 的训练数据中划出，并排除 SSP 评测实例。指标为归一化 exact-match accuracy 及其六数据集宏平均。

<a id="S006"></a>
**Source:** p.6, Section 4.1, “Retrieval and interaction”

**Original:** All methods use an E5 dense retriever over Wikipedia, with top-3 passages per query and up to eight solver turns. Training decoding uses temperature 0.6 and top-p 0.95; evaluation uses greedy decoding.

**中文:** 所有方法共享基于 Wikipedia 的 E5 dense retriever，每次查询返回 top-3 passage，Solver 最多交互 8 轮。训练解码使用 temperature 0.6、top-p 0.95；评测使用 greedy decoding。

<a id="S007"></a>
**Source:** p.6, Section 4.1, “Training”

**Original:** CoEvoKG, Search-R1, SSP, and all ablations receive the same budget of 314 optimization steps. The checkpoint with highest accuracy on a validation split drawn from training data is selected, then evaluated once on the SSP held-out subsets.

**中文:** CoEvoKG、Search-R1、SSP 和全部消融实验都使用相同的 314 个优化步预算。作者用从训练数据划出的验证集选择 accuracy 最高的 checkpoint，随后在 SSP 留出子集上只评测一次。

## 阅读提示

论文正文可以确认测试集名称、数量、隔离用途和总体训练预算，但不能确认正式训练 source manifest 的完整数据集名单、版本、来源占比和样本 ID。作者代码进一步显示 chain pool 的公开构造实现以训练题 gold answer 为 KILT 页面锚点，再进行反向随机游走；这一代码证据已在主调研报告中与论文正文分开标注。
