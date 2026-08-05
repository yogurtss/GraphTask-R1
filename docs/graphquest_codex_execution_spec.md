# GraphQuest-Code：完整 Codex 执行规格

> 本文件是结构化实现包的单文件合并版。配置、JSON Schema 和 prompts 请使用配套 ZIP 中的原文件。

---


<!-- SOURCE: README.md -->

# GraphQuest-Code：Codex 实现规格包

> **目标**：训练一个基于 Qwen3.5-4B + LoRA 的图探索策略，使其在有限图访问预算下生成可执行的图程序，主动发现有效、具有挑战性的问题。

本目录不是论文正文，而是一份面向 Codex 的工程契约。实现时应严格遵循版本顺序，不要一次性加入自博弈、Utility Critic、视觉节点、复杂查询图等后续模块。

## 1. 首轮唯一需要回答的问题

在同一个 Mini-Wikidata 图、相同种子实体、相同 primitive graph budget、相同问题模板和相同冻结 Solver 下：

> **GraphScript + GRPO 是否比 Action-ID、随机游走和 SPARK-style 启发式游走，更高效地发现有效且位于 Solver 能力边界的问题？**

主指标：

```text
Frontier Discovery Efficiency
= 有效 frontier questions / 1,000 primitive edge visits
```

辅助指标：程序编译率、执行成功率、有效问题率、Solver pass rate 分布、LLM tokens、LLM calls、执行时间，以及 Standard / High-Branching 两个 split 的结果。

## 2. 首轮冻结的设计决定

- 基座模型：`Qwen/Qwen3.5-4B`。
- Explorer：基座模型 + `explorer_lora`，先 SFT，再 GRPO。
- Solver：同一基座的独立冻结实例；首轮不训练 Solver。
- 首轮只做文本图，不输入图像；保留未来多模态接口。
- 图数据：先从 `lianglz/KGQAGen-10k` 的 train split `proof` 三元组并图，快速形成沙盒。
- 查询结构：只做 2-hop chain。
- 答案：唯一实体答案。
- Verbalizer：白名单关系的确定性模板；不把自然语言生成作为首轮变量。
- GraphScript：受限、类型化 JSON AST；绝不执行任意 Python。
- 奖励：硬有效性门控 + frontier reward - graph cost；不叠加 novelty、coverage、IRT、utility 等奖励。
- 框架：Transformers + PEFT + TRL；rollout 可接独立 vLLM server。
- 完整 Wikidata、Virtuoso/QLever、veRL、自博弈和视觉扩展全部后置。

## 3. 交互模式

### Action-ID 基线

模型每次只能选择一个候选动作：

```json
{"action_id": "A03"}
```

环境执行一个原子图操作，再返回新状态。

### GraphScript 主方法候选

模型生成一个受限程序：

```json
{
  "version": "0.1",
  "ops": [
    {"op": "start", "entity": "$seed", "out": "h0"},
    {"op": "follow", "in": "h0", "relation": "P57", "direction": "out", "limit": 16, "out": "h1"},
    {"op": "follow", "in": "h1", "relation": "P19", "direction": "out", "limit": 16, "out": "h2"},
    {"op": "require_unique", "in": "h2"},
    {"op": "emit", "in": "h2"}
  ]
}
```

Executor 确定性执行程序并返回答案、证据路径和成本。GraphScript 与 Action-ID 必须使用同一组 primitive operators 和同一 edge-visit 预算。

## 4. 执行顺序

1. **P0 数据与环境跑通**：KGQAGen proof → Mini Graph → 2-hop task → 模板问题 → 冻结 Solver。
2. **P1 非学习基线**：Uniform、Relation-balanced、SPARK-style。
3. **P2 Action-ID 原型**：Frozen Qwen、LoRA SFT、LoRA GRPO。
4. **P3 GraphScript 原型**：Parser/Executor、LoRA SFT、LoRA GRPO。
5. **P4 严格等预算比较**：Standard 与 High-Branching split，3 个随机种子。
6. **P5 Go/No-Go**：核心假设成立后，才进入交替 self-play。

## 5. 目录说明

- `CODEX_EXECUTION_PROMPT.md`：可直接复制给 Codex 的总任务说明。
- `docs/`：研究背景、架构、数据、GraphScript、训练、实验和验收标准。
- `configs/`：建议的 smoke/MVP 配置。
- `schemas/`：数据与程序的 JSON Schema 契约。
- `prompts/`：Explorer、Solver、Verbalizer 的初始提示词。

## 6. 推荐命令接口

Codex 应实现如下 CLI；内部细节可调整，但命令语义不得改变：

```bash
python -m graphquest.cli prepare-data --config configs/smoke.yaml
python -m graphquest.cli build-graph --config configs/smoke.yaml
python -m graphquest.cli generate-candidates --method uniform --config configs/smoke.yaml
python -m graphquest.cli generate-candidates --method spark_weighted --config configs/smoke.yaml
python -m graphquest.cli calibrate-solver --config configs/smoke.yaml
python -m graphquest.cli build-sft-data --mode action_id --config configs/mvp_action_id.yaml
python -m graphquest.cli build-sft-data --mode graphscript --config configs/mvp_graphscript.yaml
python -m graphquest.cli train-sft --config configs/mvp_graphscript.yaml
python -m graphquest.cli train-grpo --config configs/mvp_graphscript.yaml
python -m graphquest.cli evaluate --config configs/mvp_graphscript.yaml
python -m graphquest.cli compare --runs outputs/runs/* --output outputs/report.html
```

## 7. 不允许 Codex 自行加入的内容

首轮禁止：

- 任意 Python `exec/eval`；
- 完整 Wikidata 下载；
- Neo4j、Virtuoso、QLever；
- LangGraph；
- Solver 训练或双角色同步更新；
- intersection、comparison、count、temporal、negation；
-视觉输入；
- Utility Critic、Solver Pool、coverage memory、IRT；
- LLM Judge 作为答案正确性的主要判据；
- 将多个奖励简单相加以追求结果。

先证明程序化交互在本任务中有效，再扩展。

---

<!-- SOURCE: CODEX_EXECUTION_PROMPT.md -->

# 给 Codex 的执行提示词

你需要实现 `GraphQuest-Code` 的首轮可证伪 MVP。请先阅读本目录所有文档，尤其是：

1. `README.md`
2. `docs/01_mvp_scope.md`
3. `docs/02_architecture.md`
4. `docs/03_graphscript_spec.md`
5. `docs/04_data_preparation.md`
6. `docs/05_training_and_rewards.md`
7. `docs/06_experiments.md`
8. `docs/07_acceptance_criteria.md`

## 总原则

- 先实现最小闭环，再训练模型。
- 每个阶段必须有单元测试、可复现实验和明确输出。
- 不要实现文档中标为“后续”的模块。
- 不允许执行模型生成的任意 Python；GraphScript 必须解析为白名单 AST。
- 所有图访问必须经过统一 BudgetMeter，不能有绕过预算的 API。
- Action-ID 与 GraphScript 必须共享相同 GraphBackend、primitive operators、数据 split、Verbalizer、Solver 和评估代码。
- 所有外部网络访问必须封装在数据下载阶段；训练与评估必须可离线运行。
- 所有随机过程必须接受 seed，并在 manifest 中记录配置、Git commit、数据哈希和模型 revision。

## 实现阶段与提交边界

### Commit 1：项目脚手架与契约

实现：

- `pyproject.toml`
- `src/graphquest/` 包结构
- Pydantic 数据模型
- 配置加载
- CLI 空命令
- JSONL logging
- pytest 基础设施

验收：

- `pytest -q` 可运行；
- 所有 schema 有 round-trip 测试；
- 配置错误能给出清楚异常。

### Commit 2：数据准备

实现 KGQAGen-10k adapter：

- 下载/加载 train、dev、test；
- 解析 `proof` 中的 QID/PID；
- 只使用 train proof 构建训练图；
- 按 seed QID 做 train/dev/test 隔离；
- 输出 Parquet 与 manifest；
- 支持 smoke 与 mvp 两种采样规模。

验收：

- 不读取 test 问题文本生成训练任务；
- 每条 triple 保留来源 sample id；
- 数据哈希稳定；
- smoke 图可在 CPU 内存中加载。

### Commit 3：GraphBackend、BudgetMeter 与任务执行器

实现：

- 内存邻接表后端；
- 正向/反向关系访问；
- primitive edge visit 计数；
- 2-hop chain 执行；
- unique-answer gate；
- no-shortcut gate；
- provenance/support trace。

验收：

- 相同输入得到完全相同输出；
- 达到预算时立即终止；
- 任意调用路径都不能绕过 BudgetMeter；
- 构造至少 500 个有效 2-hop 任务。

### Commit 4：Verbalizer、Solver 与校准

实现：

- 关系模板配置；
- canonical question；
- local graph context：support + 固定数量 distractors；
- 冻结 Qwen3.5-4B Solver client；
- 结构化答案解析与 alias exact match；
- K 次采样成功率与缓存。

验收：

- 同一任务的 context 在固定 seed 下稳定；
- Solver 输出无法解析时记为错误，不调用 LLM judge；
- 找到使候选任务通过率主要位于 0.2–0.8 的 solver/config 组合。

### Commit 5：非学习基线与 Action-ID

实现：

- Uniform walk；
- Relation-balanced walk；
- SPARK-style weighted walk；
- Frozen Qwen Action-ID；
- Action-ID SFT trace generator；
- Action-ID LoRA SFT。

验收：

- 所有方法在相同 episode seeds 和 graph budget 下运行；
- 输出统一 `GeneratedTask`；
- 记录 edge visits、LLM calls、tokens 和 wall time。

### Commit 6：GraphScript Parser/Executor

实现 `GraphScript v0.1`：

- JSON AST parser；
- `start/follow/require_unique/emit`；
- handle 类型检查；
- relation allowlist；
- op 数、limit 和预算限制；
- pretty printer；
- 语法错误分类。

验收：

- fuzz 测试不能触发任意代码执行；
- 不合法 relation、handle、direction、limit 必须拒绝；
- 程序执行 provenance 与等价 Action-ID 轨迹一致；
- 预算统计一致。

### Commit 7：GraphScript SFT

实现：

- 从 Uniform、Relation-balanced 和 SPARK-style 有效轨迹生成 1,000–3,000 条 GraphScript SFT 样本；
- Qwen3.5-4B + PEFT LoRA；
- 关闭 thinking；
- 只输出 JSON；
- checkpoint 保存/恢复。

验收：

- held-out seeds 上 JSON parse rate ≥ 95%；
- executable rate ≥ 90%；
- budget violation = 0；
- 输出长度符合限制。

### Commit 8：GRPO

实现：

- 自定义 reward；
- valid gate；
- frontier reward；
- primitive graph cost；
- solver batch/caching；
- Action-ID 和 GraphScript 共用 reward service；
- 训练中断恢复。

验收：

- 20–100 step smoke GRPO 无 NaN、无死锁；
- reward 与离线重算完全一致；
- SFT checkpoint 与 GRPO checkpoint 可独立评估。

### Commit 9：统一评价与 HTML 报告

实现：

- Standard / High-Branching split；
- 3 个随机种子；
- bootstrap 95% CI；
- 核心效率指标；
- 程序错误类型；
- 训练曲线；
- 方法对比表；
- failure cases。

验收：

- 一个命令重建所有表格与图；
- Action-ID 与 GraphScript 成本口径一致；
- 报告明确区分 absolute counts 与 per-budget metrics。

## 停止条件

完成 Commit 9 后停止。除非核心 Go 条件成立且得到明确指令，不要继续实现：

- Explorer–Solver 交替训练；
- actual learning gain reward；
- Utility Critic；
- interactive GraphScript；
- branching query graph；
- visual nodes。

## 编码质量

- Python 3.11；完整类型标注；
- 公共接口有 docstring；
- 核心逻辑不依赖 notebook；
- 不吞异常；
- 日志禁止记录 access token；
- 数据与模型路径全部由配置指定；
- CPU 单元测试不加载 4B 模型；模型测试使用 mock client；
- 每次提交更新 `CHANGELOG.md` 和 `outputs/manifests/` 示例。

---

<!-- SOURCE: docs/00_research_brief.md -->

# 研究背景、定位与参考方法

## 1. 核心研究问题

GraphQuest 不以“回答一个已经给定的问题”为起点，而是把图本身视为任务空间：

> 在有限图访问预算下，模型应该到图的什么区域、以什么操作组合进行探索，才能持续发现有效且对当前 Solver 有挑战的问题？

当前版本进一步吸收 Graph-as-Code 的交互思想：模型不在 token 空间中逐条模拟图计算，而是生成受限、可执行的 GraphScript，由确定性环境完成遍历、集合运算、验证与 provenance 记录。

形式化地，Explorer 生成程序：

\[
c \sim \pi_\theta(c\mid G, e_0, b, \mathcal S)
\]

Executor 运行：

\[
(a, S^*, m)=\operatorname{Execute}(G,c)
\]

其中：

- \(e_0\)：种子实体；
- \(b\)：图访问预算；
- \(\mathcal S\)：图 schema/允许关系；
- \(a\)：答案；
- \(S^*\)：支持证据；
- \(m\)：edge visits、operators、latency 等成本。

程序随后被 verbalize 为问题 \(q\)，由固定 Solver 评估其经验难度。

## 2. 参考方法与借鉴边界

### 2.1 VisPlay

VisPlay 让同一个 VLM 交替承担 Questioner 与 Reasoner，并用 GRPO、难度和多样性奖励在无标注图像上形成自演化课程。

借鉴：

- Questioner–Solver 角色分离；
- 问题应处于当前 Solver 的能力边界；
- 后续可交替更新，而不是静态生成一次数据。

不直接照搬：

- 图环境能够执行查询并产生确定答案，不需要依赖银答案多数投票；
- 首轮不共同训练两个角色，避免环境非平稳。

### 2.2 Graph-R1 / Search-on-Graph-R1

这些方法把图检索建模为多轮 Agent–Environment 交互，并通过 RL 学习如何为给定问题寻找答案。

借鉴：

- 图作为可交互环境；
- 图调用成本应进入评价；
- SFT 冷启动后再进行 RL。

区别：

- 它们是 `question -> search -> answer`；
- GraphQuest 是 `graph -> exploration program -> new question + answer + support`。

### 2.3 SPARK

SPARK 使用科学文档构建多模态 KG，从 KG 路径生成问题，让 Proposer 与 Solver 在信息不对称下交替训练。

借鉴：

- KG 为问题生成和奖励验证提供结构基础；
- relation/hop difficulty 可用于启发式课程；
- SPARK-style weighted walk 是首轮重要 baseline。

局限与机会：

- 路径主要由加权随机游走获得；
- 核心不是学习一个预算受限的问题发现策略；
- Solver 失败不一定等同于问题有训练价值。

### 2.4 KGQAGen

KGQAGen 使用 LLM 与 Wikidata/SPARQL 交互，生成带答案、SPARQL 和 proof 的复杂问题，并发布 KGQAGen-10k。

借鉴：

- LLM-guided graph exploration；
- SPARQL/图执行验证；
- proof 子图作为快速沙盒来源。

区别：

- KGQAGen 是提示驱动的生成 pipeline；
- GraphQuest 要学习一个显式 RL policy，并在固定预算下优化 frontier/utility。

### 2.5 Actions Speak Louder than Prompts

该 ICLR 2026 Oral 系统比较 prompt linearization、固定图工具与 Graph-as-Code。其关键结论是：模型生成程序、由环境执行图计算，能够避免把大邻域直接塞进上下文，并比逐次固定工具调用具有更强组合表达能力。

对 GraphQuest 的启发：

- 让模型决定“执行什么图计算”，而不是在自然语言中手工模拟图算法；
- 程序可以作为 RL 宏动作；
- 执行结果天然可审计、可复现并能记录成本。

不能直接宣称的创新：

- 代码交互用于图推理已被该工作证明；
- Code-on-Graph 也已将 programmatic reasoning 用于给定 KGQA 问题。

GraphQuest 的差异应定义为：

> 现有程序化图方法解决给定任务；GraphQuest 学习编写图探索程序，主动发现能够训练另一个模型的新任务。

## 3. 论文潜在贡献

首轮只验证第一项，后续按证据扩展：

1. **Programmatic Question Discovery**：将 Graph-as-Code 从“解题”迁移为“发现题目”。
2. **Budget-Fair Interaction Study**：在完全相同 primitive graph budget 下比较 action、tool 与 program 三种探索方式。
3. **Self-Evolving Graph Curriculum**：后续通过 Explorer–Solver 交替更新形成动态课程。
4. **Learning-Utility Objective**：后续用真实 Solver 泛化增益替换难度代理。

## 4. 首轮可证伪假设

### H1

在相同 edge-visit budget 下，GraphScript GRPO 的 Frontier Discovery Efficiency 高于 SPARK-style weighted walk。

### H2

GraphScript 相对 Action-ID 的收益在 High-Branching split 上更明显。

### H3

GraphScript 的收益不能仅由 SFT 模仿解释；GRPO 应在 held-out seeds 上进一步提高效率。

任何假设不成立，都应如实保留结果，不应通过继续添加奖励模块掩盖。

---

<!-- SOURCE: docs/01_mvp_scope.md -->

# MVP 范围与明确非目标

## 1. MVP 输入和输出

输入：

- 一个小型有向、带标签知识图；
- 一个 seed entity；
- 允许使用的 relation whitelist；
- 最大 2-hop；
- primitive edge-visit budget。

Explorer 输出：

- Action-ID 轨迹，或 GraphScript v0.1 程序。

环境输出：

- 可执行查询；
- 唯一答案实体；
- support trace；
- canonical natural-language question；
- graph cost；
- Solver pass rate；
- 最终 reward。

## 2. Mini Graph 建议规模

### Smoke

- KGQAGen train proof 样本：200–500；
- 实体：约 500–2,000；
- 三元组：约 2,000–8,000；
- 关系：频率最高且可模板化的 8–12 种；
- seed：20–40；
- 目标：跑通数据、程序执行、Solver 与 20–100 step GRPO。

### MVP

- proof 样本：1,000–3,000；
- 实体：约 1,000–5,000；
- 三元组：约 10,000–30,000；
- 关系：10–20；
- seed：50–100；
- candidate tasks：500–2,000；
- probe tasks：100–300；
- Explorer episodes：5,000–20,000。

实际规模根据连通性调整，不以凑数字为目标。

## 3. 查询语义

首轮仅支持：

```text
seed --r1--> intermediate --r2--> answer
```

集合语义：

```text
H1 = follow({seed}, r1)
H2 = follow(H1, r2)
```

只有 `|H2| == 1` 才接受为唯一答案问题。

首轮不支持：

- intersection；
- comparison；
- count；
- argmax；
- filter；
- temporal qualifier；
- negation；
- 开放式长答案。

## 4. Solver 范围

首轮 Solver 不进行图工具搜索。它接收：

- canonical question；
- 固定大小的 local graph context；
- support triples 与同分布 distractors 的混合。

原因：

- 将首轮变量限制在 Explorer 交互方式；
- 避免同时实现第二个 RL Agent；
- 让 Solver 难度来自图关系理解与证据选择，而不是纯参数知识。

后续再替换为 Graph-R1-style tool Solver。

## 5. 模型配置

- Explorer base：`Qwen/Qwen3.5-4B`；
- Explorer adapter：LoRA；
- Solver：冻结的独立 Qwen3.5-4B 实例；
- Verbalizer：模板；
- thinking：关闭；
- vision encoder：首轮不使用，不解冻。

选择统一多模态基座是为了后续支持视觉节点，但首轮实验不允许视觉因素介入。

## 6. 明确非目标

MVP 不需要证明：

- 生成的问题已经达到最终人工数据质量；
- self-play 能无限迭代；
- GraphScript 在所有图规模都优于工具；
- 模型能够处理完整 Wikidata；
- 视觉扩展有效；
- learning utility 可以被预测。

MVP 只验证交互表示与 RL 探索是否值得继续。

---

<!-- SOURCE: docs/02_architecture.md -->

# 系统架构与接口契约

## 1. 总体数据流

```text
KGQAGen-10k train proof
        │
        ▼
MiniGraph Builder ──► GraphBackend ──► BudgetMeter
                                    │
Seed + schema ──► Explorer Policy ──┤
                                    ▼
                      Action-ID Env / GraphScript Executor
                                    │
                                    ▼
                        Query + Answer + Support Trace
                                    │
                                    ▼
                           Canonical Verbalizer
                                    │
                                    ▼
                     Local Context Builder + Frozen Solver
                                    │
                                    ▼
                          Validity / Frontier / Cost Reward
                                    │
                                    ▼
                               SFT / GRPO Update
```

## 2. 模块边界

### 2.1 DataAdapter

职责：

- 读取 KGQAGen；
- 解析 QID/PID；
- 输出标准 Triple；
- 管理 split 和来源。

禁止：

- 在模型训练过程中访问公网；
- 将 test question 文本用于训练。

### 2.2 GraphBackend

统一接口：

```python
class GraphBackend(Protocol):
    def neighbors(
        self,
        entities: EntitySet,
        relation_id: str,
        direction: Literal["out", "in"],
        limit: int,
        budget: BudgetMeter,
    ) -> EntitySet: ...

    def relations(
        self,
        entities: EntitySet,
        direction: Literal["out", "in"],
        budget: BudgetMeter,
    ) -> list[RelationStat]: ...

    def label(self, entity_id: str) -> str: ...
    def aliases(self, entity_id: str) -> list[str]: ...
```

首轮实现内存邻接表。后续 CSR、QLever 或 Virtuoso 必须保持接口兼容。

### 2.3 BudgetMeter

唯一允许的计费入口：

```python
@dataclass
class BudgetUsage:
    edge_visits: int
    operators: int
    returned_entities: int
    llm_input_tokens: int
    llm_output_tokens: int
    llm_calls: int
    execution_ms: float
```

硬限制：

- `max_edge_visits`；
- `max_operators`；
- `max_returned_entities`；
- `max_program_ops`；
- `max_llm_calls`。

任何 GraphBackend 操作都必须接收 BudgetMeter。

### 2.4 ExplorerPolicy

统一接口：

```python
class ExplorerPolicy(Protocol):
    def generate(self, observation: ExplorerObservation) -> ExplorerOutput: ...
```

实现：

- UniformExplorer；
- RelationBalancedExplorer；
- SparkWeightedExplorer；
- QwenActionIdExplorer；
- QwenGraphScriptExplorer。

### 2.5 TaskExecutor

输入 ExplorerOutput，返回：

```python
class GeneratedTask(BaseModel):
    task_id: str
    seed_entity: str
    interaction_mode: str
    program: dict | None
    action_trace: list[dict]
    relation_path: list[str]
    answer_ids: list[str]
    answer_labels: list[str]
    support: list[Triple]
    question: str | None
    valid: bool
    rejection_reason: str | None
    cost: BudgetUsage
```

### 2.6 Verbalizer

首轮确定性：

```python
question = template_bank.render(seed, r1, r2, answer_type)
```

relation template 只从 allowlist 中读取。没有模板的 relation 不进入图任务空间。

### 2.7 ContextBuilder

构建 Solver 输入：

- 必须包含 support；
- 固定数量 distractors；
- distractors 从相同 seed neighborhood 或同关系类型采样；
- 固定随机种子；
- 所有方法使用相同 context 构建策略。

### 2.8 SolverClient

接口：

```python
class SolverClient(Protocol):
    def solve_many(
        self,
        tasks: list[SolverTask],
        num_samples: int,
        temperature: float,
    ) -> list[list[SolverResponse]]: ...
```

实现：

- MockSolverClient：单元测试；
- TransformersSolverClient：小规模；
- VLLMSolverClient：训练与批量评价。

### 2.9 RewardService

同一个 RewardService 服务 Action-ID 与 GraphScript，防止评价漂移。

## 3. 角色隔离

```text
Explorer = Base Qwen3.5-4B + explorer_lora（可训练）
Solver   = Base Qwen3.5-4B 或 solver_lora（首轮冻结）
```

要求：

- 不共享 optimizer；
- 不共享 rollout history；
- reward 调用的 Solver checkpoint 固定；
- checkpoint id 写入每个任务记录。

## 4. 运行模式

### Offline deterministic mode

用于：数据、执行器、baseline、离线 reward 重算。

### Model mode

用于：SFT/GRPO。模型服务和图环境通过结构化 JSON 交互。

## 5. 输出目录

```text
outputs/
  manifests/
  datasets/
  graphs/
  candidates/
  sft/
  checkpoints/
  evaluations/
  reports/
  cache/solver/
```

每次 run 必须生成：

- resolved config；
- git commit；
- dependency lock hash；
- model revision；
- data hash；
- random seed；
- wall-clock start/end；
- metrics JSON。

---

<!-- SOURCE: docs/03_graphscript_spec.md -->

# GraphScript v0.1 规范

## 1. 设计原则

GraphScript 是“程序作为动作”，不是通用 Python。

目标：

- 可组合；
- 可验证；
- 可计费；
- 可审计；
- 适合 4B 模型生成；
- 绝不允许任意代码执行。

首轮采用 JSON AST，避免 Python 语法和 sandbox 安全问题。

## 2. 顶层结构

```json
{
  "version": "0.1",
  "ops": []
}
```

约束：

- `version` 必须为 `0.1`；
- `ops` 长度 1–5；
- 只允许白名单字段；
- 多余字段默认拒绝；
- JSON 外不能有自然语言、Markdown fence 或 CoT。

## 3. 操作

### 3.1 start

```json
{"op": "start", "entity": "$seed", "out": "h0"}
```

约束：

- MVP 只能从 `$seed` 开始；
- `out` 必须为未使用 handle；
- handle 格式 `h[0-7]`。

### 3.2 follow

```json
{
  "op": "follow",
  "in": "h0",
  "relation": "P57",
  "direction": "out",
  "limit": 16,
  "out": "h1"
}
```

约束：

- `in` 必须存在；
- relation 必须在 episode allowlist；
- direction 只能是 `out` 或 `in`；
- `1 <= limit <= config.max_follow_limit`；
- 执行采用集合语义并去重；
- 扫描到的每条邻接边计入 edge visits。

### 3.3 require_unique

```json
{"op": "require_unique", "in": "h2"}
```

要求 handle cardinality 为 1，否则程序无效。

### 3.4 emit

```json
{"op": "emit", "in": "h2"}
```

要求：

- 必须是最后一个 op；
- 前面必须出现 `require_unique`；
- 路径中必须恰好有两个 `follow`；
- 输出答案、relation path 和 provenance。

## 4. MVP 合法形状

唯一合法主形状：

```text
start -> follow -> follow -> require_unique -> emit
```

允许早停/无效输出仅用于记录错误，不接受为训练任务。

## 5. 明确禁止

- `import`、`open`、`exec`、`eval`；
- 循环、递归、函数定义；
- 文件系统、网络、shell；
- `all_nodes/all_edges`；
- 无限制邻居枚举；
- 动态 relation 字符串拼接；
- 模型指定未在 observation 中出现的 relation；
- 访问全图统计以绕过局部预算。

## 6. 解析错误分类

至少记录：

```text
NON_JSON
EXTRA_TEXT
UNSUPPORTED_VERSION
UNKNOWN_OP
EXTRA_FIELD
INVALID_HANDLE
DUPLICATE_HANDLE
RELATION_NOT_ALLOWED
INVALID_DIRECTION
LIMIT_EXCEEDED
INVALID_SHAPE
MISSING_UNIQUE
MISSING_EMIT
BUDGET_EXCEEDED
EMPTY_RESULT
NON_UNIQUE_RESULT
SHORTCUT_DETECTED
```

## 7. 执行语义

Executor 必须：

1. 先完整静态验证 AST；
2. 再逐 op 执行；
3. 每次操作通过 BudgetMeter；
4. 记录输入/输出 cardinality；
5. 保存实际访问的 support edges；
6. 任何错误立即终止；
7. 不允许部分成功被接受。

## 8. 等价性测试

对任意合法 GraphScript：

```text
GraphScript execution result
== equivalent Action-ID primitive sequence result
```

必须比较：

- answer set；
- support/provenance；
- edge visits；
- relation path；
- rejection reason。

## 9. 未来版本，但首轮不实现

### v0.2 interactive

新增：

- `inspect(handle, fields)`；
- 最多两轮程序生成；
- environment 返回压缩统计后再次编程。

### v0.3 query graph

新增：

- `filter_type`；
- `intersect`；
- `compare`；
- `count`。

### v0.4 multimodal

新增：

- `request_visual`；
- `inspect_table`；
- `crop_region`。

这些版本只有在 v0.1 的结果支持程序化交互后才开发。

---

<!-- SOURCE: docs/04_data_preparation.md -->

# 数据准备：从快速沙盒到论文规模

## 1. 首轮数据源

使用 Hugging Face 数据集：

```text
lianglz/KGQAGen-10k
```

其标准字段包括：

```text
id
seed
question
answer
sparql
proof
```

首轮只使用 train split 的：

- `seed`；
- `proof`；
- 实体/关系标签。

原始自然语言 `question` 不作为 Explorer 训练目标，也不用于构造模板，以减少复现原数据问题分布的风险。

## 2. 为什么适合快速验证

- 文件小；
- proof 直接提供三元组；
- 不需要下载完整 Wikidata；
- 可以快速构建包含真实 QID/PID 的图；
- dev/test 可保留为外部 probe。

限制：

- proof union 不是完整 Wikidata 邻域；
- 图的分支结构受已有问题生成分布影响；
- 只能用于机制验证，不可作为最终主实验的唯一图。

## 3. Proof 解析

兼容两种 proof 表示：

```json
["entity label (Q123)", "relation label (P45)", "entity label (Q678)"]
```

或：

```json
{"subject": "...", "predicate": "...", "object": "..."}
```

解析规则：

- subject 必须提取 QID；
- predicate 必须提取 PID；
- object 若无 QID，则视为 literal；
- MVP 默认只保留 entity-to-entity triples；
- 保存原始 label；
- 保存 source dataset id 和 split；
- 去重但不丢来源列表。

标准输出：

```text
triples.parquet
entities.parquet
relations.parquet
sources.parquet
manifest.json
```

## 4. 图构建

### 4.1 关系筛选

从 train proof 中：

1. 统计 relation 频率；
2. 排除 literal-heavy、难以自然模板化或高度技术性的 relation；
3. 选择 10–20 个有明确方向语义的 relation；
4. 为每个 relation 编写正向/反向模板；
5. 固化为 `relation_templates.yaml`。

### 4.2 图边

每条原始边保存：

```text
subject_id
predicate_id
object_id
subject_label
predicate_label
object_label
source_ids
```

GraphBackend 同时建立：

- outgoing adjacency；
- incoming adjacency；
- entity degree；
- relation frequency；
- relation-conditioned degree。

### 4.3 训练/验证/测试隔离

按 seed QID 隔离：

- train seeds：用于 SFT/GRPO；
- dev seeds：超参数与 early stopping；
- test seeds：最终结果。

要求：

- 同一个 seed 不跨 split；
- test question 文本不进入任何训练数据；
- 允许共享全局实体作为邻居，但必须报告 overlap；
- 额外提供 strict entity-disjoint split 作为稳健性测试（若规模允许）。

## 5. 任务候选生成

枚举或采样：

```text
seed -> r1 -> intermediate -> r2 -> answer
```

Gate：

- 两个 relation 都在 allowlist；
- answer 是实体且有英文 label；
- 最终 answer set 唯一；
- seed != answer；
- 路径无自环；
- seed 与 answer 之间不存在允许关系下的一跳 shortcut；
- 模板可生成问题；
- support 可完整重建。

输出：

```text
candidate_tasks_train.parquet
candidate_tasks_dev.parquet
candidate_tasks_test.parquet
```

这些 oracle candidates 仅用于：

- SFT trace 生成；
- baseline 上界；
- 单元测试；
- Solver 校准。

RL Explorer 不能直接看到 oracle answer/path。

## 6. Standard 与 High-Branching split

### Standard

随机选择满足 Gate 的 seeds/tasks。

### High-Branching

优先选择：

- seed 或 intermediate degree 位于候选的前 25%；
- 可用 relation 数较多；
- relation-conditioned neighbor 数较多；
- 存在同类型 distractors。

目的：检验程序化宏动作是否只在复杂图环境中有优势。

## 7. Local graph context

Solver context 固定结构：

```text
Question: ...
Facts:
1. [entity] --[relation]--> [entity]
...
Return JSON only: {"answer": "..."}
```

包含：

- 全部 support triples；
- 8–20 条 distractor triples；
- distractors 优先来自同 seed neighborhood、同 relation 类型或相同 answer type；
- 固定 context size；
- shuffle seed 固定。

## 8. 数据 manifest

必须记录：

```json
{
  "source_dataset": "lianglz/KGQAGen-10k",
  "dataset_revision": "...",
  "created_at": "...",
  "parser_version": "...",
  "train_sample_ids_hash": "...",
  "relation_allowlist": ["..."],
  "num_entities": 0,
  "num_relations": 0,
  "num_triples": 0,
  "split_seed_hashes": {},
  "file_sha256": {}
}
```

## 9. 论文规模扩展

核心机制成立后：

1. 从固定 Wikidata dump 构建真实局部邻域；
2. KGQAGen-10k 作为现代复杂 QA 基准；
3. QAWiki/WikiKGQA 2026 作为人工独立测试；
4. CWQ/GrailQA 作为经典桥接；
5. STaRK-MAG 或科学文档图作为跨图扩展。

不要在首轮实现这些数据源。

---

<!-- SOURCE: docs/05_training_and_rewards.md -->

# 模型、SFT、GRPO 与奖励

## 1. 模型角色

### Explorer

```text
Qwen/Qwen3.5-4B + explorer_lora
```

推荐初始 LoRA 配置：

```yaml
r: 16
alpha: 32
dropout: 0.05
bias: none
target_modules:
  - q_proj
  - k_proj
  - v_proj
  - o_proj
  - gate_proj
  - up_proj
  - down_proj
```

必须可由配置修改，不应写死。

### Solver

- Qwen3.5-4B；
- 冻结；
- 独立服务；
- 结构化 JSON answer；
- 不使用 Explorer adapter。

### Thinking

Explorer 动作/程序生成默认关闭 thinking。输出 token 上限：

- Action-ID：16–32；
- GraphScript：128–256。

## 2. SFT 冷启动

### 2.1 数据来源

从以下有效轨迹混合：

- Uniform；
- Relation-balanced；
- SPARK-style weighted；
- 少量 oracle candidates。

避免只模仿一个启发式策略。

### 2.2 输入

GraphScript prompt 至少包含：

```text
seed id / label
allowed relations and labels
max hops = 2
budget
GraphScript schema summary
```

不提供答案和 oracle path。

### 2.3 输出

只允许 JSON AST。

### 2.4 SFT 指标

- exact JSON parse rate；
- schema-valid rate；
- executable rate；
- valid-task rate；
- average edge visits；
- relation path entropy。

## 3. Solver 校准

RL 前先在 500–1,000 个合法候选上运行 Solver。

目标：

- 大多数任务不是全对或全错；
- pass rate 分布覆盖约 0.2–0.8；
- Standard 与 High-Branching 都有可区分梯度。

可调参数：

- distractor 数；
- Solver temperature；
- K samples；
- relation allowlist；
- question template。

不通过校准不得开始 RL。

## 4. Frontier reward

对任务 \(x\)，Solver 采样 K 次：

\[
\hat p(x)=\frac{1}{K}\sum_{k=1}^K\mathbb 1[\hat a_k=a^*]
\]

首轮建议 `K=4`。

边界奖励：

\[
R_f(x)=\exp\left(-\frac{(\hat p(x)-\mu)^2}{2\sigma^2}\right)
\]

初始：

```yaml
mu: 0.5
sigma: 0.2
```

这只是可学习性代理，不应在论文中宣称等同真实训练价值。

## 5. 硬有效性门控

```text
V(x) = 1 iff:
- output parses
- schema valid
- program executes
- budget respected
- exactly two follow ops
- unique answer
- no shortcut
- answer label exists
- question generated
- answer not leaked in question
```

无效任务：

```text
reward = invalid_reward  # 建议 -1.0
```

## 6. 成本

统一成本：

\[
C(x)=
\alpha E(x)+
\beta O(x)+
\gamma T_{out}(x)
\]

其中：

- \(E\)：primitive edge visits；
- \(O\)：operators；
- \(T_{out}\)：Explorer output tokens。

首轮 reward：

\[
R(x)=
\begin{cases}
-1,&V(x)=0\\
R_f(x)-\lambda C_{norm}(x),&V(x)=1
\end{cases}
\]

不要加入额外 reward term。

## 7. GRPO 训练

首轮建议：

- 每个 prompt group size：4–8；
- prompt 由 train seeds 采样；
- reward function 可异步批量调用 Solver；
- Solver response cache key 包含 task hash、solver revision、temperature、seed；
- checkpoint 可恢复；
- 定期在固定 dev seeds 上离线评价。

需要记录：

- reward mean/std；
- valid rate；
- parse rate；
- edge visits；
- p_solve 分布；
- KL；
- output length；
- relation path distribution。

## 8. Action-ID 与 GraphScript 的训练公平性

- 相同 base checkpoint；
- 相同 LoRA 配置；
- 相同 train seeds；
- 相同 SFT example 数；
- 相同 optimizer steps 或相同 token budget，两种口径都报告；
- 相同 reward service；
- 相同 graph edge budget；
- 相同 Solver 与 context。

## 9. 后续替换，而非叠加

### V2 自博弈

交替训练 Explorer LoRA 和 Solver LoRA。

### V3 Learning Utility

用实际短期训练增益替换 frontier reward：

\[
\Delta(B)=Perf(Update(S,B),D_{probe})-Perf(S,D_{probe})
\]

### Utility Critic

仅当真实 \(\Delta\) 有效但过贵时，才用预测器近似它。

不是在 frontier reward 上继续叠加新的 quality term。

---

<!-- SOURCE: docs/06_experiments.md -->

# 对比实验、指标与分析

## 1. 方法矩阵

| ID | 方法 | 模型训练 | 交互单位 | 目的 |
|---|---|---|---|---|
| B0 | Uniform Walk | 无 | 单边 | 随机下界 |
| B1 | Relation-Balanced | 无 | 单边 | 简单均衡启发式 |
| B2 | SPARK-style Weighted | 无 | 单边 | 最重要启发式 baseline |
| B3 | Frozen Qwen Action-ID | 无 | 单边 | prompt-guided 工具基线 |
| B4 | Action-ID SFT | LoRA SFT | 单边 | 模仿学习基线 |
| M0 | Action-ID GRPO | LoRA SFT+RL | 单边 | 原子动作 RL |
| B5 | GraphScript SFT | LoRA SFT | 程序 | 程序表示本身的收益 |
| M1 | GraphScript GRPO | LoRA SFT+RL | 程序 | 主方法候选 |

首轮不强制实现完整 KGQAGen pipeline 作为直接 baseline；后续论文规模再加入。

## 2. 公平比较

所有方法共用：

- Mini Graph；
- train/dev/test seeds；
- relation allowlist；
- max 2 hops；
- edge-visit budget；
- canonical templates；
- context builder；
- frozen Solver；
- validity gates；
- evaluation tasks；
- random seeds。

Program 一次 LLM call 可以包含多个 ops，但不能获得更多 edge visits。

## 3. 主要指标

### 3.1 Valid Task Rate

\[
\frac{\#valid\ tasks}{\#episodes}
\]

### 3.2 Frontier Task Rate

定义 frontier interval，例如：

```text
0.25 <= p_solve <= 0.75
```

\[
\frac{\#valid\ frontier\ tasks}{\#episodes}
\]

### 3.3 Frontier Discovery Efficiency

\[
\frac{\#valid\ frontier\ tasks}{edge\ visits/1000}
\]

这是首轮主指标。

### 3.4 Program/Action Efficiency

- valid tasks / 1K operators；
- valid tasks / 1K output tokens；
- frontier tasks / LLM call；
- wall-clock throughput。

### 3.5 Structural Diversity

只做分析，不进入 reward：

- unique relation paths；
- relation path entropy；
- seed coverage；
- intermediate entity coverage。

## 4. 质量指标

- answer uniqueness；
- no-shortcut rate；
- support faithfulness；
- answer leakage rate；
- question template validity；
- solver pass-rate histogram。

## 5. Standard / High-Branching

分别报告全部指标。

关键分析：

```text
interaction_mode × branching_split
```

如果 GraphScript 只在 High-Branching 有优势，应明确定位为条件性收益，而不是宣称普遍优越。

## 6. 统计

- 3 个训练 seed；
- test episode 固定；
- bootstrap 95% CI；
- 主比较报告绝对差与相对提升；
- 不仅报告最佳 run；
- 多次比较时控制解释，不追求堆显著性检验。

## 7. 消融

首轮只保留必要消融：

1. SFT vs SFT+GRPO；
2. Action-ID vs GraphScript；
3. 无 cost penalty vs 有 cost penalty；
4. Standard vs High-Branching；
5. template context distractor 数量。

不要一开始加入十几个模块消融。

## 8. Failure analysis

至少输出：

- top parser errors；
- empty result；
- non-unique result；
- shortcut；
- over-budget；
- repeated relation；
- Solver 全对/全错；
- GraphScript 比 Action-ID 更差的案例；
- 高度节点上的程序爆炸案例。

## 9. 结果解释边界

首轮结果只能支持：

- 交互方式在小型、固定图上的探索效率差异；
- 受限程序是否适合 4B 模型；
- RL 是否改善任务发现。

不能直接支持：

- 生成数据一定提升 fresh Solver；
- 方法可扩展到完整 Wikidata；
- 多模态扩展有效；
- 程序方式在所有图任务中优于工具方式。

---

<!-- SOURCE: docs/07_acceptance_criteria.md -->

# 阶段验收与 Go/No-Go

## P0 数据与图

必须满足：

- KGQAGen train proof 解析成功率 ≥ 99%；
- 输出文件有 SHA256；
- 无 test question 泄漏；
- 至少 500 个有效 2-hop unique-answer candidates；
- GraphBackend CPU 单测通过；
- 预算无法绕过。

否则：修复数据和图，不开始模型训练。

## P1 Solver 校准

必须满足：

- JSON answer parse rate ≥ 98%；
- 候选问题 pass rate 不是集中在 0 或 1；
- 至少 30% 候选处于预定 frontier interval；
- context distractor 生成稳定。

否则：调整 relation allowlist、模板、distractors 或 Solver sampling。

## P2 GraphScript SFT

目标：

- held-out seed JSON parse ≥ 95%；
- AST schema valid ≥ 93%；
- executable ≥ 90%；
- budget violation = 0；
- exact 2-hop shape ≥ 90%。

若 4B 模型无法达到：

1. 缩短 schema；
2. 使用 constrained decoding；
3. 增加 SFT；
4. 仍失败再回退到 Qwen3-VL-4B 或更强 Qwen checkpoint。

不要通过放开任意 Python 来解决。

## P3 GRPO smoke

必须满足：

- 训练无 NaN；
- reward 离线重算一致；
- 无环境死锁；
- checkpoint 恢复后指标连续；
- invalid rate 不持续恶化。

## P4 主要 Go 条件

建议至少满足其一：

### Go-A

GraphScript GRPO 相对 SPARK-style：

- Frontier Discovery Efficiency 相对提升 ≥ 15%；
- 3 个 seed 中至少 2 个方向一致；
- 95% CI 不显示明显负向；
- 成本没有数量级恶化。

### Go-B

GraphScript GRPO 相对 Action-ID GRPO：

- 在 High-Branching split 相对提升 ≥ 10%；
- Standard split 不显著变差；
- compile/execution rate 可接受。

### Go-C

GraphScript 并不提高任务数量，但显著降低 LLM calls/tokens，并保持相同 frontier efficiency。

## No-Go / 改向条件

- GraphScript 在等 edge budget 下持续低于 Action-ID；
- 优势完全来自更高 graph budget；
- 4B 模型长期无法生成稳定程序；
- Solver reward 不能区分任务；
- RL 仅学习输出固定 relation path；
- 提升只在 train seeds 出现。

No-Go 时的正确做法：

- 保留 Action-ID 作为主交互；
- 将 GraphScript 定位为高分支扩展或负结果分析；
- 不继续叠加模块制造提升。

## P5 进入自博弈前的额外条件

- 生成任务可用于训练 fresh Solver；
- fresh Solver 在 held-out probe 上出现正向增益；
- 问题不只是重复少数 relation path；
- 至少一个独立模型或 checkpoint 能复现收益。

---

<!-- SOURCE: docs/08_scale_up_roadmap.md -->

# 核心成立后的扩展路线

本文件不属于首轮实现范围。

## V2：Interactive GraphScript

目标：允许程序先检查局部图，再根据反馈继续编程。

流程：

```text
program 1: start/follow/inspect
environment: cardinality, relation stats, handle
program 2: follow/require_unique/emit
```

实现框架可迁移到 veRL AgentLoop。

## V3：交替 self-play

角色：

```text
Explorer = base + explorer_lora
Solver   = base + solver_lora
```

每轮：

1. 冻结 Solver，更新 Explorer；
2. 生成 frontier tasks；
3. 冻结 Explorer，更新 Solver；
4. 保存 Solver snapshot；
5. 重新校准 frontier。

两个 adapter 不同时更新。

## V4：Learning Utility

先测真实训练增益：

```text
copy current solver
train a few steps on candidate batch
measure held-out probe improvement
```

若 actual utility 明显优于 frontier proxy，则用它替换 reward。

Utility Critic 仅用于摊销计算成本。

## V5：复杂查询图

按失败需求扩展 GraphScript：

- filter_type；
- intersect；
- compare；
- argmax；
- temporal filter。

每次只增加一种 operator，并加入对应 deterministic tests。

## V6：论文规模数据

- 固定 Wikidata snapshot；
- KGQAGen-10k；
- QAWiki/WikiKGQA 2026；
- CWQ/GrailQA；
- STaRK-MAG 或科学文档图。

## V7：视觉节点

Qwen3.5-4B 基座保留视觉能力。未来 observation 可加入媒体句柄，但不直接把全部图像塞进上下文。

GraphScript 新操作示例：

```text
request_visual(handle, top_k)
inspect_table(handle)
crop_region(handle, box)
```

程序负责“看哪里”，VLM 负责“看到了什么”。

## 模块加入原则

| 实际观察到的问题 | 才加入的机制 |
|---|---|
| 长期生成重复 | persistent skill memory |
| 只攻击单一 Solver | solver pool |
| reward 过于昂贵 | utility critic |
| 长程序奖励稀疏 | step-level shaping |
| chain 表达不足 | query-graph operators |
| Solver 遗忘 | replay/history mixing |

没有对应失败证据时，不增加模块。

---

<!-- SOURCE: docs/10_task_checklist.md -->

# Codex 任务清单

## Foundation

- [ ] 建立 Python 3.11 项目与 lockfile
- [ ] 配置 Pydantic settings
- [ ] 建立 CLI
- [ ] 建立 JSONL logger
- [ ] 建立 manifest 与 hashing
- [ ] 建立 pytest

## Data

- [ ] KGQAGen adapter
- [ ] proof parser
- [ ] QID/PID parser
- [ ] train-only graph builder
- [ ] relation allowlist
- [ ] relation template file
- [ ] seed-disjoint split
- [ ] Standard / High-Branching split

## Graph

- [ ] adjacency backend
- [ ] reverse adjacency
- [ ] BudgetMeter
- [ ] 2-hop executor
- [ ] unique gate
- [ ] shortcut gate
- [ ] provenance

## Task and Solver

- [ ] canonical verbalizer
- [ ] distractor sampler
- [ ] solver prompt
- [ ] vLLM client
- [ ] answer normalization
- [ ] solver response cache
- [ ] calibration report

## Baselines

- [ ] uniform
- [ ] relation-balanced
- [ ] SPARK-style weighted
- [ ] frozen Qwen action-id

## Action-ID model

- [ ] observation renderer
- [ ] output parser
- [ ] SFT trace builder
- [ ] LoRA SFT
- [ ] GRPO

## GraphScript

- [ ] JSON AST models
- [ ] static validator
- [ ] executor
- [ ] error taxonomy
- [ ] pretty printer
- [ ] equivalence tests
- [ ] SFT trace builder
- [ ] LoRA SFT
- [ ] GRPO

## Evaluation

- [ ] per-episode metrics
- [ ] per-budget metrics
- [ ] 3 random seeds
- [ ] bootstrap CI
- [ ] Standard vs High-Branching
- [ ] failure cases
- [ ] HTML report

## Scope guard

- [ ] 未实现任意 Python exec
- [ ] 未训练 Solver
- [ ] 未下载完整 Wikidata
- [ ] 未加入复杂 reward
- [ ] 未加入视觉
- [ ] 未加入 self-play

---

<!-- SOURCE: docs/09_references.md -->

# 主要参考资料

1. **Actions Speak Louder than Prompts: A Large-Scale Study of LLMs for Graph Inference.** ICLR 2026 Oral. 研究 prompt、固定图工具和 Graph-as-Code 三种交互方式。
   https://arxiv.org/abs/2509.18487

2. **VisPlay: Self-Evolving Vision-Language Models from Images.** Questioner–Reasoner 自演化、GRPO、难度与多样性奖励。
   https://arxiv.org/abs/2511.15661

3. **SPARK: Self-Play with Asymmetric Reward from Knowledge Graphs.** KG 路径驱动的 Proposer–Solver 自博弈。
   https://arxiv.org/abs/2605.05546

4. **Graph-R1: Towards Agentic GraphRAG Framework via End-to-End Reinforcement Learning.** 多轮图检索 Agent。
   https://arxiv.org/abs/2507.21892

5. **Search-on-Graph-R1: Training Large Language Models to Search Knowledge Graphs with Reinforcement Learning.** Freebase、WebQSP/CWQ/GrailQA、SFT+RL。
   https://arxiv.org/abs/2607.18481

6. **Code-on-Graph: Iterative Programmatic Reasoning via Large Language Models on Knowledge Graphs.** 给定问题后的 Planning–Coding–Executing。
   https://arxiv.org/abs/2606.03705

7. **Diagnosing and Addressing Pitfalls in KG-RAG Datasets / KGQAGen.** Wikidata、SPARQL 验证、KGQAGen-10k。
   https://arxiv.org/abs/2505.23495
   https://github.com/liangliang6v6/KGQAGen
   https://huggingface.co/datasets/lianglz/KGQAGen-10k

8. **Qwen3.5-4B official model card.** 统一语言与视觉能力，可用于后续视觉扩展。
   https://huggingface.co/Qwen/Qwen3.5-4B

9. **TRL GRPOTrainer.** 自定义 reward、vLLM 集成与 VLM 支持。
   https://huggingface.co/docs/trl/grpo_trainer

10. **veRL Agent Loop and LoRA support.** 后续多轮 Agentic RL 与大规模训练。
    https://verl.readthedocs.io/en/latest/advance/agent_loop.html
    https://verl.readthedocs.io/en/latest/advance/ppo_lora.html

---
