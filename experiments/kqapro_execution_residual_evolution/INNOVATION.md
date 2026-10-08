# ERE：执行分歧残差驱动的反事实图程序自进化

## 一句话创新

ERE 不再让 Questioner 随机增加任务，也不叠加 retriever、refiner 或另一个 reward network；它把
Solver 已经产生的错误 GraphScript 与 certified program 的**中间执行 denotation**对齐，找到最早
语义分歧槽，再由同一知识图谱双执行认证一个只差该槽的 executable counterfactual candidate
（可执行反事实候选），用成对偏好更新 Solver。该证书只证明候选可执行且答案分歧；候选对当前
incumbent 的 hardness 需要另测，不能从单槽构造直接推出。

核心闭环只有一个：

```text
Solver 执行错误 -> 最早 denotation 残差 -> 单槽反事实 -> 双执行证书 -> Solver 偏好更新
```

## 当前实现真正缺的信号

这条路线来自代码与历史结果的交叉审查，不是从候选模块列表中拼出来的：

- 当前 `program_structure_f1` 把程序压成 operator **集合**后做 F1；relation、direction、literal、
  qualifier、同一 operator 的次数和 SSA 拓扑都不进入这个分数。因而两个使用相同 operator、但在
  决定答案的 relation 上相反的程序仍可得到结构分 1。见
  [`ms_swift_reward.py`](../../graphtask_r1/training/ms_swift_reward.py)。
- 现有 Program-first Questioner 先抽一个固定 certified program，后续主要变化是问题表述；Solver
  的执行错误不会反向改变下轮程序的具体语义槽，因此 program 本身没有沿能力残差演化。见
  [`selfplay.py`](../../graphtask_r1/training/selfplay.py) 与
  [`questioner_sampling.py`](../../graphtask_r1/training/questioner_sampling.py)。
- 已记录的最高 first-64 只是 5/64，对同批 SFT 的 3/64 仅多两题，exact McNemar `p=0.625`；另一个
  未参与 checkpoint screening 的 64 题是 4/64 对 2/64，`p=0.5`。两者都是方向信号，不足以支持
  已提高精度的结论。
- 当前完整 relation catalog 令单条 prompt 约 44k 字符、10.6k tokens。ERE 保持该 prompt 和推理
  contract 不变，让 A/B 唯一变量仍是反事实自进化算法；动态检索或 prompt 缩短不被混入本实验。

因此优先修复的是“合法但语义只错一处”的梯度，而不是再增加一个生成器、检索器或推理时修正器。

## 机制定义

给定固定知识图谱快照 `G`、问题 `x`、certified program `P+` 和 Solver 输出 `P_hat`。对每个产生
handle 的程序前缀执行一次，得到：

\[
Z(P,G)=\left(z_1,\ldots,z_T\right),\qquad
z_t=\operatorname{Exec}(P_{\le t},G).
\]

materialized denotation 的相似度是 answer F1；`all_entities` 等不能安全物化的前缀使用显式 deferred
状态，仅在状态和预算参数完全相同时视为相同。ERE 用动态规划求一对一、保持顺序的最大相似度
对齐：

\[
A^*=\arg\max_A\sum_{(i,j)\in A}\operatorname{F1}(z_i(\hat P),z_j(P^+)),
\qquad
s_A={\sum_{(i,j)\in A}\operatorname{F1}(z_i,z_j)\over
\max(|Z(\hat P)|,|Z(P^+)|)}.
\]

最早没有 exact-denotation 对齐的 gold prefix 定义为 residual `r`。只有 predicted/gold operation
算子相同时才从字段差异中抽取槽 `s`；`intersect`/`union` 是唯一映射为 `set_operator` 的跨算子
闭集，其他跨算子差异记为 `OPERATOR_MISMATCH`，不伪装成缺失字段信号。随后仅对具有可消费候选值
的信号统计 `c(s)`；下一轮 mutation 以 `c(s)` 降序优先，而非均匀随机改程序。

对任务 `i`，mutation operator `M_s` 只改变一个顶层语义字段：

\[
P_i^- = M_s(P_i^+),\qquad d_{\text{slot}}(P_i^+,P_i^-)=1.
\]

relation/attribute/qualifier 候选优先满足 catalog 的 domain/range 类型相容；direction、comparator、
min/max、intersect/union 使用封闭的语义对偶。pair 只有满足以下证书才进入训练：

\[
\begin{aligned}
&\operatorname{Exec}(P_i^+,G;B)=y_i,\\
&\operatorname{Exec}(P_i^-,G;B)=y_i^- ,\\
&y_i\ne\varnothing,\quad y_i^-\ne\varnothing,\quad y_i^-\ne y_i,
\end{aligned}
\]

其中两次执行共享 snapshot、relation allowlist 和预算 `B`。因此 negative 不是语法坏例或空结果
捷径，而是与正确程序仅差一个语义决策的可执行反事实。

Solver 更新采用标准 DPO，不再混入额外手工 reward：

\[
\mathcal L_{\mathrm{ERE}}=-\mathbb E_i\log\sigma\left(\beta\left[
\log{\pi_\theta(P_i^+\mid x_i)\over\pi_0(P_i^+\mid x_i)}-
\log{\pi_\theta(P_i^-\mid x_i)\over\pi_0(P_i^-\mid x_i)}
\right]\right).
\]

`pi_0` 是进入本轮的 incumbent，`pi_theta` 是 ERE 更新后的 Solver。下一轮重新采集
`pi_theta` 的 train 输出即可形成自进化，而 graph、证书和随机 seed 均保持 instance-scoped、可重放。
只有 residual profile 产生可定位且受 mutation 支持的有效语义槽时，pair 构造才是完整 ERE。若
effective signal 为零、槽不受支持，或只能按默认槽顺序/随机选择 candidate，构造仍可作为双执行
**random/default single-slot ablation** 运行，但 manifest 必须显式标记该模式，结果不能归因于
residual-guided ERE，也不能据此宣称完整创新链产生了技术效果。

## 为什么这不是模块堆砌

ERE 只替换当前流程中信息量不足的一环：从“terminal success / 粗 operator-set 匹配”改为“执行
轨迹定位后的单槽对比学习”。前缀执行负责定位，单槽 mutation 负责产生训练信号，双执行负责证明
pair 有效，DPO 消费同一 pair；四步是一个不可拆的因果链，不是并列的四个模型。

明确不加入：

- 新的实体链接器、检索器或答案 refiner；
- 独立 critic/reward model；
- 多路 ensemble、self-consistency 或 test-time reranking；
- 用 val/blind 反推 residual profile 或 mutation 频率。

## 与直接相关工作的差异

| 工作 | 其核心机制 | ERE 的实质差异 |
|---|---|---|
| [Absolute Zero / AZR](https://arxiv.org/abs/2505.03335) | 单模型提出并求解开放式代码/数学任务，用代码执行器验证任务与答案 | 不声称 zero-data，也不开放式生成题目；在固定 KG 和真实 question 上定位最早 denotation 分歧，构造只差一个槽的成对程序证书 |
| [R-Zero](https://arxiv.org/abs/2508.05004) | Challenger 与 Solver 两个独立模型从零生成任务并在能力边界共同演化 | 不训练 Challenger；curriculum 的坐标不是整题难度，而是当前 Solver 的 relation/direction/qualifier 等可执行语义残差 |
| [Self-Questioning Language Models](https://arxiv.org/abs/2508.03682) | proposer 生成问题，依据“不太易/不太难”奖励，Solver 无 gold 时用多数投票近似正确性 | 不使用多数投票或题级 solve-rate；正确与错误程序都由同一 KG 严格执行，训练信号是同问题下的最小语义反事实 |
| [Knowledge-Graph Paths as Intermediate Supervision](https://arxiv.org/abs/2605.05702) | 用构题路径给 proposer 关系上下文，并用 waypoint coverage 给错误搜索轨迹部分奖励 | 它复用 positive construction path；ERE 对 predicted/gold output-prefix denotation 做 DP 对齐，并生成 answer-divergent negative program 供 pairwise update |
| [Learning from Mistakes](https://aclanthology.org/2021.findings-emnlp.222/) | 在弱监督语义解析中利用搜索到的错误 logical forms 改善 consistent-form 搜索与伪程序问题 | ERE 已有 certified program，不做大空间 weak-supervision search；错误先被定位到最早执行槽，再约束成 edit-distance-one、双执行认证的 executable counterfactual candidate；hardness 待测 |
| [Execution-based Spurious Program Filtering](https://aclanthology.org/2023.emnlp-main.425/) | 在多个 world 上执行候选程序，以 execution representation 多数投票过滤 spurious consistent programs | ERE 不做 pool majority filtering；同一 KQAPro graph 上比较每个 output prefix，保留故意 answer-divergent 的反事实作为负偏好 |

最接近的共有元素分别是 self-play、KG intermediate signal、execution filtering 和 mistake learning；
本实验需要验证的组合新意是“**最早前缀 denotation 残差定位 → 残差频率驱动的单槽 mutation → 同图
双执行 divergence certificate → 成对 Solver 更新**”。不能把任何一个共有元素单独包装成创新。

## 论文价值与可证伪假设

论文主张应限制为三个可测假设：

1. 前缀 denotation residual 比 terminal 0/1 提供更高覆盖率的可定位训练信号；
2. residual-prioritized 单槽 candidate 比随机单槽、随机程序 candidate 有更高的可执行、非空且
   answer-divergent 反事实接受率；候选 hardness 另以 incumbent chosen/rejected log-prob margin 检验；
3. 在相同 pair 数、token 数和训练步下，ERE 相对 incumbent 提高未见 KQAPro EM，且 F1/tool
   success 不回退。

最小消融足够，不添加新模块：

- incumbent，不再训练；
- dual-executed random single-slot（移除 residual priority）；
- 完整 ERE。

报告 pair proposal/acceptance、residual coverage、slot 分布、EM/F1/tool success、按 gold operator
分桶、paired task wins/losses、exact McNemar 和固定 seed 的 paired bootstrap CI。dev 用于一次
门槛判断；blind 只在 dev 胜出后运行一次。此前
`5/64` 的 first-64 结果被多次用于筛选，不能当作新 split 的统计基线。

## 专利角度的可保护链条

可讨论的发明链不是“知识图谱 + DPO”，而是一个具体的数据生成与模型更新方法：

1. 在相同图快照和预算下，分别执行预测程序和认证程序的 output-producing prefixes；
2. 按 denotation 相似度做保持因果顺序的对齐并确定最早执行分歧；
3. 将分歧映射到有限语义槽，并依据历史 Solver 槽残差选择一个 type-compatible edit；
4. 对原程序和单槽反事实进行双执行，仅签发非空、答案分歧且 positive-gold 一致的证书；
5. 根据证书形成 chosen/rejected 条件程序对，更新 Solver，并用新 Solver 的残差驱动下一轮。

可记录的技术效果包括：减少无效/空 negative 的训练计算、把错误定位从 terminal 提前到具体 graph
operation、提升 pair 的语义邻近性与可审计性、使训练数据和更新轮次可由 trace/hash 重放。申请时
应把 snapshot、预算、slot-distance-one 证明、两侧 denotation 和更新轮次写入实施例，而非只给
抽象算法名称。

这只是工程上的可专利点梳理，**不是新颖性检索结论、专利性意见或 freedom-to-operate（FTO）
分析**。执行式语义解析、反事实数据增强和 preference optimization 已有大量先例；提交前必须由
专利代理人做权利要求检索、司法辖区分析，并以实验结果证明技术效果。

## 主要风险与停止条件

- parse 失败的 Solver 输出不能形成细粒度 prefix residual，画像可能偏向“已经会输出合法 JSON”
  的样本；应如实报告 unresolved 比例。
- type-compatible relation 仍不等于自然语言语义邻近；单槽约束只能降低、不能消除这个风险。
- answer-divergent 只证明 candidate 对当前图是错的，不保证它在模型概率空间中足够困难；hardness
  待测，需用 incumbent log-prob margin 或训练 loss 验证。
- 0.6B mini 的统计功效有限。dev 不严格提高 EM，或 F1/tool success 回退时，停止并写失败分析；
  不打开 blind，不更新 incumbent。
- dev 胜而 blind 败说明选择偏差或泛化不足；保留独立代码和负结果，不注册“精度提升”。
