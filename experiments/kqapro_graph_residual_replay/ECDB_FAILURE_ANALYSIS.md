# ECDB 方法审查与失败记录（2026-09-04）

## 最终判定

`ECDB v2 candidate-set` 的执行状态分三层：训练与 dev 评测已完成；狭义数值筛选以 `4→5` 返回
`passed=true`；最终科学晋级判定为 **failed_semantic_attribution**，不晋级：

- 冻结 dev128 的官方 Exact/F1 从 incumbent 的 `4/128` 变为 `5/128`，但唯一净增题来自错误程序碰巧输出正确的 `no`；相对 incumbent 是 `0` 个 genuine gain。
- 相对 ECDB v1，v2 有两个 accidental gain，同时丢失 v1 唯一的 genuine semantic repair；按程序语义审计是 `0` 个 genuine gain、`1` 个 genuine loss。
- v2 在 42 个需要完整同名实体候选集的 dev 题上，一次也没有生成 `exact limit>1`。因此不能把名义 `+1/128` 归因于 candidate-set 表示或 ECDB 机制。
- 双侧 exact McNemar 检验无论对 incumbent 还是 v1 都是 `p=1.0`；没有统计提升证据。
- 数值 gate 没有识别 accidental exact，所以其 `passed=true` 不用于模型晋级。control 未训练，blind 未推理，incumbent 未替换。

这是在 v1 无提升后进行的一次有机制依据的算法修正，不是超参数搜索。v2 仍未得到真实精度提升，故按预定规则到此停止，只保留负结果。

## 被审查的方法

ECDB 的当前计算链为：

```text
固定当前策略失败
  -> schema-valid GraphScript 的逐前缀执行状态
  -> operation + denotation LCS 对齐
  -> 把当前策略的一个 operation 移植到 certified-gold scaffold
  -> 重绑定 SSA handle 并执行完整 grafted program
  -> 只保留可重放且终值错误的反事实
  -> full-chosen / local-rejected 非对称偏好训练
  -> 固定 dev 配对评测
```

这里可以安全保留的一条 novelty 主线只有：

> **executed-state boundary → current-policy conditional operation → whole-program counterfactual graft certificate → boundary-scoped preference**

更具体地说，ECDB 的窄差异点是“负方向”的反事实构造：将当前策略实际产生的局部错误 operation 注入可执行的 gold scaffold，再用完整程序重放证明该局部干预确实导致终值失败。不能把 first-error、DPO、局部 token 加权、teacher forcing、SSA typing、on-policy refresh 或 candidate-set 分别写成创新。

当前实现也不能笼统称为“parser/SSA/executor boundary”：457 个错误 rollout 中有 217 个（47.5%）是 `NON_JSON / INVALID_SCHEMA / INVALID_HANDLE / TYPE_MISMATCH`，在进入执行边界定位前就被排除。准确表述应是 **schema-valid GraphScript 的 execution/denotation boundary**。

## 与相邻工作的边界

step/token 级偏好与“定位关键错误”本身已有充分先例：[Step-DPO](https://arxiv.org/abs/2406.18629) 在推理步骤上构造偏好，[Target-DPO](https://aclanthology.org/2025.emnlp-main.707/) 与 [Focused-DPO](https://aclanthology.org/2025.findings-acl.498/) 将更新集中到关键 token/片段，[Critical Tokens](https://proceedings.mlr.press/v267/lin25j.html) 研究少量关键 token 对推理结果的影响。因此，ECDB 不能以“局部 DPO”或“first error”为 novelty。

[Verified Critical Step](https://aclanthology.org/2026.findings-acl.1974/) 和 [CausalFlow](https://arxiv.org/abs/2605.25338) 已沿着对失败轨迹做因果/验证干预的方向推进；它们是当前最接近的 novelty 威胁。ECDB 可检验的窄区分不是向失败轨迹注入正确修复，而是把 **当前策略错误 operation 反向注入 certified gold scaffold**，并要求 whole-program execution certificate。

知识图谱问答中的代码式语义解析也不是空白：[CodeAlignKGQA](https://aclanthology.org/2025.coling-main.267/) 已研究代码形式与 KGQA 对齐；[Candidate Expressions](https://arxiv.org/abs/2410.00414) 也使用候选表达式改进语义解析。candidate-set 只能作为 ECDB 的观测覆盖组件，不能独立声明 novelty。

## v1：边界局部化没有提高 Exact

| 项目 | incumbent | ECDB v1 |
|---|---:|---:|
| Exact/F1 | 4/128 | 4/128 |
| tool success | 7/128 | 8/128 |
| genuine gain | — | 1 |
| answer-exact loss | — | 1（incumbent 本身是 accidental exact） |

v1 使用 64 个正 margin hard pairs，边界 family 为 root/transform/projection `27/21/16`。训练 32 steps、2 epochs；mean reward margin `0.006737`、mean preference accuracy `0.5625`。它真实修复了 Makerere University 题的 `select_between(attribute="students count", mode="max")`，但总 Exact 没有提升，严格门禁失败，未运行 control/blind。

失败后没有重复调 LR、LoRA rank 或 epoch，而是检查漏掉的 gold 表示。原转换器把同名实体 union 展开为多个不可观测 ID，导致大量 gold 无法认证，这形成 v2 的单一算法修改。

## v2：candidate-set 覆盖修正及其结果

v2 增加了“完整同名候选集”序列化：只有当同一 label 的 exact resolver 返回集合与 gold entity set 完全相等，才把直接 entity union 折叠成一个 bounded `resolve_entity`。转换后仍要求：

1. gold GraphScript 在固定图、allowlist 和预算下执行；
2. 执行答案等于 task gold；
3. 执行还原程序的 canonical signature 等于源 program。

这个修改把 rollout gold certification 从 `407` 提高到 `493`，unobservable rejection 从 `236` 降到 `2`，证明它解决了 **训练数据可观测性**。但它没有解决模型决策：

- 新增 86 个可认证任务中只有 24 个能进入 eligible boundary；
- v2 hard64 中只有 9 条含 multi-candidate chosen，6 条把该 resolve 作为 marked boundary；
- 只有 2/64 是固定 query/match/out、纯粹隔离 `limit=1` 与 `limit=N` 的 cardinality 对照；
- 42 个 candidate-set dev 任务中，incumbent、v1、v2 生成 `exact limit>1` 的次数均为 `0`。

### 训练信号

| 训练量 | ECDB v1 | ECDB v2 |
|---|---:|---:|
| hard pairs | 64 | 64 |
| steps / epochs | 32 / 2 | 32 / 2 |
| loss | 0.689896 | 0.691212 |
| mean reward margin | 0.006737 | 0.004245 |
| median reward margin | 0.006375 | 0.001247 |
| mean preference accuracy | 0.5625 | 0.5078 |
| positive-margin logging steps | 22/32 | 17/32 |

v2 第二个 epoch 没有形成 margin 上升趋势；信号接近随机分离。当前模板的实际 active-token 统计为 chosen 平均 `121.16`、rejected 平均 `27.11`，aggregate ratio `4.47×`。因此该 objective 实际包含很强的隐式 chosen SFT anchor，不能被描述成纯 local DPO，也难以把变化归因到 boundary token。

### 冻结 dev128

| 指标 | incumbent | ECDB v1 | ECDB v2 |
|---|---:|---:|---:|
| Exact/F1 | 4/128 | 4/128 | 5/128 |
| tool success | 7 | 8 | 8 |
| successful-but-wrong | 3 | 4 | 3 |
| parse failure | 51 | 55 | 51 |
| execution failure | 70 | 65 | 69 |

v2 相对 incumbent 的表面 paired flip 是 `1 gain / 0 loss`；相对 v1 是 `2 gains / 1 loss`。样本太少，且逐程序审计后全部净增都不成立。

## 所有 Exact flip 的程序语义审计

### Ben Foster：v2 相对 incumbent 的唯一 gain 是 accidental exact

问题：`Was Ben Foster of Manchester United F.C. born in 1981?`

gold 语义是：Manchester United F.C. 经 `member of sports team` 反向 follow，和完整的同名 Ben Foster 候选集相交，查询 `date of birth`，再验证年份 1981。

v2 却使用错误关系 `member of`，仍以 `limit=1` resolve Ben Foster，然后执行 `query_relation`，把关系答案拿去和年份 1981 比较。它没有查询出生日期，也没有使用候选集机制；错误 denotation 恰好得到 `no`。因此这是 answer-level accidental exact，不是正确程序。

### Jazz/Yugoslavia：相对 v1 的第二个 gain 也是 accidental exact

gold 应从 Yugoslavia 经 `country of origin` 反向 follow、按 Jazz 类型过滤后计数。v2 复现 incumbent 的错误程序：union Jazz/Yugoslavia，再用 synthetic `Q1000000000` type 过滤得到空集，计数碰巧为 `0`。

### Amsterdam/Makerere：v2 丢失 v1 的 genuine repair

v1 正确使用 `select_between(..., attribute="students count", mode="max")` 得到 Makerere University。v2 回退到 incumbent 的错误 `query_relation` 并触发 `EMPTY_RESULT`。

严格结论：

- v2 vs incumbent：`0 genuine gain`；
- v2 vs v1：`0 genuine gain + 1 genuine loss + 2 accidental gains`。

## 根因

1. **目标表示要求模型预测不可见 cardinality。** v2 将完整候选集写成 `limit=N`，但问题文本不提供 `N`。模型继续输出训练先验更强的 `limit=1`；提升 gold coverage 不等于可学习的推理目标。
2. **边界覆盖不足。** 47.5% 的错误在 schema-valid execution boundary 之前被排除，最终 hard64 只占 457 个错误的 14.0%。这不是加更多 defensive parser fallback 能解决的问题。
3. **LCS 边界不等于因果边界。** 当前对齐同时要求 operation 名称和 denotation 相同；语义等价但异构的 operation 会被判为差异。若首个 graft terminal-equivalent，当前流程直接丢题，不继续寻找后续真正改变终值的干预；v2 有 12 个此类 terminal-semantic rejection。
4. **类型兼容只检查输入。** hard64 中 14 条（21.9%）graft 的输出 kind 与 gold operation 不一致，包括 `query_attribute → follow` 等。它们只能叫 input-state-compatible，不能叫完整 typed-compatible；只有 30/64 是同 operator 的参数级反事实。
5. **训练集中于浅层。** boundary gold step `0/1/2 = 26/24/14`，没有一条 `>=3`。family quota 没有 depth quota，学到的是 root 决策，不足以支撑 long-horizon credit assignment 主张。
6. **loss 的局部性不对称且混杂。** full chosen 与 local rejected 的 active-token 暴露相差约 4.47 倍；v2 margin/accuracy 接近随机，dev 上 19 个任务只是在失败类型之间迁移。

## 下一轮唯一建议：gold-prefix boundary-conditioned ECDB

不要扩展为多条 novelty。下一轮仍沿用同一 ECDB 主线，但把“从完整失败轨迹做 LCS 后跨前缀移植”改为“在已执行 gold state 上直接探测当前策略的下一步 operation”：

```text
certified gold prefix/state at depth d
  -> current policy generates one conditional next operation
  -> exact SSA + input/output contract check
  -> graft this operation before the untouched gold suffix
  -> whole-program replay tests whether terminal answer changes
  -> retain the earliest causal wrong operation
  -> local preference + explicit small gold anchor
  -> refresh policy and repeat at registered depth buckets
```

算法上的优先修改如下，它们都服务于这条主线：

1. **把 cardinality 从预测目标中移除。** DSL 改成 `resolve_entity(query, match="exact", scope="all_exact", max_results=20, out=...)`，或统一使用固定上限并由 backend 认证 exact set completeness；不要让模型猜隐藏的 `N`。
2. **gold-prefix conditioned probing。** 对每个已执行 gold prefix/denotation teacher-force 上下文，让 incumbent 生成一个 next op。这样 current-policy operation 与实际 SSA state 同源，不再依赖 operation+denotation LCS，也能在 ambiguous root 上直接采到 `one` 对 `all_exact` 的真实错误。
3. **完整 I/O contract 与因果筛选。** 要求 input kind、output kind、SSA handle 全部兼容；枚举可行位置并重放到终值。首个 graft 若 terminal-equivalent，继续搜索，而不是丢弃任务。优先保留可执行但答案错误的 `TOOL_WRONG_ANSWER`，而不是靠 execution failure 提供信号。
4. **显式拆分 loss。** 使用 `L_local-DPO + λ L_gold-CE`，chosen/rejected 只在匹配的 causal dependency slice 上做对称比较，gold anchor 单独计量。预注册 λ，报告两项 loss 与 active-token/梯度占比；不要把 SFT anchor 隐藏在非对称 mask 里。
5. **按深度刷新。** 设 root/transform/projection 之外的 depth quota，至少覆盖 `>=3` 的边界；每一阶段更新后重新 rollout，避免浅层错误永久垄断 pair。

不建议先做更多 epoch、更大 LoRA 或宽泛超参 sweep：当前瓶颈是目标不可观测、因果定位和 loss 混杂，扩大相同信号更可能过拟合。constrained decoding 可以作为独立的合法率消融，但 v1/v2 已表明“更可执行”不等于“答案正确”，不应成为 novelty 主线。

## 下一轮预注册的可证伪门槛

- 与 v2 相同的 64 pairs、optimizer 和评测 dev；只改变上面的 boundary-conditioned 构造与显式 loss。
- graft 的 input/output kind 兼容率必须为 100%，并单独报告同 operator 参数反事实占比。
- schema-valid failure 的 localizable rate 必须显著高于当前 52.5%；hard set 必须有预注册的 `depth>=3` 配额。
- 在冻结的 42 个 candidate-set dev 子集上，报告 `all_exact` emission rate、root-denotation recall、canonical-program correctness 和 answer Exact；不能只看最后字符串。
- 主晋级条件同时要求：总 dev Exact 严格提高、canonical-program-correct 不下降、所有净增题通过逐程序语义审计、tool success 不下降。
- 同 pair 比较 local+anchor、full/full 与 chosen-only anchor，区分 boundary preference 和普通 SFT 的贡献。
- 只有 genuine dev gain 满足上述门槛后才允许一次 blind；否则继续只记录失败。

## 复现与证据

- frozen dev SHA-256：`cbb11ddfc5a53153b5f602907fc4c3b97011daf05e8fe547beb53fd221ad04b0`
- frozen blind SHA-256：`85f95dc6b977e82bcb761f418895de36d544e907125a9d146032cb25d948dcdf`（未推理）
- incumbent predictions SHA-256：`e13efecda6fb6ae24c7b5720c5868569f7dada8f637575cbb0cd5f20ea05c372`
- v1 hard train SHA-256：`68c3621826ed514ef0d775e29a7d8bbb2acfded3b1d2d98a9af8d5f7ea3eafca`
- v1 adapter SHA-256：`8c254267aaa58e8f488d5b1b4efca51f8dfdc3bfbb8225f756894c77c65c8567`
- v2 rollout SHA-256：`b824a9a2fcf19260de166427cf5689a75562ab8007a062f8b29ed5be989ec7c0`
- v2 hard train SHA-256：`007ad6a3447a9d01884d32375cb517f2e59f50d91d2055f958ed648389af188a`
- v2 adapter SHA-256：`6baaa8b6943c51b66699f862231d152331b921350366acc2fc3d6480c4a4a993`
- v2 predictions SHA-256：`17de9a36c65fdc78403476c707799ef66532e982ca563974de1156def28a462d`

主要本地产物：

- v1：`outputs/experiments/kqapro-graph-residual-replay/boundary-v1/`
- v2：`outputs/experiments/kqapro-graph-residual-replay/boundary-v2-candidate-set/`
- v2 training checkpoint：`boundary-v2-candidate-set/dpo-boundary/v0-20260904-140755/checkpoint-32`
- v2 eval：`boundary-v2-candidate-set/eval-dev-ecdb-v2-candidate-set/`
- paired comparisons：`dev-v2-vs-incumbent.json`、`dev-v2-vs-v1.json`

仓库中原有的 `README.md`、`INNOVATION.md` 和 `FAILURE_ANALYSIS.md` 记录的是更早的 GRR mini 实验；
其中旧 dev/hash、full-response objective 与“数值 dev 胜出后打开 blind”的规则不适用于本次 ECDB
最终语义门禁。配置文件存在也不代表 control 或 blind 已执行。

## 验证状态

- 相关单元/实验测试：56 passed；
- `make lint`：passed；
- `make typecheck`：passed，101 source files；
- `make test`：436 passed、1 skipped；
- 本次相关 tracked diff 的 whitespace check：passed。

实验测试必须使用登记的 `.conda/ms-swift-debug` 环境；系统 Python 未安装 `ms-swift`，不作为实验运行
环境。任何与本实验无关的既有 worktree 修改均未纳入本次结论，也未被改写。
