# ERE scale444：扩量诊断与停止原因

## 结论

本轮把 mini 的 128 个 DPO pair 扩到 444 个，与 contract-large SFT 的训练记录数对齐；其余
训练超参数保持不变。冻结 dev128 上的正确任务仍为 **8/128**，与 incumbent 和 mini 完全相同，
所以本轮 **不晋级、不运行 blind、不合并模型，也不更新正式精度或创新性效果主张**。

扩量使模型更明确地区分训练 pair：平均 DPO loss 从 mini 的 `0.69288497` 降到
`0.68301476`，平均 reward margin 从 `0.00083079` 升到 `0.02086299`，平均 reward accuracy
从 `0.5000` 升到 `0.6869`。但这个训练内收益只把 tool success 从 incumbent 的 9/128 提到
11/128；新增的两次可执行程序答案仍然错误，没有转化为 EM/F1。

## 受控变量

[scale444.yaml](configs/scale444.yaml) 与 [mini.yaml](configs/mini.yaml) 使用相同的 incumbent、seed、
模板、最大长度、1 epoch、micro batch 1、gradient accumulation 4、学习率 `5e-7`、DPO
`beta=0.1` 和 LoRA 参数。只改变：

| 项目 | mini | scale444 |
|---|---:|---:|
| pair 记录数 | 128 | 444 |
| optimizer steps | 32 | 111 |
| save cadence | 32 | 111 |
| 输出目录 | `dpo-mini` | `dpo-scale444` |

`scale444` 的 444 条训练记录与 contract-large SFT 的 444 条记录相同，但 DPO 每条同时包含
chosen/rejected。按 tokenizer 后的平均 combined pair length `21463.97` 估算，本轮约处理
`9,530,004` sequence tokens；contract-large SFT 约为 `4,857,802`，前者约为后者的 `1.96x`。
因此，本轮失败不能再合理地归因于“训练记录或 token 暴露明显更少”。这只是近似的 sequence-token
比较，不等价于严格 FLOPs 对齐。

## 数据构造证据

重新挖掘的画像仍只有 30 个去重 question，且 `effective_signal_count=0`。因此 444 对全部标记为
`dual_executed_default_single_slot_ablation`，`residual_guided=false`；它不是完整的
execution-residual self-evolution。

pair builder 得到 444 对时的槽分布为：

| mutation slot | 数量 | 占比 |
|---|---:|---:|
| relation | 175 | 39.41% |
| comparator | 109 | 24.55% |
| mode | 96 | 21.62% |
| attribute | 46 | 10.36% |
| set_operator | 14 | 3.15% |
| direction | 3 | 0.68% |
| qualifier | 1 | 0.23% |

relation、comparator 和 mode 合计 380/444（85.59%）；literal、type、query 均为 0。构造过程中
另有 32,078 次候选拒绝，其中 `EMPTY_RESULT=23464`、`UNCHANGED_DENOTATION=5141`、
`EXECUTION_ERROR=3297`、`BUDGET_EXCEEDED=176`。这说明双执行认证有效过滤了无效反事实，
但并未保证留下的是 incumbent 概率空间中最易混淆、最能修复端到端错误的 negative。

## 训练与冻结 dev 结果

训练输出为
`outputs/experiments/kqapro-execution-residual-evolution/dpo-scale444/v0-20260831-075403/checkpoint-111`。
1 epoch 共 111 step，耗时 4,013.48 秒，峰值显存 11.775 GiB。

| 指标 | incumbent | mini-128 | scale444 | scale444 − incumbent |
|---|---:|---:|---:|---:|
| Correct / 128 | 8 | 8 | 8 | 0 |
| EM | 6.25% | 6.25% | 6.25% | 0 pp |
| F1 | 6.25% | 6.25% | 6.25% | 0 pp |
| Tool success | 9 | 10 | 11 | +2 |
| Tool success rate | 7.03125% | 7.8125% | 8.59375% | +1.5625 pp |
| Model request failed | 0 | 0 | 0 | 0 |

scale444 与 incumbent 的正确 task 集合完全相同：candidate-only `0`，baseline-only `0`，exact
McNemar 双侧 `p=1.0`。固定 seed 4242、10,000 次 paired bootstrap 的 95% CI 为：EM/F1
`[0, 0]`，tool success `[0, 3.90625]` pp。严格 gate 只因
`candidate_correct_count_strictly_higher=false` 失败。

与 mini 直接配对时也没有正确 task 变化；tool success 只多 1/128，bootstrap 95% CI 为
`[0, 2.34375]` pp。相对 incumbent 新增的两个 tool success 分别是：

- `EMPTY_RESULT -> TOOL_WRONG_ANSWER`：1 个；
- `INVALID_HANDLE -> TOOL_WRONG_ANSWER`：1 个。

也就是说，扩大数据确实让两个输出跨过了解析/执行门槛，但没有修复语义答案。

## 为什么扩量没有提高精度

1. **没有 self-evolution 因果信号。** `effective_signal_count=0` 时，构造器只能按固定默认槽顺序
   采样。扩大的是静态单槽消融，不是由当前 Solver 错误驱动的课程。
2. **训练目标与主要 dev 错误错位。** scale444 的 128 个终态中，`NON_JSON=34`、
   `ENTITY_NOT_FOUND=30`、`INVALID_SCHEMA=14`，三类合计 78/128。当前 pair 只在两个合法、可执行
   GraphScript 间做语义槽判别，不能直接监督这些发生在语义执行之前的错误。
3. **槽覆盖严重偏斜。** 85.59% 的 pair 集中在 relation/comparator/mode，而 direction、qualifier
   几乎没有，literal/type/query 完全缺失。规模增加主要重复强化一条窄分布。
4. **可执行分歧不等于 hard negative。** 当前 mutator 接受首个满足单槽、非空和答案分歧的候选，
   未利用 incumbent 条件 log-prob 选择它真正容易混淆的反事实。训练统计改善只能说明 DPO 学会了
   这批 pair，不能说明这些 pair 对真实生成错误有因果针对性。
5. **端到端证据直接否定了规模假设。** 本轮记录数已与扩大 baseline 对齐，估算 token 暴露还约为
   baseline 的 1.96x，但正确集合逐题不变。当前瓶颈是反馈质量和任务对齐，而不是原始 pair 数量。

## 决策

依据“未高于之前结果则不更新需要做的实验，只整理原因”的约束，本轮到此停止：

- 不运行 blind；
- 不合并或替换 incumbent；
- 不把 scale444 写入正式 `RESULTS`、论文性能表或专利效果实施例；
- 保留独立 config、checkpoint 和机器可读比较结果，供后续因果设计复核。

下一次若继续，不应再单独扩大同类 pair。仍沿 graph + self-evolution 主线，只值得验证一个算法变化：
用固定 incumbent 在 train 上重新采样真实错误残差，仅保留可定位的同-operator residual，然后在所有
双执行合法的单槽候选中，按 incumbent 条件 log-prob 选择最混淆 negative。它把“执行残差 ->
反事实 -> 更新”闭环补齐，而不增加额外 critic、retriever、refiner 或 ensemble。

## 可重放标识

- Git 起点：`d93cb3d3354312ee4dbfbe81b7afd4ed59e2fd08`
- canonical dev SHA-256：`967b3d6494d12c531208a6a46588300a686fe6198bd1fe0bc83000fdb450b514`
- residual profile SHA-256：`192cc7a75d8c5615f08960e0f93f7942227a81b3ef59be9453d7429d3464d24e`
- pair train SHA-256：`18b1c1e468bf899c5e2fe3d574bd6b588c4d6337e56915b994421ee12f659103`
- pair certificate SHA-256：`b02348441f62ceb37d476de317112a1e007271e92784e898749155acd8154d96`
- candidate adapter SHA-256：`065616da0f023793460862c88c62b704f285d243f1d26579b4704151c2394118`
- candidate predictions SHA-256：`c47336b10831d98aaf0b5111f0c72464032ac7cd3720618842f4234db50ecb43`
- candidate metrics SHA-256：`3bfe1b36103b5be1c3e50c39ed1e8c1710f45ce58951f5b2d3552950cbd5076e`
- incumbent paired comparison：
  `outputs/experiments/kqapro-execution-residual-evolution/dev-paired-comparison-scale444.json`
  (`6d5877a9ec9e9debb200acad38f20ab8df5163862f034fad9455d85fd35751ca`)
- mini paired comparison：
  `outputs/experiments/kqapro-execution-residual-evolution/dev-paired-comparison-mini-vs-scale444.json`
  (`fd2830f2b5d2fce24952e277f7f930f860bed72ee4ed140c152e6d7097453def`)
- promotion decision：
  `outputs/experiments/kqapro-execution-residual-evolution/dev-promotion-scale444.json`
  (`9a0f7b6e8db8b6086151bdece170aab6c86de968a65f68fbf518e89056dab36e`)
