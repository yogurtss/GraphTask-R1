# GRR mini128 失败分析（2026-08-31）

> 历史记录：本页只对应旧 GRR mini 与旧 split。后续 ECDB v1/v2 的最终失败判定、当前 novelty 边界
> 和证据路径见 [ECDB_FAILURE_ANALYSIS.md](ECDB_FAILURE_ANALYSIS.md)。

## 结论

Graph-Certified Residual Replay（GRR）没有提高新冻结 dev128 的精度，不晋级：

| 指标 | incumbent | GRR checkpoint-32 | 差值 |
|---|---:|---:|---:|
| correct / 128 | 5 | 5 | 0 |
| exact match | 3.90625% | 3.90625% | 0 |
| mean F1 | 4.427083% | 4.427083% | 0 |
| tool success / 128 | 9 | 10 | +1 |
| request failures | 0 | 0 | 0 |

paired comparison 是 candidate-only 1 题、baseline-only 1 题，exact two-sided McNemar `p=1.0`。
严格门槛要求 candidate correct count 增加，故 `dev-promotion.json` 选择 baseline 并返回失败状态。
blind128 没有推理、误差分析或 checkpoint 选择；现有 incumbent、旧实验结果和模型指针均未修改。

历史 `8/128` 来自另一冻结 task 集，只能作背景。这里唯一有效的结论是同一新 dev 上的 `5→5`。

## 受控变量与规模

底层数据与扩大 baseline 使用同一个 audited KQAPro pool：19,941 certified train、11,768 val。GRR
不是另一次 444-record 扩量训练，而是先用较小的等预算算法检验隔离 pair 来源：

- 固定 train probe：256 tasks，seed 4242；
- gold 在相同 graph snapshot/allowlist/budget 下重新认证：167；
- 因 certified gold 超过 200 edge budget 跳过：89，不自适应回填；
- fixed incumbent greedy rollout：167；其中错误 155、正确 12；
- 按冻结 task rank 取真实错误：128 pairs；
- DPO：128 records、1 epoch、effective batch 4、32 steps、LR `5e-7`、beta 0.1、LoRA rank 32；
- optimizer、prompt、解码和执行合同与此前 mini128 相同，唯一算法变量是 rejected 来源。

pair 的自然当前策略错误分布为：

| 阶段 | 数量 | 主要 reason |
|---|---:|---|
| predicted execution | 71 | ENTITY_NOT_FOUND 42、EMPTY_RESULT 18、RELATION_NOT_ALLOWED 8 |
| predicted parse | 50 | NON_JSON 24、INVALID_SCHEMA 16、TYPE_MISMATCH 7 |
| executable but wrong | 7 | TOOL_WRONG_ANSWER 7 |

这验证了代码审查判断：当前模型的主要 residual 是 parser/executor failure，旧 444-pair 合法合成
反事实没有直接覆盖它们。数据生成方向是有效诊断，但该训练形式没有产生净精度收益。

## 训练证据

final checkpoint 固定为 `dpo-mini/v0-20260831-103418/checkpoint-32`，没有用 dev 筛选 checkpoint：

- adapter SHA-256：`2c16bca35d6997750e781d1e7fbdd3d66f5b9534ca93f1e6ca40ae81e022ac54`；
- global step：32/32，epoch 1.0；
- train loss：0.69137526；
- 32 个 logging step 的 mean preference accuracy：0.5078125；
- mean reward margin：0.0041156，只有 17/32 steps 为正 margin；
- 无 NaN、OOM、请求失败或自动 best-checkpoint 选择。

训练健康但 preference separation 很弱；因此失败不能归因于运行中断，更多是完整响应级目标没有给
最早 parse/execution 错误足够精确的信用分配。

## 配对变化：修复与回退互相抵消

candidate-only：`gt_kqapro_val_588ffd711dfdbebc`

- 问题：`When was the association football club that is headquartered in Darlington established?`
- incumbent 多生成一个不存在的 type `Q1000000000`，执行得到 EMPTY_RESULT；
- GRR 删除该伪 type filter，正确执行为 `1883`。

baseline-only：`gt_kqapro_val_596e1495b3cfe5c7`

- 问题：`Which has fewer employees, Vivendi or Nintendo?`
- incumbent 正确使用 `select_between(..., attribute="employees", mode="min")`，得到 Nintendo；
- GRR 改成在 Vivendi entity handle 上执行 `verify`，触发 TYPE_MISMATCH，丢掉原正确题。

额外 tool success 但仍错误：`gt_kqapro_val_8db553966e5424b2`

- 问题：`Which has fewer season, The Golden Girls or The Monkees with Peter Tork in the cast?`
- incumbent 因把 answer handle 与 entity handle 做 intersect 而 TYPE_MISMATCH；
- GRR 变为可执行程序，但错误地 count cast intersection，输出 `0`，gold 是 `The Monkees`。

所以 `tool +1` 是真实的执行可用性改善，但没有转化成 answer correctness。

## 错误分布不是净减少，而是重分配

| reason | incumbent | GRR | 变化 |
|---|---:|---:|---:|
| NON_JSON | 32 | 35 | +3 |
| ENTITY_NOT_FOUND | 34 | 32 | -2 |
| EMPTY_RESULT | 19 | 14 | -5 |
| INVALID_HANDLE | 7 | 3 | -4 |
| TYPE_MISMATCH | 9 | 13 | +4 |
| INVALID_SCHEMA | 11 | 12 | +1 |
| RELATION_NOT_ALLOWED | 1 | 3 | +2 |
| BUDGET_EXCEEDED | 5 | 5 | 0 |
| executable wrong | 4 | 5 | +1 |
| correct | 5 | 5 | 0 |

GRR 减少了 EMPTY_RESULT、INVALID_HANDLE 和 ENTITY_NOT_FOUND，却增加 NON_JSON、TYPE_MISMATCH、
INVALID_SCHEMA 和 RELATION_NOT_ALLOWED。完整序列 DPO 学到了全局输出偏移，但没有稳定保持已正确的
SSA/operation 决策。

## 计算与延迟说明

evaluator 报告的 `mean_completion_tokens` 会把 fallback 最终失败记录记为零，不能代表实际 primary
generation 长度。按 `primary_completion_tokens` 重新审计：incumbent 为 233.64 tokens/task，GRR 为
238.24（+1.97%）。服务端 mean latency 为 12.33s→18.44s（+49.5%），但 candidate 是 base+LoRA、
baseline 是 merged model，因此该差值包含 adapter serving 开销，不能直接当作合并后算法延迟结论。

## 根因判断

1. **task-local on-policy 数据合同是必要但不充分。** 它成功命中真实错误分布，也修复了一题和一个
   tool failure；因此不是“负例完全无关”。
2. **完整响应级 DPO 的 credit 太粗。** parse、SSA type、entity resolution 与 terminal semantic
   错误都只表现为一个序列偏好，模型不知道第一个破坏 graph execution 的 operation 在哪里。
3. **更新发生错误迁移。** 低 mean margin 与一赢一负说明模型跨过少数 greedy 边界，但没有稳定
   约束已正确前缀；简单增加相同 pair 数量可能同时放大收益和回退，缺少机制依据。
4. **可执行不等于正确。** tool success +1 的题把 TYPE_MISMATCH 改成了可执行但语义错误的 count，
   证明下一步应优化 graph execution boundary 的局部决策，而非只追求合法 JSON/可执行率。

## 停止与后续边界

本轮按预注册规则停止：不打开 blind，不训练 444-pair 版本，不改 seed/LR/epoch，不把 tool +1 登记为
精度提升，也不更新 incumbent。

若未来获准开启一个新受控实验，优先的单一算法改动是 **graph execution-boundary credit
assignment**：在当前策略输出与 certified gold 的最早 parser/SSA/executor 失败 operation 处对齐
前缀，只对失败 operation 到下一个可执行边界施加偏好，保持已正确执行前缀不动。它直接针对本轮
“修一题、坏一题”的信用分配问题；在有新 dev 证据前只能作为假设，不能宣称会提高精度。

## 证据文件

- rollout：`outputs/experiments/kqapro-graph-residual-replay/mini/rollout_manifest.json`
- pairs：`outputs/experiments/kqapro-graph-residual-replay/mini/pairs/manifest.json`
- baseline：`outputs/experiments/kqapro-graph-residual-replay/eval-dev-incumbent/`
- candidate：`outputs/experiments/kqapro-graph-residual-replay/eval-dev-grr/`
- paired comparison：`outputs/experiments/kqapro-graph-residual-replay/dev-paired-comparison.json`
- promotion gate：`outputs/experiments/kqapro-graph-residual-replay/dev-promotion.json`
