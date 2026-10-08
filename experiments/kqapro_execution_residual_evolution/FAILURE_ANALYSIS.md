# ERE mini：未晋级原因分析

## 结论

本轮 **不晋级**。冻结 dev128 上，candidate 与 incumbent 的 EM/F1 完全相同，只有一次额外的
“程序可执行但答案错误”。严格门槛要求 candidate correct count 增加，因此没有运行 blind、full
训练或模型合并，也没有改写现有 incumbent、主实验配置或“最佳精度”记录。

| 指标 | incumbent | candidate | 差值 |
|---|---:|---:|---:|
| Exact count | 8/128 | 8/128 | 0 |
| EM | 6.25% | 6.25% | 0 pp |
| F1 | 6.25% | 6.25% | 0 pp |
| Tool success | 9/128 (7.03125%) | 10/128 (7.8125%) | +0.78125 pp |
| Model request failed | 0 | 0 | 0 |

八个正确 task 在两侧完全相同：candidate-only `0`，baseline-only `0`，exact McNemar 双侧
`p=1.0`。固定 seed 4242、10,000 次 paired bootstrap 的 95% CI 为：EM `[0, 0]`、F1
`[0, 0]`、tool success `[0, 2.34375]` pp。paired gate 与仓库 promotion gate 均失败，后者明确
选择 `merged-step56`。

机器可读结果：

- `outputs/experiments/kqapro-execution-residual-evolution/dev-paired-comparison.json`
- `outputs/experiments/kqapro-execution-residual-evolution/dev-promotion.json`
- 有效预测目录为 `eval-dev-incumbent-localnet/` 与 `eval-dev-ere-localnet/`。
- 第一次 sandbox-denied 的全请求失败结果已重命名为
  `eval-dev-incumbent-INVALID-sandbox-transport/`，不参与任何统计。

## 这次实际训练的是什么

代码复核后，当前训练不能称为完整 ERE，只能称为
`dual_executed_default_single_slot_ablation`：

1. 历史 completion artifact 共 67 个观测，去重后只有 30 个 question，而且来自多个 optimizer
   step，并非固定 `merged-step56` snapshot。
2. 修正跨算子误定位后，30 个观测的 `effective_signal_count=0`。原先唯一的 `query` residual
   实际是 `resolve_entity` 对 `all_entities` 的 `OPERATOR_MISMATCH`。
3. 因此 128 个 preference pair 没有被 execution residual 驱动，而是按默认槽顺序生成。

修正后的 30 条画像为：

| 原因 | 数量 |
|---|---:|
| INVALID_SCHEMA | 10 |
| ENTITY_NOT_FOUND | 5 |
| INVALID_HANDLE | 4 |
| RELATION_NOT_ALLOWED | 3 |
| NON_JSON | 2 |
| RESIDUAL_UNRESOLVED | 2 |
| BUDGET_EXCEEDED | 1 |
| EXECUTION_ERROR | 1 |
| OPERATOR_MISMATCH | 1 |
| TERMINAL_CORRECT | 1 |

pair builder 在访问 428 个 train task 后得到 128 对，task-level yield 为 `29.91%`。accepted slot
分布为 relation 55、comparator 32、mode 25、attribute 13、set-operator 3；direction、qualifier、
literal、type 和 query 均为 0。训练数据 SHA-256 为
`1f5b20e56c1188534d754677e486124dc2407181b44073eb4d310ca38a510f61`，certificate SHA-256 为
`a308ac587a9218c87f279f6aa7c0c0fb2d246284fffa7ca3087100f49e18ec14`。

DPO 从合并后的 incumbent 启动，所以 adapter-off reference 确实是进入本轮的 `pi_0`，不是原始
base。128 对、1 epoch、effective batch 4 共 32 step；最终 checkpoint 在训练前已经登记为
`checkpoint-32`。训练统计：

- mean/train loss `0.69288497`，与 `log(2)=0.69314718` 几乎相同；
- mean reward margin `0.00083079`；
- mean reward accuracy `0.5`；
- 训练耗时 1,096.87 秒，峰值显存 11.27 GiB。

这些数值表明 pair 虽然可执行且答案分歧，但没有证据说明它们在 incumbent 概率空间中是 hard
negative。

## 为什么没有提高 EM

1. **因果信号没有进入训练。** 设计中的“当前 Solver 最早 denotation residual”在本轮覆盖率为
   0；实际数据只是单槽双执行消融，不能把结果归因于完整 ERE。
2. **执行分歧不等于概率困难。** relation 类型相容与答案不同只能证明 negative 合法，不能证明
   `pi_0` 会混淆 chosen/rejected。接近 `log(2)` 的 loss 和近零 margin 与弱更新一致。
3. **主要误差发生在语义槽之前。** incumbent 的 dev terminal failure 是 119/128，candidate 是
   118/128；大量错误是 NON_JSON、INVALID_SCHEMA、实体解析或 relation allowlist 失败。默认
   relation/comparator mutation 没有针对这些失败。
4. **槽覆盖偏斜。** 128 对中 55 对已经在 relation，其他关键槽缺席；这更像一种固定数据增强，
   不是按能力残差形成的 self-evolving curriculum。
5. **额外执行没有转化为正确答案。** candidate 只把一个 failure 变成 TOOL_WRONG_ANSWER，正确
   task 集合未改变。candidate 平均延迟还从 12.55 秒增至 19.42 秒；延迟不是本轮硬门槛，但也
   没有补偿零精度收益。

## 代码审查后的保留修正

这些修正都只在本独立目录，未接入现有训练线：

- 跨 operator 的 residual 记为 `OPERATOR_MISMATCH`；只有相同 operator 才比较语义字段，唯一闭集
  例外是 `intersect`/`union -> set_operator`。
- profile 明确输出 localized/effective signal；零信号时 pair manifest 强制标为 ablation，不能
  冒充 ERE。
- mutation 逐候选保存 slot、operation index、候选值 hash、stage 与 reason code，不保存原始值。
- 冻结 split 禁止静默覆盖；A/B gate 拒绝重复 task ID 和任何 `MODEL_REQUEST_FAILED`。

## 下一次只值得验证的一条改动

不建议在本轮上继续加 epoch、扩大 LoRA 或叠加模块。若继续 ERE，应只做一个因果闭环：

1. 用固定 hash 的当前 incumbent 在一个冻结 train 子集重新采样，禁止混用训练过程中的多个 step；
2. 只有出现可消费的同-operator execution residual 才构造 pair；
3. 在所有单槽、双执行通过的候选中，用 incumbent 的条件 log-prob margin 选择它最容易混淆的
   negative，再做同预算 DPO；
4. 仍只注册一个 final checkpoint，并沿用本次未见 dev/blind 门槛。

这仍是一条“graph execution residual -> single-slot counterfactual -> self-evolving update”的算法链，
没有增加 retriever、critic、refiner 或 test-time ensemble。只有它在新 dev/blind 上提高 EM 后，才有
资格把精度收益写入论文或专利实施例；当前结果只支持方法设计与失败诊断，不支持效果主张。

## 可重放标识

- Git 起点：`d93cb3d3354312ee4dbfbe81b7afd4ed59e2fd08`
- dev SHA-256：`967b3d6494d12c531208a6a46588300a686fe6198bd1fe0bc83000fdb450b514`
- blind SHA-256：`7c3c9b1a2bfc589517a1861b5710001af1579d8cad251230b105ffd83067c55e`
  （仅完整性校验，未进行 blind 推理或误差分析）
- graph SQLite SHA-256：`be5918376b3d292fe009cd35b928993c4e0b378bbfaeaec12d9ce6bdbe352453`
- relation catalog SHA-256：`5fad40470a05a20a4b90fc3af73f135eebff4d53fa0cb2c20f95e3038ec6bccf`
- incumbent weight SHA-256：`acf263621172a3088f1ebf77a9befc31538a08a0944a89f1e591efe018bf7106`
- candidate adapter SHA-256：`e7ac99939bb428191780330db10a2f77447f89a60293e7f371b84417397243a8`
- evaluator config SHA-256：`3b4def87d1b46b513efa608ea0ddffdb64fbb823fbce895ceec057d62349e0d4`
