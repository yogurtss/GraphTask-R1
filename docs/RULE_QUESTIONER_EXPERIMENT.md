# 独立的 Program-first Questioner 实验

这个实验不替换现有 Questioner/self-play 流程。它把“采样可执行程序”和“生成自然语言问题”拆开，
便于与当前端到端 Questioner 做严格 A/B：

```text
规则采样 Program -> 转为 GraphScript -> 有界/无界执行一致性检查 -> 严格认证
                    |                                      |
                    +-> SFT: Program -> 真实 KQAPro question
                    +-> RL:  固定 Program -> Questioner 只生成 {"question":"..."}
```

gold answer 只来自认证程序的执行结果。SFT 和 RL 都不会把 answer 放进 Questioner prompt；RL
completion 也不再生成程序。独立变体名为 `rule_program_question_v1`，只有数据行显式携带这个变体时，
`ms_swift_reward.py` 才分派到新 reward；现有数据和默认路径不变。

## 1. 采样实验

```bash
PYTHONPATH=. python scripts/experiment_path_sampler.py \
  --graph-db data/processed/kqapro/kqapro-v03-full-audit/graph.sqlite \
  --reference-tasks data/processed/kqapro/kqapro-v03-full-audit/train/tasks.parquet \
  --relation-catalog data/processed/kqapro/kqapro-v03-full-audit/relation_catalog.json \
  --attempts 1000 --seed 42 \
  --output-dir outputs/experiments/rule-path-sampler-n1000
```

本机 N=1000 的结果：

| strategy | 构造成功率 | 严格认证率 | 算子覆盖 | 真实长度区间命中率 |
|---|---:|---:|---:|---:|
| naive_path | 86.2% | 58.0% | 3/18 | 63.6% |
| bounded_path | 100.0% | 68.6% | 3/18 | 65.2% |
| family_balanced | 98.8% | 62.2% | 13/18 | 97.4% |

推荐 `family_balanced`：它略牺牲严格认证率，但明显改善算子覆盖，同时长度最贴近真实 KQAPro。
下游只导出 `strict_certified=true` 的候选，因此构造成功和严格认证是两个独立指标。

## 2. 同时生成独立 SFT 和 self-play 数据

```bash
PYTHONPATH=. python scripts/prepare_rule_questioner_data.py \
  --graph-db data/processed/kqapro/kqapro-v03-full-audit/graph.sqlite \
  --reference-tasks data/processed/kqapro/kqapro-v03-full-audit/train/tasks.parquet \
  --candidates outputs/experiments/rule-path-sampler-n1000/family_balanced/candidates.jsonl \
  --baseline-mixed-sft outputs/qwen3-0.6b-contract-large/data/mixed-sft.parquet \
  --sft-count 188 --selfplay-count 64 --opponent-samples 1 --seed 42 \
  --output-dir outputs/experiments/rule-questioner-contract-large/data
```

输出包括：

- `questioner-sft.parquet`：188 条固定 Program 到真实问题的 Questioner SFT；
- `mixed-sft.parquet`：保留基线全部 256 条 Solver，只替换 188 条 Questioner；
- `questioner-selfplay.parquet`：64 条固定且严格认证的 Program，Questioner 只生成问题。

Questioner reward 会从任意服务端前缀中提取首个完整的 `{"question":"..."}`，因此
`</tool_call>{"question":...}` 不会再被记为 `NON_JSON`；对象 schema、问题/程序对齐、答案泄漏和
认证状态仍分别记录。

## 3. 单样本 opponent 的 archive 门槛

确定性 Transformers opponent 且 `opponent_samples=1` 时，旧 `pass_rate` 只能取 0 或 1。
因此 frontier 区间 `[0.25, 0.75]` 数学上不可能接收任何候选。独立流程使用：

```text
difficulty_signal = (program_parse_rate + program_execution_rate + mean_f1) / 3
```

单样本仍能得到 0、1/3、2/3、1 四档难度。promotion 命令为：

```bash
PYTHONPATH=. python scripts/promote_rule_questioner_archive.py \
  --candidates path/to/candidate_archive.sqlite \
  --archive path/to/archive.sqlite \
  --report path/to/archive_admission.json \
  --min-difficulty 0.25 --max-difficulty 0.75
```

对本次 64 条真实 opponent 结果，旧 pass-rate frontier 接收 0 条（59 `TOO_HARD`、5
`TOO_EASY`），新分阶段信号接收 8 条（7 条解析成功但执行失败、1 条执行成功但语义未完全正确）。
round 1 可将门槛设为 `[0, 1]` 接收全部认证题，后续再切回 `[0.25, 0.75]`。

## 4. contract-large A/B

两侧都使用 Qwen3-0.6B、444 条训练样本、1 epoch、222 steps、同一 256 条 Solver 数据和同一
64 条 KQAPro val：

| 数据 | exact/F1 | tool success | fallback |
|---|---:|---:|---:|
| 原 mixed SFT | 2/64 = 3.125% | 4/64 = 6.25% | 93.75% |
| Program-first Questioner SFT | 3/64 = 4.6875% | 5/64 = 7.8125% | 92.1875% |

这个 64 题结果显示没有精度退化且略有提升，但差异只有一个样本，不能视为统计显著。新模型的
主要剩余问题是 GraphScript schema/执行质量，而不是开头的 `</tool_call>`：本次 PT 部署中该前缀
出现 0 次，失败主要为 `NON_JSON`、`INVALID_SCHEMA` 和实体解析错误。
