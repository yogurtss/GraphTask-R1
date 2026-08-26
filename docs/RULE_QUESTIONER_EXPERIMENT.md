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

上面的 N=1,000 用于比较采样策略。4B 正式流程先把 `family_balanced` 候选池扩大到 10,000 次
尝试，使严格认证后的唯一样本足以支持约 5,000 条 Questioner SFT 和 4,096 条 self-play seeds：

```bash
PYTHONPATH=. python scripts/experiment_path_sampler.py \
  --graph-db data/processed/kqapro/kqapro-v03-full-audit/graph.sqlite \
  --reference-tasks data/processed/kqapro/kqapro-v03-full-audit/train/tasks.parquet \
  --relation-catalog data/processed/kqapro/kqapro-v03-full-audit/relation_catalog.json \
  --attempts 10000 --seed 42 \
  --output-dir outputs/experiments/rule-path-sampler-n10000
```

4B 默认数据规模使用主流程约 10k 行的 1:1 mixed SFT，并生成 4,096 条 Program-first seeds：

```bash
PYTHONPATH=. python scripts/prepare_rule_questioner_data.py \
  --graph-db data/processed/kqapro/kqapro-v03-full-audit/graph.sqlite \
  --reference-tasks data/processed/kqapro/kqapro-v03-full-audit/train/tasks.parquet \
  --candidates outputs/experiments/rule-path-sampler-n10000/family_balanced/candidates.jsonl \
  --baseline-mixed-sft outputs/sft-data/preflight/mixed-train-accepted.parquet \
  --selfplay-count 4096 --opponent-samples 4 --seed 42 \
  --output-dir outputs/experiments/rule-questioner-4b-large/data
```

省略 `--sft-count` 时，脚本从 baseline mixed SFT 自动读取实际 Questioner 行数并等量替换，因此
即使 5,000 条源任务认证后有少量 shortfall，仍能保持 Solver 行完全不变且维持严格 1:1 A/B。

下面的 188/64 命令保留为 0.6B bounded A/B，不作为新的默认规模：

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

## 5. 六阶段 self-play 脚本

Program-first 4B 数据准备和约 10k mixed SFT 完成后，先设置六阶段配置引用的路径：

```bash
export INITIAL_ADAPTER=$PWD/outputs/sft/qwen3-4b-kqapro-v03/v0/checkpoint-625
export BASE_TASKS=$PWD/outputs/sft-data/tasks/train.parquet
export QUESTIONER_SEEDS=$PWD/outputs/experiments/rule-questioner-4b-large/data/questioner-selfplay.parquet
export KQAPRO_RELATION_CATALOG=$PWD/data/processed/kqapro/kqapro-v03-full-audit/relation_catalog.json
export GRAPHTASK_KQAPRO_DB=$PWD/data/processed/kqapro/kqapro-v03-full-audit/graph.sqlite
```

然后用一个 Bash 进程依次启动三轮 Questioner/Solver，共六个彼此独立的 Python 进程：

```bash
bash scripts/run_rule_questioner_selfplay_phases.sh \
  configs/training/selfplay_qwen3_4b_rule_questioner_large.yaml \
  outputs/selfplay/rule-questioner-qwen3-4b-10k
```

不传参数时，脚本默认就是上面的 4B 配置和输出目录。该配置每轮运行 1,024 个 Questioner
episode、2,048 个 Solver episode，使用 4 路 opponent/rollout；默认 GPU 拓扑为 3 张 actor GPU
加 1 张独立 SGLang opponent GPU。0.6B/64 条配置仍可通过显式传入
`configs/training/selfplay_qwen3_0_6b_kqapro_contract_large.yaml` 使用。

4B 配置将 `curriculum_min_archive_growth=128` 作为观测目标，不再把它当作必须精确达到的固定
数量：有 1--127 条新任务时记录 `shortfall_allowed`；本轮为 0 时记录 `empty_backfill`，并以
certified base pool 回填 Solver 数据继续运行。两种情况都会保留结构化 admission/rejection
统计，便于区分 `TOO_HARD`、`TARGET_MISMATCH` 等根因。格式损坏、Program-first 固定程序缺失、
Questioner 阶段未完成等契约错误仍会中止，避免悄悄训练错误数据。

脚本会在启动前检查 adapter、base tasks、Program-first Questioner seeds、relation catalog 和 graph
DB。六阶段的顺序、失败后的继续策略和 phase manifest 恢复规则与
`scripts/run_selfplay_curriculum_phases.sh` 相同；整个脚本可安全重跑，已完成阶段会自动 no-op。
不要并发启动两个脚本实例，因为六阶段共享 archive、manifest、GPU 和 opponent 端口。

该脚本负责训练阶段隔离，不会把最后 checkpoint 自动视为最佳 checkpoint。每轮 Solver 完成后仍应
按照 `docs/KQAPRO_TRAINING.md` 的固定 held-out 流程运行 `kqapro-promote
--require-promotion`；只有 exact match、F1、tool success 均不退化且至少一项严格提升时，才将
该 Solver 作为下一轮或最终模型。
