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

## 0. 准备 4B 所需数据

所有命令从仓库根目录执行。这里只保留 4B 主流程必需的数据步骤：5,000 条 train、完整 val、
`graph.sqlite`、relation catalog 和约 10k 行的 1:1 baseline mixed SFT。文件名是
`graph.sqlite`，不是 `graph.slite`。

```bash
export PYTHONPATH=$PWD
export KQAPRO_RAW=$PWD/data/raw/kqa_pro
export KQAPRO_DIR=$PWD/data/processed/kqapro/kqapro-v03-full-audit
export SFT_DATA_DIR=$PWD/outputs/sft-data

# 仅在 KQAPRO_RAW 中还没有 kb/train/val/test JSON 时执行：
python -m pip install huggingface_hub
python -m graphtask_r1.cli data fetch \
  --dataset kqapro --raw-dir "$PWD/data/raw"

python -m graphtask_r1.cli data prepare \
  --dataset kqapro \
  --raw-dir "$KQAPRO_RAW" \
  --output-dir "$KQAPRO_DIR" \
  --splits train,val --train-sample-size 5000 \
  --verification-mode source --trace-mode none \
  --max-witness-facts 0 --seed 42 --workers 1

export TRAIN_TASKS="$KQAPRO_DIR/train/tasks.parquet"
export VAL_TASKS="$KQAPRO_DIR/val/tasks.parquet"
export WORK_DIR="$SFT_DATA_DIR"
export GRAPH_DB_PATH="$KQAPRO_DIR/graph.sqlite"
export MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
export MODEL_TYPE=qwen3
export SOLVER_RATIO=1
export QUESTIONER_RATIO=1
export MAX_LENGTH=32768
export SELFPLAY_SEED_COUNT=4096
export SEED=42

bash scripts/prepare_mixed_sft_data.sh

test -s "$KQAPRO_DIR/graph.sqlite"
test -s "$KQAPRO_DIR/train/tasks.parquet"
test -s "$KQAPRO_DIR/val/tasks.parquet"
test -s "$SFT_DATA_DIR/relation_catalog.json"
test -s "$SFT_DATA_DIR/preflight/mixed-train-accepted.parquet"
```

`prepare_mixed_sft_data.sh` 一次完成 deep audit、relation catalog、Solver/Questioner 导出、1:1
混合和真实模板长度预检。它不会重复数据；实际行数以 `outputs/sft-data/sft_data.env` 和 preflight
summary 为准。已有匹配图会自动复用，只有明确重建时才给 `data prepare` 添加 `--rebuild-graph`。

## 1. 采样实验

```bash
PYTHONPATH=. python scripts/experiment_path_sampler.py \
  --graph-db data/processed/kqapro/kqapro-v03-full-audit/graph.sqlite \
  --reference-tasks data/processed/kqapro/kqapro-v03-full-audit/train/tasks.parquet \
  --relation-catalog outputs/sft-data/relation_catalog.json \
  --attempts 1000 --seed 42 \
  --output-dir outputs/experiments/rule-path-sampler-n1000
```

采样期间默认每 5 秒向 stderr 打印一次结构化进度，包括当前 strategy、完成比例、proposal trials、
构造成功数、候选数、严格认证数及实时成功率；用 `--progress-interval-s 10` 可调整刷新间隔。
最终 comparison JSON 仍单独写到 stdout，方便重定向或交给其他脚本解析。

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
  --relation-catalog outputs/sft-data/relation_catalog.json \
  --attempts 10000 --seed 42 \
  --strategies family_balanced \
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

输出包括：

- `questioner-sft.parquet`：固定 Program 到真实问题的 4B Questioner SFT；
- `mixed-sft.parquet`：保留 baseline 全部 Solver，只等量替换 Questioner；
- `questioner-selfplay.parquet`：4,096 条固定且严格认证的 Program-first seeds。

Questioner reward 会从任意服务端前缀中提取首个完整的 `{"question":"..."}`，因此
`</tool_call>{"question":...}` 不会再被记为 `NON_JSON`；对象 schema、问题/程序对齐、答案泄漏和
认证状态仍分别记录。

## 3. 使用 Program-first mixed SFT 训练 4B

第 2 节只生成训练数据，不会自动启动 SFT。正式 A/B 使用
`outputs/experiments/rule-questioner-4b-large/data/mixed-sft.parquet`：其中 baseline Solver 行
保持不变，只把 Questioner 行替换为 Program-first Questioner SFT。训练仍使用主线的 Qwen3-4B、
GraphScript v0.3 和相同超参数，但写入独立输出目录，避免覆盖 baseline checkpoint。

先检查并显式设置输入输出：

```bash
export RULE_QUESTIONER_DATA=$PWD/outputs/experiments/rule-questioner-4b-large/data
export SFT_TRAIN_DATA=$RULE_QUESTIONER_DATA/mixed-sft.parquet
export SFT_OUTPUT_DIR=$PWD/outputs/sft/rule-questioner-qwen3-4b
export NUM_GPUS=4
export MAX_LENGTH=32768
export EVAL_STRATEGY=no
unset VAL_DATA SFT_VAL_DATA EVAL_STEPS EVAL_ROLLOUT_N

test -s "$SFT_TRAIN_DATA"
python scripts/validate_ms_swift_data.py \
  --kind sft --input "$SFT_TRAIN_DATA"
```

先 dry-run 检查最终配置和环境，再启动训练：

```bash
python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml \
  --dry-run

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml
```

默认配置为 2 epochs、LoRA rank/alpha `32/64`、learning rate `1e-5`，global batch 为
`4 × 1 × 8 = 32`，并通过 `force_disable_validation: true` 强制关闭训练期 validation；即使父 shell
残留 `EVAL_STRATEGY=steps` 或 `VAL_DATA` 也不会读取 val。训练日志和 checkpoint 写入
`SFT_OUTPUT_DIR`。
显存不足时应降低 micro batch 并相应提高 gradient accumulation，不要依靠截断丢掉 GraphScript
尾部。CUDA 12.4 和 ms-swift 环境安装见 `docs/MS_SWIFT_CUDA_12_4.md`。

ms-swift 不保证生成 `checkpoint-last`。训练完成后按修改时间解析真实的最新 checkpoint，并将它
作为第 6 节六阶段 self-play 的初始 adapter：

```bash
SFT_ADAPTER_FILE="$(find "$SFT_OUTPUT_DIR" -type f \
  -path '*/checkpoint-*/adapter_config.json' -printf '%T@ %p\n' \
  | sort -nr | head -n 1 | cut -d ' ' -f 2-)"
test -n "$SFT_ADAPTER_FILE"
export INITIAL_ADAPTER="$(dirname "$SFT_ADAPTER_FILE")"
test -f "$INITIAL_ADAPTER/adapter_config.json"
```

这里的 SFT adapter 是 LoRA 增量权重，不能脱离配置中的基础模型单独部署。A/B 时 baseline 和
Program-first 两侧应使用相同基础模型、seed、epoch、batch 和学习率，只改变 mixed SFT 数据。

## 4. 单样本 opponent 的 archive 门槛

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

## 5. contract-large A/B

两侧都使用 Qwen3-0.6B、444 条训练样本、1 epoch、222 steps、同一 256 条 Solver 数据和同一
64 条 KQAPro val：

| 数据 | exact/F1 | tool success | fallback |
|---|---:|---:|---:|
| 原 mixed SFT | 2/64 = 3.125% | 4/64 = 6.25% | 93.75% |
| Program-first Questioner SFT | 3/64 = 4.6875% | 5/64 = 7.8125% | 92.1875% |

这个 64 题结果显示没有精度退化且略有提升，但差异只有一个样本，不能视为统计显著。新模型的
主要剩余问题是 GraphScript schema/执行质量，而不是开头的 `</tool_call>`：本次 PT 部署中该前缀
出现 0 次，失败主要为 `NON_JSON`、`INVALID_SCHEMA` 和实体解析错误。

## 6. 六阶段 self-play 脚本

第 3 节 Program-first 4B SFT 完成并设置 `INITIAL_ADAPTER` 后，再设置六阶段配置引用的其他路径：

```bash
export BASE_TASKS=$PWD/outputs/sft-data/tasks/train.parquet
export QUESTIONER_SEEDS=$PWD/outputs/experiments/rule-questioner-4b-large/data/questioner-selfplay.parquet
export KQAPRO_RELATION_CATALOG=$PWD/outputs/sft-data/relation_catalog.json
export GRAPHTASK_KQAPRO_DB=$PWD/data/processed/kqapro/kqapro-v03-full-audit/graph.sqlite

test -f "$INITIAL_ADAPTER/adapter_config.json"
test -s "$BASE_TASKS"
test -s "$QUESTIONER_SEEDS"
test -s "$KQAPRO_RELATION_CATALOG"
test -s "$GRAPHTASK_KQAPRO_DB"
```

然后用一个 Bash 进程依次启动三轮 Questioner/Solver，共六个彼此独立的 Python 进程：

```bash
bash scripts/run_rule_questioner_selfplay_phases.sh \
  configs/training/selfplay_qwen3_4b_rule_questioner_large.yaml \
  outputs/selfplay/rule-questioner-qwen3-4b-10k
```

不传参数时，脚本默认就是上面的 4B 配置和输出目录。该配置每轮运行 1,024 个 Questioner
episode、2,048 个 Solver episode，使用 4 路 opponent/rollout；默认 GPU 拓扑为 3 张 actor GPU
加 1 张独立 SGLang opponent GPU。

Rule Questioner 的 opponent 请求保留 seed 中由认证程序导出的 relation 子集，不再把它覆盖为完整
KQA Pro relation catalog。单张 opponent GPU 默认最多同时执行 8 个 completion；wrapper 到 SGLang
的单次请求上限为 240 秒，训练 reward 到 wrapper 的上限为 300 秒。对应配置项是
`opponent_max_concurrency`、`opponent_model_request_timeout_s` 和
`opponent_request_timeout_s`。这些限制只控制服务容量和失败边界，不改变正式训练的
`opponent_samples=4` 难度分布。

4B 配置固定 `val_data: null`、`validation_samples: null` 和 `enable_grpo_validation: false`；六阶段
wrapper 还会强制设置 `EVAL_STRATEGY=no` 并清除 `VAL_DATA/EVAL_STEPS/EVAL_ROLLOUT_N`。因此 SFT
和三轮 self-play 的 Questioner/Solver 更新均不执行 val。第 0 节生成的 val 只供训练结束后的独立
held-out 评测与 checkpoint 晋级使用。

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
