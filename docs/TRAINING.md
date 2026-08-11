# ms-swift 训练手册

仓库只有一条训练运行时：ms-swift。SFT、GRPO 和 self-play 共享本地数据适配器
`graphtask_r1/training/ms_swift_plugin.py`；训练 Parquet 保持 GraphTask 自身的中立 schema，
加载时才转换为 ms-swift 的 `messages`、`tools` 和 reward 输入。

## 1. 训练前质量门

开始 GPU 作业前必须满足：

- `data audit` 无重复 task ID、损坏 JSON 或 certificate replay 错误；
- gold answer 全部由 certified program 执行产生；
- SFT 使用真实 tokenizer/template 完成长度预检；
- SFT、GRPO 与评测均记录 `interaction_mode=graphscript`、`graphscript_version=0.2`；
- KILT GRPO train/val 与 SSP 最终测试完全隔离；
- 先用 ToyGraph 或 bounded KILT smoke 数据跑通，再扩大数据和 GPU 数。

推荐目录：

```text
data/training/
├── kqapro_graphscript_v02_sft_train.parquet
├── kqapro_graphscript_v02_sft_val.parquet
└── kilt-certified-grpo-v2/
    ├── train/solver_grpo.parquet
    └── val/solver_grpo.parquet
```

## 2. SFT 长度预检

预检会调用与训练相同的 model type、Qwen3 template、Hermes agent template 和
`truncation_strategy=raise`。超过长度的样本不会被静默截断，而是写入带 reason code 的独立
Parquet。

```bash
python scripts/preflight_ms_swift_sft.py \
  --input data/training/kqapro_graphscript_v02_sft_train.parquet \
  --accepted-output outputs/preflight/kqapro-train-accepted.parquet \
  --rejected-output outputs/preflight/kqapro-train-rejected.parquet \
  --summary-output outputs/preflight/kqapro-train-summary.json \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --model-type qwen3 --max-length 32768
```

`--max-length` 支持 1–40960。若从 32K 提高到 40K，应重新运行预检并记录新的 summary；不要仅
修改训练参数后继续使用旧 accepted 文件。

## 3. SFT

```bash
export SFT_TRAIN_DATA=$PWD/outputs/preflight/kqapro-train-accepted.parquet
export SFT_VAL_DATA=$PWD/data/training/kqapro_graphscript_v02_sft_val.parquet
export SFT_OUTPUT_DIR=$PWD/outputs/sft/qwen3-4b-kqapro-v02
export NUM_GPUS=4
export MAX_LENGTH=32768

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml --dry-run

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml
```

脚本使用 LoRA、BF16、SDPA 和显式 seed。显存不足时依次降低 `MAX_LENGTH`、
`MICRO_BATCH_SIZE`，再提高 `GRADIENT_ACCUMULATION_STEPS`；不要让模板自动截断程序尾部。

## 4. KILT GRPO

先生成 bounded、可回放的数据：

```bash
export GRAPHTASK_KILT_DB=$PWD/data/processed/kilt/kilt-2019-08-01-v1/graph.sqlite

python -m graphtask_r1.cli data bootstrap-kilt-grpo \
  --output-dir data/training/kilt-certified-grpo-v2 \
  --count 20608 --pool-limit 1000000 --val-ratio 0.02484472 --seed 42 \
  --interaction-mode graphscript --graphscript-version 0.2
```

### colocate smoke test

```bash
export KQAPRO_SFT_ADAPTER=$PWD/outputs/sft/qwen3-4b-kqapro-v02/checkpoint-last
export KILT_GRPO_TRAIN_DATA=$PWD/data/training/kilt-certified-grpo-v2/train/solver_grpo.parquet
export KILT_GRPO_VAL_DATA=$PWD/data/training/kilt-certified-grpo-v2/val/solver_grpo.parquet
export KILT_GRPO_OUTPUT_DIR=$PWD/outputs/grpo/kilt-v02-smoke
export NUM_GPUS=1
export VLLM_MODE=colocate
export ROLLOUT_N=2
export MAX_COMPLETION_LENGTH=4096

python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_kilt_solver_grpo_ms_swift_cuda124.yaml --dry-run

python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_kilt_solver_grpo_ms_swift_cuda124.yaml
```

确认 smoke test 后再提高 completion length、generation 数和 GPU 数。正式 server 模式先在独立
GPU 上运行 `scripts/rollout_ms_swift.sh`，再以 `VLLM_MODE=server` 启动 GRPO。GraphScript 是单次
程序生成，不启用多轮 scheduler；只有显式 `INTERACTION_MODE=tool` 的消融实验才启用它。

## 5. Self-play

默认配置是 KILT + GraphScript v0.2：

```bash
export INITIAL_ADAPTER=$KQAPRO_SFT_ADAPTER
export BASE_TASKS=$PWD/data/training/kilt-certified-grpo-v2/train/tasks.parquet
export VAL_DATA=$KILT_GRPO_VAL_DATA
export QUESTIONER_SEEDS=$PWD/data/training/kilt_questioner_seeds.parquet
export KILT_RELATION_CATALOG=$PWD/data/training/kilt-certified-grpo-v2/relation_catalog.json
export KQAPRO_RELATION_CATALOG=$PWD/data/processed/kqapro/kqapro-v1/relation_catalog.json

python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay --dry-run
```

真实运行每轮会启动一个冻结的 SGLang opponent、组装 questioner/solver mixed Parquet、调用
ms-swift GRPO、查找新 LoRA adapter，并写入 round manifest。配置 hash、dataset hash、adapter
路径和 ms-swift 版本用于恢复；修改配置后不能从旧 manifest 继续。外部图调用保留 timeout、retry、
cache 和 trace ID。

## 6. 最终验证

HotpotQA、TriviaQA 和 NaturalQuestions 统一转换为 `BenchmarkExample`，但不转换为训练
`TaskCertificate`。启动冻结 solver 服务后：

```bash
python -m graphtask_r1.cli evaluate benchmark \
  --input data/processed/ssp/hotpotqa/examples.parquet \
  --output-dir outputs/eval/hotpotqa \
  --solver-url http://127.0.0.1:18080 \
  --snapshot kilt-2019-08-01-v1 --graphscript-version 0.2
```

除 EM/F1 外必须报告 program parse rate、execution rate、operator count、passage search count 和
latency，避免把“代码格式失败”“执行失败”和“程序语义错误”混为一类。

## 7. 入口索引

| 用途 | 文件 |
| --- | --- |
| SFT | `scripts/train_ms_swift_sft.sh` |
| GRPO | `scripts/train_ms_swift_grpo.sh` |
| rollout server | `scripts/rollout_ms_swift.sh` |
| SFT 模板预检 | `scripts/preflight_ms_swift_sft.py` |
| 数据加载与字段转换 | `graphtask_r1/training/ms_swift_data.py` |
| dataset/reward/scheduler 注册 | `graphtask_r1/training/ms_swift_plugin.py` |
| mixed-role round orchestration | `graphtask_r1/training/selfplay.py` |
