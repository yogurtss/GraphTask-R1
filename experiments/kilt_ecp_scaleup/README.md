# KILT ECP 模型与数据扩展实验

本目录只保存扩展实验的配置和协议，不复用或覆盖此前 0.6B smoke test 的输出目录。
训练框架固定为 `ms-swift==3.10.3`，直接从同一模型初始化进行 GRPO，不做 SFT。

## 研究问题

需要分别回答两个问题，不能把模型规模和数据规模同时改变后只报告一个结果：

1. 在原 48/16 划分上，模型从 0.6B 扩展到 4B 后，ECP-v4 是否仍超过
   Search-R1；
2. 固定 4B 后，把数据扩展到目标 384/128，优势是否能在三个种子上保持；
3. 只有第二项通过，才运行 8B 确认实验。

主要指标是 KILT-F1，Answer-F1 和 provenance R-precision 必须同时报告。
正式通过条件定义在 [protocol.yaml](protocol.yaml)：平均 KILT-F1 增量为正、至少
2/3 个种子为正、provenance 不下降，并且 Answer-F1 下降不超过 0.05。

## 目录

```text
kilt_ecp_scaleup/
├── README.md
├── protocol.yaml
└── configs/
    ├── qwen3_4b/
    │   ├── search_r1.yaml
    │   └── ecp_v4.yaml
    └── qwen3_8b/
        ├── search_r1.yaml
        └── ecp_v4.yaml
```

每个模型规模下的两个配置除奖励函数和 ECP reranker 外保持一致。所有配置均为
`initial_adapter: null`、`train_questioner: false`，所以不会隐式加载旧 SFT 或 0.6B
adapter。

## 执行顺序

### 1. 4B、原 48/16 数据的 scale-only gate

当前可直接复用的冻结数据为：

```bash
export GRAPHTASK_KILT_DB=data/processed/kilt/kilt-2019-08-01-500/graph.sqlite
export FROZEN_RECORDS=outputs/kilt-evidence-selfplay-0.6b/direct_rl_round_500_seed71/records.parquet
export FROZEN_SPLIT=outputs/kilt-evidence-selfplay-0.6b/direct_rl_round_500_seed71/split.json
export SCALE_ROOT=outputs/kilt_ecp_scaleup/qwen3_4b_scale_only/seed71
export EVIDENCE_SEED=71
```

从同一份 records 和 split 导出两个训练臂，只有 reward variant 不同：

```bash
PYTHONPATH=. python scripts/export_evidence_rl_records.py \
  --records "$FROZEN_RECORDS" \
  --split-manifest "$FROZEN_SPLIT" \
  --output-dir "$SCALE_ROOT/data/search_r1" \
  --seed "$EVIDENCE_SEED" \
  --solver-reward-variant search_r1_em

PYTHONPATH=. python scripts/export_evidence_rl_records.py \
  --records "$FROZEN_RECORDS" \
  --split-manifest "$FROZEN_SPLIT" \
  --output-dir "$SCALE_ROOT/data/ecp_v4" \
  --seed "$EVIDENCE_SEED" \
  --solver-reward-variant ecp_v4
```

4B 协议模型为 `Qwen/Qwen3-4B`。本机目前发现的是
`Qwen3-4B-Instruct-2507` 缓存；如果为了节省下载而改用它，两个训练臂必须使用
完全相同的 checkpoint，并把结果另记为 `qwen3_4b_instruct_2507`，不能与
`Qwen/Qwen3-4B` 混成同一组。

先 dry-run 检查展开后的训练计划：

```bash
export EVIDENCE_MODEL_PATH=Qwen/Qwen3-4B
export EVIDENCE_RERANKER_PATH=outputs/kilt-evidence-selfplay-0.6b/ecp_counterfactual_seed71/reranker.json

EVIDENCE_RL_DATA="$SCALE_ROOT/data/search_r1" \
EVIDENCE_RL_OUTPUT="$SCALE_ROOT/train/search_r1" \
PYTHONPATH=. python scripts/run_evidence_selfplay_rl.py \
  experiments/kilt_ecp_scaleup/configs/qwen3_4b/search_r1.yaml --dry-run

EVIDENCE_RL_DATA="$SCALE_ROOT/data/ecp_v4" \
EVIDENCE_RL_OUTPUT="$SCALE_ROOT/train/ecp_v4" \
PYTHONPATH=. python scripts/run_evidence_selfplay_rl.py \
  experiments/kilt_ecp_scaleup/configs/qwen3_4b/ecp_v4.yaml --dry-run
```

确认计划后去掉两个命令末尾的 `--dry-run` 即可训练。baseline 与 candidate
分别写入 `$SCALE_ROOT/train/search_r1` 和 `$SCALE_ROOT/train/ecp_v4`，不会覆盖旧实验。

### 2. 扩展到 384/128

当前 `kilt-2019-08-01-500` 子图只能产生已有的 64 个冻结挑战；14 GB 的
`graph.building.sqlite` 仍是构建中产物，不能作为正式实验快照。因此数据扩展阶段
应在完整图构建结束后生成 512 个冻结挑战，再固定 384/128 split。两个训练臂、
三个种子和后续 8B 都必须复用相同 challenge IDs。

对新的冻结 `records.parquet` 重复上面的导出步骤，并依次使用种子
`71, 113, 191`。ECP reranker 只使用新 split 的训练部分生成：

```bash
PYTHONPATH=. python scripts/export_ecp_counterfactuals.py \
  --records "$EXPANDED_RECORDS" \
  --split-manifest "$EXPANDED_SPLIT" \
  --graph-db "$GRAPHTASK_KILT_DB" \
  --output-dir "$SCALE_ROOT/counterfactual" \
  --seed "$EVIDENCE_SEED"

PYTHONPATH=. python scripts/train_ecp_counterfactual_reranker.py \
  --examples "$SCALE_ROOT/counterfactual/retriever_contrastive.parquet" \
  --graph-db "$GRAPHTASK_KILT_DB" \
  --output "$SCALE_ROOT/counterfactual/reranker.json" \
  --seed "$EVIDENCE_SEED"
```

不要用 validation answer、validation provenance 或最终测试指标训练 reranker。

### 3. 8B confirmation gate

只有 4B 的 384/128 三种子实验满足通过条件后，才使用
`configs/qwen3_8b/`。8B 两个训练臂从同一 `Qwen/Qwen3-8B` 初始化，复用 4B
阶段冻结的 challenges、split 和训练侧 reranker。当前未发现本地 8B 权重，配置
已经准备好，但本步骤会需要模型可用后再执行。

## 评测

从每次训练生成的 `direct_rl_update.json` 读取 `final_adapter`。Search-R1 评测不加
reranker 或 causal selection；ECP-v4 使用二者。两边固定 `max_turns=6`：

```bash
PYTHONPATH=. python scripts/run_kilt_interactive_eval.py \
  --examples "$FROZEN_RECORDS" \
  --split-manifest "$FROZEN_SPLIT" \
  --graph-db "$GRAPHTASK_KILT_DB" \
  --model "$EVIDENCE_MODEL_PATH" \
  --adapter "$SEARCH_R1_ADAPTER" \
  --max-turns 6 \
  --output "$SCALE_ROOT/eval/search_r1.json"

PYTHONPATH=. python scripts/run_kilt_interactive_eval.py \
  --examples "$FROZEN_RECORDS" \
  --split-manifest "$FROZEN_SPLIT" \
  --graph-db "$GRAPHTASK_KILT_DB" \
  --model "$EVIDENCE_MODEL_PATH" \
  --adapter "$ECP_V4_ADAPTER" \
  --counterfactual-reranker "$EVIDENCE_RERANKER_PATH" \
  --causal-selection \
  --max-turns 6 \
  --output "$SCALE_ROOT/eval/ecp_v4.json"
```

不要根据 validation 结果为某一个训练臂单独增加 turns、rollouts 或训练 epoch。
需要修改预算时，应创建一个新实验阶段，并同时重跑两个训练臂。

## 与旧实验的关系

此前 0.6B、ECP-v1—v5 和 16+32 退火实验均保留原输出路径。研究记录已集中归档到
[KILT/ECP research archive](../../docs/research/kilt_ecp/README.md)，扩展实验的新结论
应写入本目录的新报告，而不是继续追加到旧 smoke-test 文档中。
