# SFT 后 Questioner / Solver 能力可视化

这个探针按组顺序运行，每组包含：

- 1 个 Questioner seed prompt，生成 `candidates_per_role` 条候选；
- 1 个 Solver RL prompt，生成相同数量的候选；
- 每条候选使用训练时的真实 `compute_score` 计算 reward；
- 终端实时打印完整 completion、raw reward、训练 reward、阶段、拒绝原因和全部分量；
- 每完成一组就刷新独立 HTML，因此运行中也可以打开报告查看。

它不会使用 SFT target 作为模型输入。两个输入都必须是 RL row schema，避免把标准答案泄漏给模型。

## 1. 准备探针输入

Questioner 直接复用 self-play seed：

```bash
export QUESTIONER_SEEDS=$PWD/outputs/sft-data/questioner-seeds.parquet
```

Solver 输入单独从 certified val tasks 导出。不要把 `mixed-train-accepted.parquet` 或
`solver-val-accepted.parquet` 传给探针，因为它们是 SFT message schema：

```bash
python -m graphtask_r1.cli data export-rl \
  --input "$KQAPRO_DIR/val/training_tasks.parquet" \
  --output outputs/sft-data/solver-probe-rl.parquet \
  --roles solver --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog "$KQAPRO_DIR/relation_catalog.json" --seed 42

export SFT_PROBE_SOLVER_INPUT=$PWD/outputs/sft-data/solver-probe-rl.parquet
```

## 2. 启动 SFT 模型服务

先把 SFT LoRA 合并到基础模型，或使用能够直接加载该 LoRA 的 OpenAI-compatible 服务。探针连接
`/v1/chat/completions`：

```bash
export SFT_PROBE_MODEL_URL=http://127.0.0.1:18100
export SFT_PROBE_MODEL=/absolute/path/to/merged-sft-model
```

同一个 SFT checkpoint 同时用于 Questioner 生成和 Solver 生成，才能反映 mixed-role SFT 后的两种
能力。

## 3. 启动专用 opponent reward 服务

Questioner 的完整 reward 包含 frozen Solver pass rate 和 novelty，因此还需要一个 `/evaluate`
服务。让它连接同一个 SFT 模型端点，并使用专用 archive；不要指向正式 self-play archive：

```bash
python -m graphtask_r1.training.opponent \
  --model-url "$SFT_PROBE_MODEL_URL" \
  --model "$SFT_PROBE_MODEL" \
  --archive outputs/visualization/sft-capability/probe-archive.sqlite \
  --port 18080 \
  --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog "$KQAPRO_DIR/relation_catalog.json" \
  --max-follow-limit 100 --max-edge-visits 200 \
  --max-completion-tokens 4096

export SFT_PROBE_OPPONENT_URL=http://127.0.0.1:18080
```

使用专用 archive 很重要：Questioner reward 会把通过认证的新任务写入 archive，复用正式 archive
会改变后续 self-play 状态。

## 4. 按组生成并查看

```bash
python -m graphtask_r1.cli visualize sft-capability \
  --config configs/evaluation/sft_capability.yaml \
  --output-dir outputs/visualization/sft-capability \
  --groups 3
```

默认每个角色每组生成 4 条候选。探针严格串行处理组和候选，适合显存紧张时做 bounded smoke，且
所有采样 seed 和 trace ID 都显式记录。模型请求带超时、重试和本地 replay cache。

输出目录：

```text
outputs/visualization/sft-capability/
├── report.html       # 双列分组可视化；每组完成后刷新
├── results.json      # 完整 prompt、completion 和 reward 分量
├── results.parquet   # 可复用的结构化结果
├── summary.json      # 两个角色的聚合指标
├── probe-archive.sqlite
└── cache/
    └── sft-capability.json
```

直接打开 `report.html`。重点观察：

- Questioner 的 stage 0–6 分布、认证率和拒绝原因；
- 同一 prompt 的不同候选是否有不同 reward；
- Solver 的 F1、exact match、解析/执行失败；
- completion 是否多样，而不是相同输出因随机 opponent 得到不同 reward。

配置模板为 [sft_capability.yaml](../configs/evaluation/sft_capability.yaml)。
