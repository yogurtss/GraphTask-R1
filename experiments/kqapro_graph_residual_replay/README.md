# KQAPro Graph-Certified Residual Replay（GRR）

> 历史文档：本页记录 2026-08-31 的 GRR mini 实验。后续 ECDB v1/v2 已实际实现并评测；v2 虽在
> 狭义数值筛选中从 `4/128` 到 `5/128`，唯一净增是错误程序的 accidental exact，最终不晋级，
> control/blind 均未运行。最新结论与方法审查见
> [ECDB_FAILURE_ANALYSIS.md](ECDB_FAILURE_ANALYSIS.md)。

以下是当时与现有训练线隔离的实验：它不修改 `graphtask_r1`、已有 ERE 目录、incumbent checkpoint
或历史结果。GRR 从一个**固定的当前 incumbent 快照**在 train 问题上的 greedy 输出中取得真实失败，
在同一图快照、relation allowlist 和执行预算下验证失败，再把该题的 certified gold GraphScript 与
模型的原始错误响应组成 DPO pair。

```text
固定 incumbent + 固定 train probe
  -> 同题 greedy GraphScript
  -> 图执行定位 parse / execution / terminal semantic residual
  -> 同题 certified gold 与原始错误响应配对
  -> DPO 更新 Solver
  -> 新的未见 dev 上做 paired A/B
  -> 仅 dev 严格胜出后打开一次 blind
```

算法边界、论文/专利价值和可证伪假设见 [INNOVATION.md](INNOVATION.md)。本轮只有一个算法变量：
pair 从“历史混合快照、跨题的合成反事实”改成“当前固定快照、同题的真实执行残差”。不加入新的
retriever、critic、reward model、reranker 或推理时修正器。

当前 128-pair 受控运行已完成：新 dev128 上 incumbent 与 GRR 都是 `5/128` EM、F1 完全持平，
GRR tool success 从 `9/128` 增到 `10/128`，但 candidate-only 与 baseline-only 各 1 题，未满足
严格 EM 提升门槛，因此未打开 blind、未晋级。证据与后续算法判断见
[FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md)。

## 0. 环境与固定路径

以下命令均从仓库根目录执行。当前可复现实例使用仓库已有的 `ms-swift==3.10.3` 环境。

```bash
export PROJECT_ROOT=$PWD
export GRR_ROOT=$PWD/experiments/kqapro_graph_residual_replay
export GRR_OUTPUT=$PWD/outputs/experiments/kqapro-graph-residual-replay
export GRR_PYTHON=$PWD/.conda/ms-swift-debug/bin/python
export GRR_SWIFT=$PWD/.conda/ms-swift-debug/bin/swift
export GRR_RESPONSE_PREFIX=$'<think>\n\n</think>\n\n'
export PYTHONPATH=$GRR_ROOT/src:$PWD

export KQAPRO_DIR=$PWD/data/processed/kqapro/kqapro-v03-full-audit
export GRAPHTASK_KQAPRO_DB=$KQAPRO_DIR/graph.sqlite
export INCUMBENT_MERGED=$PWD/outputs/experiments/rule-questioner-contract-large/selfplay-reward-fix/merged-step56
export GRR_EVAL_SPLIT=$GRR_OUTPUT/eval-split

mkdir -p "$GRR_OUTPUT"
"$GRR_PYTHON" -c 'import swift, torch, yaml; print(swift.__version__, torch.__version__)'
test -f "$INCUMBENT_MERGED/model.safetensors"
```

如果该 Conda 环境不存在，按仓库 `docs/MS_SWIFT_CUDA_12_4.md` 创建，然后安装开发依赖：

```bash
"$GRR_PYTHON" -m pip install -r requirements-dev.txt
```

## 1. 数据准备

已有审计数据时只检查四个不可变输入：

```bash
test -s "$KQAPRO_DIR/graph.sqlite"
test -s "$KQAPRO_DIR/relation_catalog.json"
test -s "$KQAPRO_DIR/train/tasks.parquet"
test -s "$KQAPRO_DIR/val/tasks.parquet"
```

从原始 KQAPro 重建与 incumbent 大实验相同的数据规模时执行：请求分层抽样 20,000 条 train，实际
certified 行数以生成 manifest/Parquet 为准；当前审计产物为 19,941 train 和 11,768 val。

```bash
export KQAPRO_RAW=$PWD/data/raw/kqa_pro

# 仅在 KQAPRO_RAW 缺少 kb.json/train.json/val.json 时下载。
"$GRR_PYTHON" -m graphtask_r1.cli data fetch \
  --dataset kqapro \
  --raw-dir "$PWD/data/raw"

"$GRR_PYTHON" -m graphtask_r1.cli data prepare \
  --dataset kqapro \
  --raw-dir "$KQAPRO_RAW" \
  --output-dir "$KQAPRO_DIR" \
  --splits train,val \
  --train-sample-size 20000 \
  --verification-mode source \
  --trace-mode none \
  --max-witness-facts 0 \
  --seed 42 \
  --workers 1
```

gold answer 只来自 certified program 在图上的执行结果。GRR 构造 pair 前还会用相同图预算重新执行
chosen；执行结果与任务 gold 不完全一致的题不会被采样，也不会由模型生成 gold。

## 2. 固定新的未见 dev/blind

当前 canonical split 已生成在 `$GRR_EVAL_SPLIT`，冻结时递归排除了历史
`predictions.parquet` 以及已有 `eval_manifest.json` 中保留过的 task ID。按 terminal op、program
cost 和 answer kind 分层后，各保留 128 条 dev/blind。当前 hash 为：

```bash
printf '%s  %s\n' \
  00954b7bb0a8ee47761d73e212baa5c2abfbc121bfeb59c8f9b18558b011c347 \
  "$GRR_EVAL_SPLIT/dev128.parquet" | sha256sum --check -
printf '%s  %s\n' \
  077f8b83d9ebfb4aa8359f0dc4fdac557be888c5788a10042765412b061f30b6 \
  "$GRR_EVAL_SPLIT/blind128.parquet" | sha256sum --check -
```

`eval-split-INVALID-reserved-overlap` 是在完整保留集扫描修复前产生、且从未推理过的废弃审计产物；
不得用于评测。canonical manifest 已存在时 `freeze-eval` 会拒绝覆盖。只有创建全新实验实例时才运行：

```bash
export NEW_GRR_OUTPUT=/absolute/path/to/new/empty/grr-output
test ! -e "$NEW_GRR_OUTPUT/eval-split/eval_manifest.json"
"$GRR_PYTHON" -m kqapro_grr freeze-eval \
  --source "$KQAPRO_DIR/val/tasks.parquet" \
  --exclude-root "$PWD/outputs" \
  --output-dir "$NEW_GRR_OUTPUT/eval-split" \
  --dev-count 128 \
  --blind-count 128 \
  --seed 4242
```

blind 在 dev 晋级前不得进行推理、误差分析或 checkpoint 选择。

## 3. 启动固定 incumbent

rollout 与 baseline dev 必须来自同一个 merged artifact。终端 A：

```bash
CUDA_VISIBLE_DEVICES=0 "$GRR_SWIFT" deploy \
  --model "$INCUMBENT_MERGED" \
  --model_type qwen3 \
  --template qwen3 \
  --response_prefix "$GRR_RESPONSE_PREFIX" \
  --max_length 32768 \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --seed 42 \
  --max_batch_size 2 \
  --served_model_name kqapro-grr-incumbent \
  --infer_backend pt \
  --host 127.0.0.1 \
  --port 18110
```

终端 B：

```bash
curl -fsS http://127.0.0.1:18110/health
curl -fsS http://127.0.0.1:18110/v1/models
export GRR_MODEL_URL=http://127.0.0.1:18110
export GRR_MODEL=kqapro-grr-incumbent
```

## 4. 收集当前快照的同题残差并构造 pair

`rollout_mini.yaml` 固定选取 train 中 stable-hash 最前的 256 题，各采样一次 temperature=0 响应，
保存 trace ID、seed、模型/图/数据 hash 和结构化失败原因。它不读取历史跨 step completion。

```bash
"$GRR_PYTHON" -m kqapro_grr collect-rollouts \
  --config "$GRR_ROOT/configs/rollout_mini.yaml"

"$GRR_PYTHON" -m kqapro_grr build-pairs \
  --config "$GRR_ROOT/configs/rollout_mini.yaml"

test -s "$GRR_OUTPUT/mini/rollout_manifest.json"
test -s "$GRR_OUTPUT/mini/pairs/manifest.json"
test -s "$GRR_OUTPUT/mini/pairs/train.jsonl"
```

pair 按冻结的 probe rank 取前 128 个真实失败，不按错误阶段配 quota，也不补采样；少于 128 个时
停止该受控实验。chosen 是该题重新执行认证的 gold GraphScript，rejected 是模型原始错误响应（只移除
服务框架固定的空 think prefix），不会把解析失败“修成”合法 negative。每条 certificate 明确记录
`same_task=true`、执行状态、答案、程序签名和响应 hash。

## 5. 先记录同集 incumbent dev

保持终端 A 的服务不变：

```bash
export GRR_EVAL_INPUT=$GRR_EVAL_SPLIT/dev128.parquet
export KQAPRO_MODEL_URL=$GRR_MODEL_URL
export KQAPRO_MODEL=$GRR_MODEL

"$GRR_PYTHON" -m graphtask_r1.cli evaluate kqapro-val \
  --config "$GRR_ROOT/configs/eval_mini.yaml" \
  --model-stage grpo \
  --input "$GRR_EVAL_INPUT" \
  --output-dir "$GRR_OUTPUT/eval-dev-incumbent"
```

完成后停止 incumbent 服务，释放 GPU。

## 6. 训练

先确认训练命令。`train_mini.yaml` 与此前 ERE mini 的 optimizer contract 相同：128 records、1 epoch、
effective batch 4、learning rate `5e-7`、DPO beta `0.1`、LoRA rank 32。这样 A/B 只改变 pair 来源。

```bash
"$GRR_PYTHON" "$GRR_ROOT/scripts/train_dpo.py" \
  --config "$GRR_ROOT/configs/train_mini.yaml" \
  --dry-run

"$GRR_PYTHON" "$GRR_ROOT/scripts/train_dpo.py" \
  --config "$GRR_ROOT/configs/train_mini.yaml"
```

训练输出写到 `$GRR_OUTPUT/dpo-mini`，不会覆盖 incumbent。将最后一个 checkpoint 明确登记：

```bash
export GRR_ADAPTER=/absolute/path/to/GRR/final/checkpoint
test -s "$GRR_ADAPTER/adapter_model.safetensors"
```

## 7. 同集 candidate dev 与 paired gate

终端 A 用相同部署参数加载 `base + GRR adapter`：

```bash
CUDA_VISIBLE_DEVICES=0 "$GRR_SWIFT" deploy \
  --model "$INCUMBENT_MERGED" \
  --adapters "$GRR_ADAPTER" \
  --model_type qwen3 \
  --template qwen3 \
  --response_prefix "$GRR_RESPONSE_PREFIX" \
  --max_length 32768 \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --seed 42 \
  --max_batch_size 2 \
  --served_model_name kqapro-grr \
  --infer_backend pt \
  --host 127.0.0.1 \
  --port 18110
```

终端 B：

```bash
export KQAPRO_MODEL_URL=http://127.0.0.1:18110
export KQAPRO_MODEL=kqapro-grr

"$GRR_PYTHON" -m graphtask_r1.cli evaluate kqapro-val \
  --config "$GRR_ROOT/configs/eval_mini.yaml" \
  --model-stage grpo \
  --input "$GRR_EVAL_SPLIT/dev128.parquet" \
  --output-dir "$GRR_OUTPUT/eval-dev-grr"

"$GRR_PYTHON" -m kqapro_grr compare \
  --baseline "$GRR_OUTPUT/eval-dev-incumbent/predictions.parquet" \
  --candidate "$GRR_OUTPUT/eval-dev-grr/predictions.parquet" \
  --output "$GRR_OUTPUT/dev-paired-comparison.json"

"$GRR_PYTHON" -m graphtask_r1.cli evaluate kqapro-promote \
  --baseline "$GRR_OUTPUT/eval-dev-incumbent/metrics.json" \
  --candidate "$GRR_OUTPUT/eval-dev-grr/metrics.json" \
  --baseline-artifact "$INCUMBENT_MERGED" \
  --candidate-artifact "$GRR_ADAPTER" \
  --min-exact-delta 0.000000001 \
  --min-f1-delta 0 \
  --min-tool-success-delta 0 \
  --require-promotion \
  --output "$GRR_OUTPUT/dev-promotion.json"
```

晋级需同时满足：candidate correct count 严格更多、mean F1 不下降、tool success 不下降、请求失败为
零。历史 `8/128` 只作背景；由于 task 集不同，不能跨集合宣称提高，唯一有效比较是上述同一新 dev
上的 incumbent/candidate paired A/B。

若 dev 未通过：停止，不运行 blind，不更新现有最佳模型、正式实验表或 incumbent 指针；只新增本
目录的 `FAILURE_ANALYSIS.md`。只有 dev 通过才将两次评测输入改为 `blind128.parquet`，对 incumbent
和 candidate 各运行一次并重复相同 compare/promotion gate。blind 失败仍不晋级。

## 8. 测试与发布边界

```bash
ruff check "$GRR_ROOT"
MYPYPATH=$GRR_ROOT/src mypy \
  "$GRR_ROOT/src" "$GRR_ROOT/scripts/train_dpo.py"
PYTHONPATH=$GRR_ROOT/src:$PWD pytest \
  "$GRR_ROOT/tests" -q

PATH="$PWD/.conda/ms-swift-debug/bin:$PATH" make lint
PATH="$PWD/.conda/ms-swift-debug/bin:$PATH" make typecheck
PATH="$PWD/.conda/ms-swift-debug/bin:$PATH" make test
```

只有 dev 与 blind 都通过才新增 `RESULTS.md` 并申请晋级；在此之前，本目录只是隔离的候选算法与
可重放实验记录。
