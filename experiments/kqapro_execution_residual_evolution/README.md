# KQAPro Execution-Residual Evolution（ERE）独立实验

本目录实现一条不修改现有 `graphtask_r1`、self-play 配置或已晋级 checkpoint 的实验线：先从
一个固定 Solver snapshot 在 **train** 上的输出定位“最早执行分歧”，再按分歧频率只改 certified
GraphScript 的一个语义槽，双执行筛出非空且答案不同的可执行反事实候选，再用 DPO 拉开正确程序与该反事实
程序的条件概率；候选对 incumbent 的 hardness 需要另测 log-prob margin，不能由构造规则预先断言。

```text
train Solver 输出
  -> gold/predicted 前缀逐步执行与单调对齐
  -> 最早分歧槽画像
  -> 单槽反事实 GraphScript
  -> chosen/rejected 同图同预算双执行认证
  -> DPO Solver 更新
  -> 未见 dev A/B
  -> 仅 dev 通过后一次 blind A/B
```

算法和相关工作的边界见 [INNOVATION.md](INNOVATION.md)。本目录之外的文件不是该实验的运行
产物；失败实验也不得覆盖当前 adapter 或修改已有结果文档。

当前已登记的 mini 运行因有效 residual 为零而属于 default single-slot ablation，dev128 与 incumbent
同为 8/128 EM，未晋级；完整证据和停止原因见 [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md)。
随后完成的 444-pair 同记录数扩量诊断仍为 8/128，仅 tool success 从 9/128 增至 11/128；因此
同样未晋级，证据与原因见 [SCALE444_FAILURE_ANALYSIS.md](SCALE444_FAILURE_ANALYSIS.md)。

## 0. 固定环境和路径

以下命令都从仓库根目录执行。当前可复现实验使用 `ms-swift==3.10.3` 环境、Qwen3-0.6B 基础
模型和现有 checkpoint-56 incumbent。

```bash
export PROJECT_ROOT=$PWD
export ERE_ROOT=$PWD/experiments/kqapro_execution_residual_evolution
export ERE_OUTPUT=$PWD/outputs/experiments/kqapro-execution-residual-evolution
export KQAPRO_DIR=$PWD/data/processed/kqapro/kqapro-v03-full-audit
export GRAPHTASK_KQAPRO_DB=$KQAPRO_DIR/graph.sqlite
export PYTHONPATH=$ERE_ROOT/src:$PWD
export ERE_BIN=$PWD/.conda/ms-swift-debug/bin
export ERE_PYTHON=$ERE_BIN/python
export ERE_SWIFT=$ERE_BIN/swift
export ERE_RESPONSE_PREFIX=$'<think>\n\n</think>\n\n'
export ERE_EVAL_SPLIT=$ERE_OUTPUT/eval-split

export BASE_MODEL=$PWD/outputs/selfplay-qwen3-0.6b-smoke/model
export INCUMBENT_ADAPTER=$PWD/outputs/experiments/rule-questioner-contract-large/selfplay-reward-fix/solver_update_v2/v0-20260826-193019/checkpoint-56
export INCUMBENT_MERGED=$PWD/outputs/experiments/rule-questioner-contract-large/selfplay-reward-fix/merged-step56
export SOLVER_ROWS=$PWD/outputs/experiments/rule-questioner-contract-large/selfplay-run/solver.parquet
export SOLVER_COMPLETIONS=$PWD/outputs/experiments/rule-questioner-contract-large/selfplay-reward-fix/solver_update_v2/v0-20260826-193019/completions.jsonl

mkdir -p "$ERE_OUTPUT"
```

本实验的准备、测试、训练和评测只使用上面两个 `.conda/ms-swift-debug` 可执行文件，不混用系统
Python、另一 Conda 环境或 `PATH` 中的 Swift。若没有当前环境，按仓库
`docs/MS_SWIFT_CUDA_12_4.md` 的版本建立到该路径；测试依赖也安装到同一个解释器，然后检查版本：

```bash
"$ERE_PYTHON" -m pip install -r "$PROJECT_ROOT/requirements-dev.txt"
"$ERE_PYTHON" -c \
  'import swift, torch, yaml; print(swift.__version__, torch.__version__)'
"$ERE_SWIFT" rlhf --help >/dev/null
test -f "$BASE_MODEL/config.json"
test -f "$INCUMBENT_ADAPTER/adapter_model.safetensors"
test -f "$INCUMBENT_MERGED/config.json"
```

## 1. KQAPro 数据准备

当前 `kqapro-v03-full-audit` 产物是请求分层抽样 20,000 条 train 后得到的 19,941 条 certified
train，以及完整 11,768 条 val。以 Parquet 实际行数和 manifest 为准，不沿用旧文档中的“5k”
描述。

已有数据时只需检查：

```bash
test -s "$KQAPRO_DIR/graph.sqlite"
test -s "$KQAPRO_DIR/relation_catalog.json"
test -s "$KQAPRO_DIR/train/tasks.parquet"
test -s "$KQAPRO_DIR/val/tasks.parquet"
```

从原始 KQAPro 重建同一规模数据时执行：

```bash
export KQAPRO_RAW=$PWD/data/raw/kqa_pro

# 仅当 KQAPRO_RAW 中没有 kb.json、train.json、val.json 时执行。
"$ERE_PYTHON" -m graphtask_r1.cli data fetch \
  --dataset kqapro --raw-dir "$PWD/data/raw"

"$ERE_PYTHON" -m graphtask_r1.cli data prepare \
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

gold answer 仍由数据转换时的 certified program 执行得到。ERE 构造 pair 时会再次执行 positive，
并要求执行结果与任务 gold 完全一致；不会让模型生成或猜测 gold。

## 2. 校验不可变的未见 dev/blind

本次实验的 canonical split 是 `$ERE_EVAL_SPLIT`，已经排除冻结时 `outputs` 下所有历史
`predictions.parquet` 出现过的 task ID，再按 gold terminal op、program cost 和 answer kind 分层，
固定了 128 条 dev 与 128 条 blind。它是 **immutable** 输入：manifest 已存在时禁止再次运行
`freeze-eval`，只校验 canonical dev/blind 当前文件和下面两个已登记 SHA-256。重新扫描活跃
`outputs` 会把后续 ERE 预测也纳入排除集合，从而产生另一个 split，不能用于同一次 A/B。

```bash
test -s "$ERE_EVAL_SPLIT/eval_manifest.json"
test -s "$ERE_EVAL_SPLIT/dev128.parquet"
test -s "$ERE_EVAL_SPLIT/blind128.parquet"
printf '%s  %s\n' \
  967b3d6494d12c531208a6a46588300a686fe6198bd1fe0bc83000fdb450b514 \
  "$ERE_EVAL_SPLIT/dev128.parquet" | sha256sum --check -
printf '%s  %s\n' \
  7c3c9b1a2bfc589517a1861b5710001af1579d8cad251230b105ffd83067c55e \
  "$ERE_EVAL_SPLIT/blind128.parquet" | sha256sum --check -
```

只有创建一个全新、尚无任何 ERE 评测产物的实验实例时，才可在其新 output 目录运行一次下面的
生成命令；对于当前 `$ERE_OUTPUT` 不要执行此段：

```bash
export NEW_ERE_OUTPUT=/absolute/path/to/a/new/empty/experiment-output
test ! -e "$NEW_ERE_OUTPUT/eval-split/eval_manifest.json"
"$ERE_PYTHON" -m kqapro_ere freeze-eval \
  --source "$KQAPRO_DIR/val/tasks.parquet" \
  --exclude-root "$PWD/outputs" \
  --output-dir "$NEW_ERE_OUTPUT/eval-split" \
  --dev-count 128 \
  --blind-count 128 \
  --seed 4242
```

新生成的 `eval_manifest.json` 保存源文件 hash、排除根、排除 task ID 集合 hash、冻结 task ID、
分层计数和两份冻结文件 hash。当前 canonical manifest 在新增 exclusion hash 字段前已经冻结，
所以只含排除数量 128；本轮另做了历史 prediction task-ID 零交集审计，但不回写该不可变 manifest。
当前实例中其他同内容目录也不是 canonical 路径。blind 在 dev 门槛通过前不得用于推理、误差分析
或 checkpoint 选择。

## 3. 从固定 Solver snapshot 的 train 输出挖残差画像

画像只读取 train self-play 产生的 `solver.parquet` 和 ms-swift `completions.jsonl`，不读取 val/dev/
blind。完整 ERE 要求 `completions.jsonl` 来自同一个已登记权重 hash 的 incumbent snapshot；对同一
question 有重复采样时，固定取最后一个 `(step, ordinal)` 输出。仓库当前登记的
`$SOLVER_COMPLETIONS` 是训练过程中跨 step 的历史轨迹，只能用于验证管线和默认单槽 ablation，
不能据此声称“当前 incumbent 残差驱动”。正式运行应先用 `$INCUMBENT_MERGED` 在固定 train 子集上
重新采样，并把模型、子集和 completion artifact 的 hash 写入实验记录后再替换该变量。

```bash
"$ERE_PYTHON" -m kqapro_ere mine-residuals \
  --tasks "$SOLVER_ROWS" \
  --completions "$SOLVER_COMPLETIONS" \
  --graph-db "$KQAPRO_DIR/graph.sqlite" \
  --relation-catalog "$KQAPRO_DIR/relation_catalog.json" \
  --output-dir "$ERE_OUTPUT/residual-profile-mini" \
  --seed 4242 \
  --max-edge-visits 200 \
  --max-returned-entities 1000

test -s "$ERE_OUTPUT/residual-profile-mini/residual_profile.json"
test -s "$ERE_OUTPUT/residual-profile-mini/residuals.jsonl"
```

`residuals.jsonl` 保留每条输出的 parse/execution 状态、完整 gold/predicted output-prefix 操作、
denotation 对齐和最早分歧；`residual_profile.json` 汇总结构化 reason、槽频率、候选观测值、预算、
输入/输出 hash，并分别给出 `localized_residual_count` 与可直接驱动 mutation 的
`effective_signal_count`。只有 gold/predicted operation 相同时才比较语义字段；唯一跨算子例外是
`intersect`/`union` 映射到 `set_operator`，其他跨算子差异显式记为 `OPERATOR_MISMATCH`。该画像是
self-evolution 的反馈，不是一个额外神经模块。

只有当画像中的有效 residual 能定位到受 mutation 支持的槽，并且 residual priority 被 pair builder
实际用于候选优先级时，本实验才称为完整 ERE。若没有 `RESIDUAL_LOCALIZED` 信号、有效槽候选为空，
或构造只能回退到默认槽顺序/随机选择，则该运行只能标为 **random/default single-slot ablation**；
不得据此归因 ERE，也不得进入 full ERE 或主张完整创新链的效果。

## 4. 单槽、双执行反事实 pair

mini 先构造 128 对。每个 accepted pair 都满足：positive 来自 certified program 且重新执行与 gold
一致；negative 只改变一个 relation/direction/qualifier/attribute/comparator/literal/mode/
set-operator/type/query 槽；两者使用同一 SQLite snapshot、relation catalog 和执行预算；两者均产生
非空答案且 denotation 不同。

```bash
"$ERE_PYTHON" -m kqapro_ere build-pairs \
  --tasks "$KQAPRO_DIR/train/tasks.parquet" \
  --graph-db "$KQAPRO_DIR/graph.sqlite" \
  --relation-catalog "$KQAPRO_DIR/relation_catalog.json" \
  --residual-profile "$ERE_OUTPUT/residual-profile-mini/residual_profile.json" \
  --output-dir "$ERE_OUTPUT/pairs-mini" \
  --count 128 \
  --seed 4242 \
  --max-edge-visits 200 \
  --max-returned-entities 1000

test -s "$ERE_OUTPUT/pairs-mini/train.jsonl"
test -s "$ERE_OUTPUT/pairs-mini/counterfactual_certificates.jsonl"
test -e "$ERE_OUTPUT/pairs-mini/mutation_attempt_rejections.jsonl"
test -s "$ERE_OUTPUT/pairs-mini/manifest.json"
```

`train.jsonl` 使用 ms-swift DPO 的 `messages` + `rejected_response` schema；certificate 记录两个规范
程序签名、两个答案集合以及唯一 edit。候选拒绝 ledger 逐条只保存 slot、operation index、候选值
hash、失败 stage 和 reason code，不保存原始候选值。manifest 对 ledger 做 hash，并明确记录
`generation_mode` 与 `residual_guided`：`effective_signal_count == 0` 时实现会忽略画像候选与优先级，
固定标为 `dual_executed_default_single_slot_ablation`。不要通过复制 pair 补足数量，以 manifest 的
`selected_pairs` 为准。

### 4.1 与 contract-large 记录数对齐的 444-pair 诊断

`scale444` 是纯扩量诊断：把 128 个 pair 扩到与 contract-large SFT 相同的 444 条训练记录，
同时保持 mini 的 seed、1 epoch、effective batch 4、学习率、DPO beta 和 LoRA 参数不变。它不使用
`full.yaml`，因为后者还会改变 epoch 和 effective batch，无法隔离数据量变量。DPO 每条同时含
chosen/rejected，故 444 条仅表示记录数对齐，不表示 token 或算力对齐。

现有 mini profile 是代码修正前的旧 schema，不能复用。下面在新目录重挖画像并构造 pairs，不覆盖
mini 的数据、manifest 或结果：

```bash
"$ERE_PYTHON" -m kqapro_ere mine-residuals \
  --tasks "$SOLVER_ROWS" \
  --completions "$SOLVER_COMPLETIONS" \
  --graph-db "$KQAPRO_DIR/graph.sqlite" \
  --relation-catalog "$KQAPRO_DIR/relation_catalog.json" \
  --output-dir "$ERE_OUTPUT/residual-profile-scale444" \
  --seed 4242 \
  --max-edge-visits 200 \
  --max-returned-entities 1000

"$ERE_PYTHON" -m kqapro_ere build-pairs \
  --tasks "$KQAPRO_DIR/train/tasks.parquet" \
  --graph-db "$KQAPRO_DIR/graph.sqlite" \
  --relation-catalog "$KQAPRO_DIR/relation_catalog.json" \
  --residual-profile "$ERE_OUTPUT/residual-profile-scale444/residual_profile.json" \
  --output-dir "$ERE_OUTPUT/pairs-scale444" \
  --count 444 \
  --seed 4242 \
  --max-edge-visits 200 \
  --max-returned-entities 1000

jq -e '
  .requested_pairs == 444 and
  .selected_pairs == 444 and
  .generation_mode == "dual_executed_default_single_slot_ablation" and
  .residual_guided == false and
  .residual_effective_signal_count == 0
' "$ERE_OUTPUT/pairs-scale444/manifest.json"
test "$(wc -l < "$ERE_OUTPUT/pairs-scale444/train.jsonl")" -eq 444
test "$(wc -l < "$ERE_OUTPUT/pairs-scale444/counterfactual_certificates.jsonl")" -eq 444
```

若 `residual_guided=false`，该运行必须继续称为扩大版
`dual_executed_default_single_slot_ablation`，不能归因于完整 ERE。由于 mini 已经查看过 canonical
dev128，scale444 在同一 dev 上只能与 incumbent/mini 做诊断性比较；即使数值提高，也不能据此打开
blind 或形成新的未见集主结论。

full 配置只扩大同一个算法的数据量，不增加模块：

```bash
"$ERE_PYTHON" -m kqapro_ere build-pairs \
  --tasks "$KQAPRO_DIR/train/tasks.parquet" \
  --graph-db "$KQAPRO_DIR/graph.sqlite" \
  --relation-catalog "$KQAPRO_DIR/relation_catalog.json" \
  --residual-profile "$ERE_OUTPUT/residual-profile-mini/residual_profile.json" \
  --output-dir "$ERE_OUTPUT/pairs-full" \
  --count 1024 \
  --seed 4242 \
  --max-edge-visits 200 \
  --max-returned-entities 1000
```

## 5. CPU/Toy 测试和训练 dry-run

先运行本目录测试与命令展开；这一步不加载模型或 GPU。

```bash
"$ERE_PYTHON" -m pytest -q "$ERE_ROOT/tests"

"$ERE_PYTHON" \
  "$ERE_ROOT/scripts/train_dpo.py" \
  --config "$ERE_ROOT/configs/mini.yaml" \
  --dry-run > "$ERE_OUTPUT/train-mini-invocation.json"
```

dry-run JSON 是可重放的 `swift rlhf --rlhf_type dpo` 参数、工作目录和本次覆盖的环境变量。mini 与
full 的 seed、incumbent reference model、数据、输出目录、batch 和 LoRA 参数分别由独立 YAML 持有，不读取模块级
随机状态。

## 6. mini DPO 训练

mini 只训练一个预注册候选：配置中的一轮训练结束 checkpoint。训练从已合并的 incumbent 开始，
因此 DPO 在关闭新 LoRA 时使用的 reference 就是进入本轮的 `pi_0`，而不是原始 base。不要根据
历史 64 条或 blind 反复挑 checkpoint。

```bash
"$ERE_PYTHON" \
  "$ERE_ROOT/scripts/train_dpo.py" \
  --config "$ERE_ROOT/configs/mini.yaml"
```

纯扩量的 444-pair 诊断使用独立配置和输出目录；`444 / 4 = 111` 个 optimizer steps，只登记最终
checkpoint：

```bash
"$ERE_PYTHON" \
  "$ERE_ROOT/scripts/train_dpo.py" \
  --config "$ERE_ROOT/configs/scale444.yaml" \
  --dry-run > "$ERE_OUTPUT/train-scale444-invocation.json"

"$ERE_PYTHON" \
  "$ERE_ROOT/scripts/train_dpo.py" \
  --config "$ERE_ROOT/configs/scale444.yaml"
```

训练完成后必须从 `dpo-scale444/*/logging.jsonl` 的 `last_model_checkpoint` 读取 candidate 路径，
不得根据 dev 选择中间 checkpoint。candidate 评测写入独立的
`eval-dev-ere-scale444-localnet/`，paired comparison 写入
`dev-paired-comparison-scale444.json`；可复用同一 canonical incumbent predictions，但不得覆盖 mini
评测目录。

训练完成后，从日志中的 `last_model_checkpoint` 取得精确路径并固定：

```bash
find "$ERE_OUTPUT/dpo-mini" -type f -name adapter_model.safetensors -print
export ERE_ADAPTER=/absolute/path/printed/by/the/training/log/checkpoint-N
test -f "$ERE_ADAPTER/adapter_model.safetensors"
```

更大规模仍是同一 DPO 目标：

```bash
"$ERE_PYTHON" \
  "$ERE_ROOT/scripts/train_dpo.py" \
  --config "$ERE_ROOT/configs/full.yaml" \
  --dry-run

"$ERE_PYTHON" \
  "$ERE_ROOT/scripts/train_dpo.py" \
  --config "$ERE_ROOT/configs/full.yaml"
```

full 只应在 pair manifest 的 `residual_guided=true` 且 mini 未见 dev 严格胜出后运行。

## 7. 同集 A/B：incumbent 与 ERE final

两侧必须使用同一个 `dev128.parquet`、同一 evaluator config、`model-stage=grpo`、seed=42、
`enable_thinking=false`、512 completion tokens、GraphScript v0.3 和相同执行预算。只有服务加载的
模型组合与 `served_model_name` 不同：baseline 是 `$INCUMBENT_MERGED`，candidate 是同一个 merged
incumbent 再加载 `$ERE_ADAPTER`。部署的 template、response prefix、长度、dtype、attention、seed
和 batch 上限必须完全相同。

终端 A 先部署 incumbent：

```bash
CUDA_VISIBLE_DEVICES=0 "$ERE_SWIFT" deploy \
  --model "$INCUMBENT_MERGED" \
  --model_type qwen3 \
  --template qwen3 \
  --response_prefix "$ERE_RESPONSE_PREFIX" \
  --max_length 32768 \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --seed 42 \
  --max_batch_size 2 \
  --served_model_name kqapro-incumbent \
  --infer_backend pt \
  --host 127.0.0.1 \
  --port 18100
```

服务就绪后，先确认 OpenAI-compatible model ID 正是评测将请求的名称：

```bash
curl -fsS http://127.0.0.1:18100/health
curl -fsS http://127.0.0.1:18100/v1/models
```

终端 B 评测 incumbent dev：

```bash
export KQAPRO_MODEL_URL=http://127.0.0.1:18100
export KQAPRO_MODEL=kqapro-incumbent

"$ERE_PYTHON" -m graphtask_r1.cli evaluate kqapro-val \
  --config configs/evaluation/kqapro_val_qwen3_0_6b_contract.yaml \
  --model-stage grpo \
  --input "$ERE_EVAL_SPLIT/dev128.parquet" \
  --output-dir "$ERE_OUTPUT/eval-dev-incumbent"
```

停止 incumbent 服务，用完全相同的部署参数启动 ERE adapter：

```bash
CUDA_VISIBLE_DEVICES=0 "$ERE_SWIFT" deploy \
  --model "$INCUMBENT_MERGED" \
  --adapters "$ERE_ADAPTER" \
  --model_type qwen3 \
  --template qwen3 \
  --response_prefix "$ERE_RESPONSE_PREFIX" \
  --max_length 32768 \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --seed 42 \
  --max_batch_size 2 \
  --served_model_name kqapro-ere \
  --infer_backend pt \
  --host 127.0.0.1 \
  --port 18100
```

```bash
curl -fsS http://127.0.0.1:18100/health
curl -fsS http://127.0.0.1:18100/v1/models
```

然后评测同一 dev：

```bash
export KQAPRO_MODEL=kqapro-ere

"$ERE_PYTHON" -m graphtask_r1.cli evaluate kqapro-val \
  --config configs/evaluation/kqapro_val_qwen3_0_6b_contract.yaml \
  --model-stage grpo \
  --input "$ERE_EVAL_SPLIT/dev128.parquet" \
  --output-dir "$ERE_OUTPUT/eval-dev-ere"
```

先做按 task ID 顺序配对的诊断；它会输出 candidate-only/baseline-only task、exact McNemar、固定
seed 的 paired bootstrap 95% CI、失败原因迁移和 operator 分桶：

```bash
"$ERE_PYTHON" -m kqapro_ere compare \
  --baseline "$ERE_OUTPUT/eval-dev-incumbent/predictions.parquet" \
  --candidate "$ERE_OUTPUT/eval-dev-ere/predictions.parquet" \
  --output "$ERE_OUTPUT/dev-paired-comparison.json" \
  --bootstrap-seed 4242 \
  --bootstrap-samples 10000
```

dev 晋级要求是 paired comparison 中 candidate correct count **严格更高**，同时 mean F1 与 tool
success 均不下降。promotion 的 baseline artifact 是实际 merged 目录，candidate artifact 参数是
实际 adapter 目录；完整 candidate provenance 必须在实验记录中写成
`base=$INCUMBENT_MERGED + adapter=$ERE_ADAPTER`。再用现有 promotion 命令让失败门槛返回非零状态：

```bash
"$ERE_PYTHON" -m graphtask_r1.cli evaluate kqapro-promote \
  --baseline "$ERE_OUTPUT/eval-dev-incumbent/metrics.json" \
  --candidate "$ERE_OUTPUT/eval-dev-ere/metrics.json" \
  --baseline-artifact "$INCUMBENT_MERGED" \
  --candidate-artifact "$ERE_ADAPTER" \
  --min-exact-delta 0.000000001 \
  --min-f1-delta 0 \
  --min-tool-success-delta 0 \
  --require-promotion \
  --output "$ERE_OUTPUT/dev-promotion.json"
```

只有上述命令成功时，才在冻结 blind 上分别部署 incumbent/ERE 并各评测一次；把两条 dev 评测
命令中的 input 改为 `$ERE_EVAL_SPLIT/blind128.parquet`，输出目录改为 `eval-blind-incumbent` 和
`eval-blind-ere`。先运行同一个 paired compare：

```bash
"$ERE_PYTHON" -m kqapro_ere compare \
  --baseline "$ERE_OUTPUT/eval-blind-incumbent/predictions.parquet" \
  --candidate "$ERE_OUTPUT/eval-blind-ere/predictions.parquet" \
  --output "$ERE_OUTPUT/blind-paired-comparison.json" \
  --bootstrap-seed 4242 \
  --bootstrap-samples 10000
```

随后用相同硬门槛运行：

```bash
"$ERE_PYTHON" -m graphtask_r1.cli evaluate kqapro-promote \
  --baseline "$ERE_OUTPUT/eval-blind-incumbent/metrics.json" \
  --candidate "$ERE_OUTPUT/eval-blind-ere/metrics.json" \
  --baseline-artifact "$INCUMBENT_MERGED" \
  --candidate-artifact "$ERE_ADAPTER" \
  --min-exact-delta 0.000000001 \
  --min-f1-delta 0 \
  --min-tool-success-delta 0 \
  --require-promotion \
  --output "$ERE_OUTPUT/blind-promotion.json"
```

历史最高记录 `5/64 = 7.8125%` 来自此前多次查看过的 first-64 结构 checkpoint 筛选，另一次独立
holdout 也只有 4/64 对 2/64 且差异不显著。它只能作为背景，**不能**与这里排除历史 task ID 后的
dev/blind 跨集合比较。有效结论只能来自本节同任务、同 prompt、同解码、同执行预算的 A/B。

## 8. 结果处理、merge 与发布边界

若 dev 没有严格胜出：停止，不打开 blind；记录 pair yield、slot 分布、训练 loss 和 dev 失败原因。
若 dev 胜出但 blind 未胜：记录 blind 反例与 paired task 差异，但不改当前实验文档、配置、模型
指针或“最佳精度”。只有 dev 和 blind 都通过时，才把结果写入本目录的 `RESULTS.md` 并申请晋级。

通过双门槛后，可把 ERE adapter 合并为一个新目录；不要写入 incumbent 目录：

```bash
export ERE_MERGED=$ERE_OUTPUT/merged-mini

CUDA_VISIBLE_DEVICES=0 "$ERE_SWIFT" export \
  --model "$INCUMBENT_MERGED" \
  --adapters "$ERE_ADAPTER" \
  --merge_lora true \
  --output_dir "$ERE_MERGED"

test -f "$ERE_MERGED/config.json"
test -f "$ERE_MERGED/tokenizer_config.json"
```

部署 merged 模型时不再叠加 LoRA：

```bash
CUDA_VISIBLE_DEVICES=0 "$ERE_SWIFT" deploy \
  --model "$ERE_MERGED" \
  --model_type qwen3 \
  --template qwen3 \
  --response_prefix "$ERE_RESPONSE_PREFIX" \
  --max_length 32768 \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --seed 42 \
  --max_batch_size 2 \
  --served_model_name kqapro-ere-merged \
  --infer_backend pt \
  --host 127.0.0.1 \
  --port 18100
```

```bash
curl -fsS http://127.0.0.1:18100/health
curl -fsS http://127.0.0.1:18100/v1/models
```

正式发布前仍按仓库约束运行：

```bash
PATH="$ERE_BIN:$PATH" make lint
PATH="$ERE_BIN:$PATH" make typecheck
PATH="$ERE_BIN:$PATH" make test
```
