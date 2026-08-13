# KQAPro 训练流程

本流程只覆盖 KQAPro：先用官方 train 做 Solver SFT，再做 GRPO 和 Questioner/Solver
self-play，最后只在官方 val 上选择 checkpoint。KILT/OpenQA 使用 GraphScript v0.2 与独立
checkpoint，不进入本流程。

## 1. 数据边界与已验证上界

- train 的 question、program、answer 可用于 SFT、GRPO、base task 和 self-play；
- val 的 question、program、answer 只用于评测和 checkpoint 选择，不进入训练、reward 或
  archive；
- KQAPro 使用同一张公开知识图谱。Questioner 可以随机采到 val 出现过的实体，因为采样不读取
  val 样本及其标签；这不是监督信息泄漏；
- 官方 test 没有可用标签，本流程不用 test；
- gold answer 始终由转换后的 certified program 执行产生，而不是直接复制原始答案。

当前实现覆盖 27/27 个 KoPL 函数。在完整 val 上，转换后程序执行结果与官方答案的 exact
agreement 为 11,768/11,797（99.7542%）；剩余 29 条均是 Count 表示语义差异。对均匀抽取的
512 条 val 样本，用本地 Qwen3.5-4B tokenizer 代理测得 SFT 总长度 p50 10,681、p95 10,906、
最大 11,542 tokens，均低于默认 32K。正式训练前仍须使用目标模型的真实 ms-swift template
执行下述 preflight。

推荐配置已经实际完整跑通：train 从 94,376 条分层选出 20,000 条，接受 19,941 条
（99.705%）；val 全量接受 11,768/11,797（99.754%）。train/val 的 88 条 rejection 全部是
`SOURCE_ANSWER_MISMATCH`，未出现转换、执行或 schema 错误。accepted train 与 val 均仍覆盖
27/27 个 KoPL 函数；两份 task parquet 的 deep audit 均为 0 invalid、0 duplicate，SFT 导出行数
分别为 19,941 和 11,768。

GraphScript v0.3 覆盖实体解析、关系遍历、交并、类型/字面量/qualifier 过滤、attribute 与
qualifier 查询、relation 与 qualifier 查询、verify、extrema、count 和 emit。canonical trace
只用于小规模诊断和 replay，不写入 GraphScript SFT completion，也不作为官方 SFT 样本的准入
条件。

## 2. 路径与环境

```bash
export KQAPRO_RAW=/mnt/g/datasets/GraphTaskDataset/raw/kqa_pro
export KQAPRO_DIR=$PWD/data/processed/kqapro/kqapro-v1
export KQAPRO_SMOKE_DIR=$PWD/data/processed/kqapro/kqapro-v1-smoke
export KQAPRO_TRAINING=$PWD/data/training
export GRAPHTASK_KQAPRO_DB=$KQAPRO_DIR/graph.sqlite

mkdir -p "$KQAPRO_TRAINING" outputs/preflight
```

`KQAPRO_RAW` 应包含 `kb.json`、`train.json` 和 `val.json`。CUDA 12.4、PyTorch 与
ms-swift 3.6.4 的安装见 [ms-swift 环境说明](MS_SWIFT_CUDA_12_4.md)。

## 3. 构建、认证与审计

先按仓库约束运行 bounded smoke；它使用完整 self-play verification gates 和 canonical trace，
但只检查 20 条，不会覆盖正式产物：

```bash
python -m graphtask_r1.cli data prepare \
  --dataset kqapro --raw-dir "$KQAPRO_RAW" --output-dir "$KQAPRO_SMOKE_DIR" \
  --splits train,val --limit 20 --train-sample-size 20 \
  --verification-mode full --trace-mode canonical --seed 42 --workers 1
```

smoke 通过后，从 94,376 条 train 中确定性分层抽取 20,000 条，并处理完整 val。分层键同时包含
KoPL operator set、terminal operator 和程序长度桶，先保证每个结构层至少一个样本，再按层大小
分配剩余额度；层内用 source ID、raw index 和显式 seed 的稳定哈希选择。SQLite snapshot 同时
保存 relation/attribute facts 及其 qualifiers；输入哈希和 converter version 匹配时会安全复用
已有图。

```bash
python -m graphtask_r1.cli data prepare \
  --dataset kqapro --raw-dir "$KQAPRO_RAW" --output-dir "$KQAPRO_DIR" \
  --splits train,val --train-sample-size 20000 \
  --verification-mode source --trace-mode none --seed 42 --workers 1

for split in train val; do
  python -m graphtask_r1.cli data audit \
    --input "$KQAPRO_DIR/$split/tasks.parquet" --kind task --deep \
    --training-view-output "$KQAPRO_DIR/$split/training_tasks.parquet"
done
```

`verification-mode=source` 仍然先将官方 KoPL 转为 typed program，再执行程序生成 gold，要求答案
非空并与官方 source answer 对账。它只跳过用于新生成 Questioner 题目的 necessity、shortcut 和
answer-leak 质量门。`trace-mode=none` 会保留一个空的 `traces.parquet` schema，训练不依赖它。
若确实需要全量 train，可传 `--train-sample-size 0`，但不推荐作为首轮 SFT。

relation catalog 从 train task 确定 snapshot，再读取该 snapshot 的完整 graph schema。它不是
从 val 问题或程序统计得到的。

```bash
python -m graphtask_r1.cli data build-relation-catalog \
  --input "$KQAPRO_DIR/train/training_tasks.parquet" \
  --snapshot kqapro-v1 --scope graph \
  --output "$KQAPRO_DIR/relation_catalog.json"
```

检查 `metrics.json`、各 split 的 `metrics.json`、`train/sampling.json` 和 `manifest.json`。
`sampling.json` 记录 source/selected rows、全部 strata 的配额、operator coverage、sampler version
和 seed；确认没有未解释的 rejection，且 snapshot、source hash 和配置符合本次实验记录。

## 4. Solver SFT

只导出 Solver，确保 train/val 使用完全相同的 GraphScript v0.3 prompt、operator set 和 catalog：

```bash
for split in train val; do
  python -m graphtask_r1.cli data export-sft \
    --input "$KQAPRO_DIR/$split/training_tasks.parquet" \
    --output "$KQAPRO_TRAINING/kqapro_graphscript_v03_sft_$split.parquet" \
    --roles solver --interaction-mode graphscript --graphscript-version 0.3 \
    --relation-catalog "$KQAPRO_DIR/relation_catalog.json" --seed 42
done
```

用目标 tokenizer/template 分别预检 train 和 val；`--require-all` 可避免静默接受超长或编码失败
样本：

```bash
for split in train val; do
  python scripts/preflight_ms_swift_sft.py \
    --input "$KQAPRO_TRAINING/kqapro_graphscript_v03_sft_$split.parquet" \
    --accepted-output "outputs/preflight/kqapro-$split-accepted.parquet" \
    --rejected-output "outputs/preflight/kqapro-$split-rejected.parquet" \
    --summary-output "outputs/preflight/kqapro-$split-summary.json" \
    --model Qwen/Qwen3-4B-Instruct-2507 --model-type qwen3 \
    --max-length 32768 --require-all
done
```

先 dry-run 核对实际脚本和环境，再启动：

```bash
export SFT_TRAIN_DATA=$PWD/outputs/preflight/kqapro-train-accepted.parquet
export SFT_VAL_DATA=$PWD/outputs/preflight/kqapro-val-accepted.parquet
export SFT_OUTPUT_DIR=$PWD/outputs/sft/qwen3-4b-kqapro-v03
export NUM_GPUS=4
export MAX_LENGTH=32768

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml --dry-run
python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml
```

### SFT batch 设置

直接修改 `configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml`：

```yaml
num_gpus: 4
micro_batch_size: 1
eval_batch_size: 1
gradient_accumulation_steps: 8
```

SFT 的全局有效 batch 为：

```text
num_gpus × micro_batch_size × gradient_accumulation_steps
```

默认是 `4 × 1 × 8 = 32`。`micro_batch_size` 是每张 GPU 每一步实际装入的样本数，对显存影响
最大；显存不足时保持为 1，通过增加 `gradient_accumulation_steps` 调整有效 batch。
`eval_batch_size` 只影响验证吞吐和显存，不参与训练有效 batch。

也可以在单次运行时用环境变量覆盖 YAML，无需修改文件：

```bash
export NUM_GPUS=2
export MICRO_BATCH_SIZE=1
export EVAL_BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=16
```

环境变量优先级高于 YAML；`--dry-run` 输出的 `environment` 是最终实际值。

## 5. Solver GRPO

GRPO train rows 只来自 KQAPro train；val rows 只供 rollout evaluation/checkpoint selection：

```bash
for split in train val; do
  python -m graphtask_r1.cli data export-rl \
    --input "$KQAPRO_DIR/$split/training_tasks.parquet" \
    --output "$KQAPRO_TRAINING/kqapro_graphscript_v03_grpo_$split.parquet" \
    --roles solver --interaction-mode graphscript --graphscript-version 0.3 \
    --relation-catalog "$KQAPRO_DIR/relation_catalog.json" --seed 42
done

export MS_SWIFT_SFT_ADAPTER=$PWD/outputs/sft/qwen3-4b-kqapro-v03/checkpoint-last
export SOLVER_RL_TRAIN_DATA=$KQAPRO_TRAINING/kqapro_graphscript_v03_grpo_train.parquet
export SOLVER_RL_VAL_DATA=$KQAPRO_TRAINING/kqapro_graphscript_v03_grpo_val.parquet
export SOLVER_GRPO_OUTPUT_DIR=$PWD/outputs/grpo/qwen3-4b-kqapro-v03
export NUM_GPUS=1
export VLLM_MODE=colocate
export ROLLOUT_N=2
export MAX_COMPLETION_LENGTH=4096

python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo_ms_swift_cuda124.yaml --dry-run
python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo_ms_swift_cuda124.yaml
```

### GRPO batch 设置

GRPO 配置增加了五个明确字段：

```yaml
num_gpus: 3
micro_batch_size: 1
eval_batch_size: 4
gradient_accumulation_steps: 4
steps_per_generation: 4
rollout_n: 4
```

这里的 batch 单位是 completion，而不是原始 prompt：

```text
训练全局 batch = num_gpus × micro_batch_size × gradient_accumulation_steps
采样 batch     = num_gpus × micro_batch_size × steps_per_generation
每轮 prompt 数 = 采样 batch ÷ rollout_n
评测 batch     = num_gpus × eval_batch_size
```

默认训练和采样 batch 都是 12 completions，每个 prompt 生成 4 条 completion，因此每次采样包含
3 个 prompt；评测 batch 也是 12。launcher 会提前拒绝不合法组合：`steps_per_generation` 必须是
`gradient_accumulation_steps` 的整数倍，采样 batch 和评测 batch 都必须能被 `rollout_n` 整除。

单 GPU smoke 可以使用文档上面的 `ROLLOUT_N=2`；若显存不足并把 `EVAL_BATCH_SIZE` 改为 2，
也应保持 `ROLLOUT_N=2`。常用覆盖示例：

```bash
export NUM_GPUS=1
export MICRO_BATCH_SIZE=1
export EVAL_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=4
export STEPS_PER_GENERATION=4
export ROLLOUT_N=2
```

这些关系遵循 ms-swift 的 completion-level GRPO batch 定义。不要把 `rollout_n` 当成普通训练
batch；它表示同一个 prompt 的候选 completion 数。

以上是单 GPU bounded smoke 参数。确认 parse、execution、reward 分量和显存后，再增加 GPU、
rollout 数与 completion 上限。正式 server 模式的启动方式见 [训练手册](TRAINING.md)。

### 可选：把 LoRA checkpoint 合并为完整权重

SFT 和 GRPO 默认产出 LoRA adapter。需要独立模型目录用于 SGLang 部署、复制或归档时，应分别
合并对应 checkpoint；GRPO 合并时不能误用其 SFT 初始化 adapter：

```bash
export BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507

export SFT_ADAPTER=$PWD/outputs/sft/qwen3-4b-kqapro-v03/checkpoint-last
export SFT_MERGED=$PWD/outputs/merged/qwen3-4b-kqapro-sft
CUDA_VISIBLE_DEVICES=0 swift export \
  --model "$BASE_MODEL" --adapters "$SFT_ADAPTER" \
  --merge_lora true --output_dir "$SFT_MERGED"

export GRPO_ADAPTER=$PWD/outputs/grpo/qwen3-4b-kqapro-v03/checkpoint-last
export GRPO_MERGED=$PWD/outputs/merged/qwen3-4b-kqapro-grpo
CUDA_VISIBLE_DEVICES=0 swift export \
  --model "$BASE_MODEL" --adapters "$GRPO_ADAPTER" \
  --merge_lora true --output_dir "$GRPO_MERGED"
```

完整的产物检查、磁盘注意事项、adapter/merged 等价性 smoke 和部署方式见
[训练手册：合并 SFT/GRPO LoRA 权重](TRAINING.md#6-合并-sftgrpo-lora-权重)及
[评测与可视化 README：合并权重后的等价性检查](KQAPRO_EVAL_VIS_README.md#7-合并权重后的等价性检查)。

## 6. Questioner/Solver self-play

Questioner seeds 直接从完整 KQAPro 图按 degree 约束确定性采样。这里不传 `--exclude`：seed
选择既不打开 val 文件，也不读取 val question/program/answer。命令显式固定 v0.3，导出行也会
记录 `graphscript_version`、`operator_set` 和 `program_profile`。

```bash
python -m graphtask_r1.cli data sample-seeds \
  --snapshot kqapro-v1 \
  --output "$KQAPRO_TRAINING/kqapro_questioner_seeds.parquet" \
  --count 4096 --pool-limit 100000 --seed 42 \
  --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog "$KQAPRO_DIR/relation_catalog.json"
```

用选定的 Solver GRPO adapter 初始化 self-play；若只比较“有/无 self-play”，对照组应固定同一个
adapter、base tasks、seed 和验证集。

```bash
export INITIAL_ADAPTER=$PWD/outputs/grpo/qwen3-4b-kqapro-v03/checkpoint-last
export BASE_TASKS=$KQAPRO_DIR/train/training_tasks.parquet
export VAL_DATA=$SOLVER_RL_VAL_DATA
export QUESTIONER_SEEDS=$KQAPRO_TRAINING/kqapro_questioner_seeds.parquet
export KQAPRO_RELATION_CATALOG=$KQAPRO_DIR/relation_catalog.json

python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay/kqapro-v03 --dry-run
python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay/kqapro-v03
```

每轮冻结 opponent，认证 Questioner 提案后才执行生成 gold，并按 base/archive/new 比例组装下一轮
数据。round manifest 保存配置哈希、数据哈希、adapter 和版本；只有配置完全一致时才能 `--resume`。
Self-play 的同名 batch 字段位于 `configs/training/selfplay.yaml`，默认 actor GPU 为 2 张，因此
训练 batch 为 8、采样 batch 为 8、评测 batch 为 4，均可按 `rollout_n=4` 正确分组。

## 7. val 选模与提升判定

所有候选 checkpoint 都使用同一份
`kqapro_graphscript_v03_grpo_val.parquet`，报告至少以下指标：

- answer exact match/F1；
- GraphScript parse rate；
- certified execution rate；
- 按 KoPL/GraphScript operator family 分桶的准确率；
- completion tokens、operator count 和执行 latency；
- Questioner acceptance/rejection reason 与 Solver reward components。

判断 self-play 是否有效时，至少比较同一 SFT/GRPO 起点下的 `0 round`、每个 self-play round 和
最佳 round；不要用 val 调 prompt、生成训练样本或回填 archive。最终保留 checkpoint、配置、
manifest、preflight summary 和 val 指标，才能把提升归因到 self-play，而不是数据或算子变化。

### 单模型评测与任意多模式对比

复制并修改 `configs/evaluation/kqapro_val.yaml`。每次命令只连接一个兼容
`/v1/chat/completions` 的模型服务；`model.model` 可以填写服务加载的本地 checkpoint/adapter 路径
或注册名。配置中的 `input_path` 固定为 certified `val/tasks.parquet`，`relation_catalog` 固定为
仅由 train graph schema 构建的 catalog。用 `--model-stage base|base_tool|sft|grpo` 声明当前
评测协议，并分别写入输出目录。`base` 与 `base_tool` 使用同一个原模型 checkpoint：前者严格直接
回答，后者接收函数说明/few-shot 并生成 GraphScript。

```bash
export KQAPRO_MODEL_URL=http://127.0.0.1:18100

# 每次先令 KQAPRO_MODEL 指向当前服务实际加载的 checkpoint，再运行一条。
export KQAPRO_MODEL=Qwen/Qwen3-4B-Instruct-2507
python -m graphtask_r1.cli evaluate kqapro-val --model-stage base \
  --config configs/evaluation/kqapro_val.yaml --output-dir outputs/evaluation/kqapro-base

python -m graphtask_r1.cli evaluate kqapro-val --model-stage base_tool \
  --config configs/evaluation/kqapro_val.yaml --output-dir outputs/evaluation/kqapro-base-tool

export KQAPRO_MODEL=$PWD/outputs/sft/qwen3-4b-kqapro-v03/checkpoint-last
python -m graphtask_r1.cli evaluate kqapro-val --model-stage sft \
  --config configs/evaluation/kqapro_val.yaml --output-dir outputs/evaluation/kqapro-sft

export KQAPRO_MODEL=$PWD/outputs/grpo/qwen3-4b-kqapro-v03/checkpoint-last
python -m graphtask_r1.cli evaluate kqapro-val --model-stage grpo \
  --config configs/evaluation/kqapro_val.yaml --output-dir outputs/evaluation/kqapro-grpo
```

评测协议有意不同：base 只收到问题、严格答案格式与格式示例，不收到 relation catalog、图工具
结果或 gold 类型；base_tool 收到 GraphScript 函数说明、handle 规则、四类示例和 catalog，但
工具失败不回退；SFT 和 GRPO 生成 GraphScript v0.3，由同一个有界执行器访问图，并在 GraphScript
请求、解析、版本检查或执行失败时使用同一 checkpoint 再发一次严格直接回答 prompt。
`predictions.parquet` 会保留 `tool_attempted`、`tool_succeeded`、`fallback_used`、结构化
`rejection_reason`、程序步骤、support triples 和预算；各目录的 `metrics.json` 报告当前模型的
exact match/F1、工具成功率、回退率/回退准确率以及 operator 分桶结果。读取任意两个或更多目录的
`metrics.json` 即可比较相应模式。模型请求使用超时、重试、trace ID 和位于各自输出目录下的
可重放响应缓存。

至少两次兼容的单模式评测完成后，可只读取指标文件生成对比（不会再次调用模型）。GRPO 未训练时
可以不传：

```bash
python -m graphtask_r1.cli evaluate kqapro-compare \
  --metrics outputs/evaluation/kqapro-base/metrics.json \
            outputs/evaluation/kqapro-base-tool/metrics.json \
            outputs/evaluation/kqapro-sft/metrics.json \
  --output outputs/evaluation/kqapro-comparison.json
```

该命令会校验所有结果来自同一数据、split、graph snapshot 和样本数。传入 base 时默认以 base 为
基线；否则以第一份指标为基线，也可传 `--baseline-stage`。输出各模式精度和相对基线的 exact
match/F1 增量。

### CLI 浏览与独立路径可视化

可视化不随全量评测自动运行。先只浏览配置中的 val 数据，不调用模型：

```bash
python -m graphtask_r1.cli visualize kqapro \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage grpo \
  --indices 0,12,41 --inspect-only
```

确认问题后，只运行当前配置的一个模型并生成单文件静态 HTML：

```bash
python -m graphtask_r1.cli visualize kqapro \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage grpo \
  --indices 0,12,41 \
  --output-dir outputs/visualization/kqapro-grpo
```

不传 `--indices` 时默认只取前三条，可用 `--limit` 调整。命令行 JSON 会打印数据预览、当前模型的
答案/正确性、推理模式、回退状态、GraphScript operator 路径和失败原因；浏览器直接打开对应输出
目录的 `paths.html` 即可，无需部署前后端。HTML 同时展示执行步骤与最多 20 条 support triples。
也可以用 `--input` 临时覆盖配置中的数据集路径。

发布代码前运行：

```bash
make lint
make typecheck
make test
```
