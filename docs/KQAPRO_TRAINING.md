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

GraphScript v0.3 覆盖实体解析、关系遍历、交并、类型/字面量/qualifier 过滤、attribute 与
qualifier 查询、relation 与 qualifier 查询、verify、extrema、count 和 emit。canonical trace
只用于认证和 replay，不写入 GraphScript SFT completion，因此无需为了训练压缩 trace。

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

先按仓库约束运行 bounded smoke；它使用独立输出目录，不会覆盖正式产物：

```bash
python -m graphtask_r1.cli data prepare \
  --dataset kqapro --raw-dir "$KQAPRO_RAW" --output-dir "$KQAPRO_SMOKE_DIR" \
  --splits train,val --limit 100 --seed 42 --workers 1
```

smoke 通过后构建完整 train/val。SQLite snapshot 同时保存 relation/attribute facts 及其
qualifiers；输入哈希和 converter version 匹配时会安全复用已有图。

```bash
python -m graphtask_r1.cli data prepare \
  --dataset kqapro --raw-dir "$KQAPRO_RAW" --output-dir "$KQAPRO_DIR" \
  --splits train,val --seed 42 --workers 1

for split in train val; do
  python -m graphtask_r1.cli data audit \
    --input "$KQAPRO_DIR/$split/tasks.parquet" --kind task --deep \
    --training-view-output "$KQAPRO_DIR/$split/training_tasks.parquet"
done
```

relation catalog 从 train task 确定 snapshot，再读取该 snapshot 的完整 graph schema。它不是
从 val 问题或程序统计得到的。

```bash
python -m graphtask_r1.cli data build-relation-catalog \
  --input "$KQAPRO_DIR/train/training_tasks.parquet" \
  --snapshot kqapro-v1 --scope graph \
  --output "$KQAPRO_DIR/relation_catalog.json"
```

检查 `metrics.json`、各 split 的 `metrics.json` 和 `manifest.json`，确认没有未解释的 rejection，
且 snapshot、source hash、seed 和 limit 符合本次实验记录。

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

以上是单 GPU bounded smoke 参数。确认 parse、execution、reward 分量和显存后，再增加 GPU、
rollout 数与 completion 上限。正式 server 模式的启动方式见 [训练手册](TRAINING.md)。

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

发布代码前运行：

```bash
make lint
make typecheck
make test
```
