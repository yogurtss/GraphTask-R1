# KQAPro 训练流程

本流程只覆盖 KQAPro：先用官方 train 做 Solver SFT，随后直接做 Questioner/Solver self-play，
最后只在官方 val 上选择 checkpoint。独立的 Solver-only GRPO 是可选 warm-up/消融，不是
self-play 前置依赖。KILT/OpenQA 使用 GraphScript v0.2 与独立
checkpoint，不进入本流程。

```text
默认：Solver SFT → Questioner/Solver self-play → KQAPro val 选模
可选：Solver SFT → Solver-only GRPO warm-up → self-play → KQAPro val 选模
```

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

### 2.1 Qwen3-8B 配置示例

训练入口不依赖固定参数量：基础模型由 `model_path` 和 `model_type` 选择，SFT、self-play
和可选 Solver-only GRPO 都采用
LoRA、BF16、gradient checkpointing，因此同一条代码路径可以配置 Qwen3-8B。仓库提供三份仅作
配置起点的示例：

- `configs/experiments/qwen3_8b_sft_ms_swift_cuda124.yaml`；
- `configs/experiments/qwen3_8b_solver_grpo_ms_swift_cuda124.yaml`（可选 warm-up）；
- `configs/evaluation/kqapro_val_qwen3_8b.yaml`。

示例使用官方 [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B)，按一台 4×H100 服务器
配置，并给出一个 `micro_batch_size=2` 的较大 batch 档位：

- SFT 使用全部 4 张 GPU，`4 × 2 × 8 = 64` 的全局有效 batch；
- self-play 内部 GRPO 或可选 Solver-only GRPO 使用 GPU 0 运行 rollout server、GPU 1–3 训练；
  训练 batch、采样 batch 和评测 batch
  分别为 24、24 和 12 completions，每个 prompt 采样 4 条 completion。

该档位没有经过真实训练验证，因此不是显存或吞吐保证；实际可用上限取决于 H100 显存规格、
CUDA/ms-swift/vLLM 版本、样本长度和 adapter 配置。
若模型已经在本地，将两个训练 YAML 的
`model_path` 改为本地模型目录即可，避免运行时访问 Hugging Face。

以下命令只解析 YAML、检查 batch 约束并打印将要执行的脚本和环境；不会启动 ms-swift、加载权重
或下载模型：

```bash
python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_8b_sft_ms_swift_cuda124.yaml --dry-run

python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_8b_solver_grpo_ms_swift_cuda124.yaml --dry-run
```

8B YAML 显式记录 `max_length`/`max_completion_length`、学习率、epoch、LoRA 参数和 GRPO
`vllm_mode`；launcher 会把这些字段传给现有脚本，环境变量仍具有最高优先级。两份 YAML 还启用
`scale_learning_rate_with_micro_batch: true`。其中 `learning_rate` 表示 micro batch 为 1 时的基准
LR，最终值按下式生成：

```text
micro_batch_size > 1 时：final LR = learning_rate × micro_batch_size
micro_batch_size = 1 时：final LR = learning_rate
```

因此当前 SFT dry-run 输出 `LR=2e-5`，GRPO 输出 `LR=2e-6`。通过环境变量修改
`MICRO_BATCH_SIZE` 也会重新计算；若同时显式设置环境变量 `LR`，则认为用户已经人工决定学习率，
不再自动缩放。正式运行前仍应先用目标 tokenizer 做长度 preflight，并从 bounded 数据开始；若
micro 2 显存不足，改回 1 后 LR 会自动回到基准值。不要通过自动截断丢掉 GraphScript 尾部。

未来实际运行 8B self-play 内部 GRPO 或可选 Solver-only GRPO 时，应先在一个终端用 GPU 0
启动 rollout server，再启动 learner；以下
只是部署示例，本次配置检查不会执行它：

```bash
export MODEL_PATH=Qwen/Qwen3-8B
export LORA_ADAPTER_PATH=/path/to/qwen3-8b-sft-adapter
export ROLLOUT_CUDA_VISIBLE_DEVICES=0
export ROLLOUT_TP_SIZE=1
export VLLM_MAX_MODEL_LEN=32768
bash scripts/rollout_ms_swift.sh
```

learner YAML 已固定 `TRAIN_CUDA_VISIBLE_DEVICES=1,2,3`，不会与 rollout GPU 重叠。修改 micro、
梯度累积或 `rollout_n` 后必须继续满足 GRPO batch 整除约束，并把 batch 变化作为单独实验记录。

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
若配置启用了 `scale_learning_rate_with_micro_batch`，`learning_rate` 是 micro batch 为 1 的基准；
micro 大于 1 时 launcher 会线性放大最终 `LR`。显式环境变量 `LR` 始终优先且不会再次缩放。

## 5. Questioner/Solver self-play

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

4096 是可复用的 seed pool，不代表每轮全部进入训练。默认 4×H100/24 小时预算配置每轮只按显式
seed 和 round index 确定性抽取 256 条 Questioner rows；完整 pool 用于跨轮保持覆盖度。

默认直接用 SFT adapter 初始化 self-play。Self-play val 必须使用 RL row schema，因此
需要先单独导出只读 Solver val。这只是数据格式转换，不会执行 Solver-only GRPO：

```bash
python -m graphtask_r1.cli data export-rl \
  --input "$KQAPRO_DIR/val/training_tasks.parquet" \
  --output "$KQAPRO_TRAINING/kqapro_graphscript_v03_solver_rl_val.parquet" \
  --roles solver --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog "$KQAPRO_DIR/relation_catalog.json" --seed 42

export INITIAL_ADAPTER=$PWD/outputs/sft/qwen3-4b-kqapro-v03/checkpoint-last
export BASE_TASKS=$KQAPRO_DIR/train/training_tasks.parquet
export VAL_DATA=$KQAPRO_TRAINING/kqapro_graphscript_v03_solver_rl_val.parquet
export QUESTIONER_SEEDS=$KQAPRO_TRAINING/kqapro_questioner_seeds.parquet
export KQAPRO_RELATION_CATALOG=$KQAPRO_DIR/relation_catalog.json

python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay/kqapro-v03 --dry-run
python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay/kqapro-v03
```

若 SFT 指标不足并运行了附录 A 的 Solver-only GRPO，可以把 `INITIAL_ADAPTER` 改为
对应 GRPO checkpoint。这是可选起点。比较“有/无 self-play”时，应固定同一个起始
adapter、base tasks、seed 和验证集。

每轮先用 `swift export --merge_lora true` 将基础模型与当前 adapter 合并到
`round_NNN/opponent_merged/`，再由 SGLang 直接加载这个完整模型作为冻结 opponent；不向 SGLang
传递 `--enable-lora` 或 `--lora-paths`。认证 Questioner 提案后才执行生成 gold，并按
base/archive/new 比例组装下一轮数据。round manifest 保存配置哈希、数据哈希、adapter 和版本；
只有配置完全一致时才能 `--resume`。

自动合并显式使用 `--load_args false`，并重新传入 `model_type`、`train_type=lora` 和 BF16。这样
`swift export` 不会从 adapter 的 `args.json` 恢复 SFT 时记录的 `external_plugins` 绝对路径；项目或
checkpoint 移动到新目录后，也不会因为旧插件目录不存在而在 `swift/utils/utils.py` 的 `py_dir`
检查处失败。保留原始 `args.json` 作为训练溯源，不要手工改写它。
Self-play 的正式预算字段位于 `configs/training/selfplay.yaml`：

```yaml
rounds: 3
questioner_episodes: 256
solver_episodes: 256
opponent_samples: 4
actor_gpus: "0,1,2"
opponent_gpus: "3"
max_completion_tokens: 4096
vllm_max_model_len: 16384
vllm_gpu_memory_utilization: 0.6
vllm_sleep_level: 1
micro_batch_size: 4
eval_batch_size: 8
gradient_accumulation_steps: 2
steps_per_generation: 4
rollout_n: 4
```

GPU 0–2 同时承担共享 actor 的训练和 colocate rollout；GPU 3 只运行冻结 Solver opponent。
GPU 3 会先完成本轮 LoRA 合并，合并进程退出后再启动 SGLang；合并日志位于
`round_NNN/logs/merge.log`，SGLang 日志位于 `round_NNN/logs/sglang.log`。每轮保留一份完整合并
模型，因此三轮运行需额外预留约三份 4B 模型权重的磁盘空间。
两组 GPU 必须非空、无重复且互不重叠，配置加载时会提前校验。每个 prompt 生成 4 条
completion。每轮最多包含 512 prompts、2048 条 actor
completions；只有通过 Questioner 基础认证的 completion 才会触发 opponent，理论上限为 4096 条
opponent completions。三轮上限分别为 6144 和 12288。`--dry-run` 的 `rollout_budget` 会打印这些
上限，便于在启动 GPU 作业前复核。

训练全局 batch 为 `3 × 4 × 2 = 24 completions`；采样全局 batch 为
`3 × 4 × 4 = 48 completions`，即每次采样包含 12 个 prompt；评测 batch 为
`3 × 8 = 24 completions`。相比 micro batch 2 的配置，训练有效 batch 保持 24 不变，但每卡
micro batch、单次采样和评测吞吐均扩大。4B 模型在 80GB H100 上默认使用 micro batch 4。
colocate vLLM 使用 60% 显存上限、16384 总上下文，并在反向阶段启用 `sleep_level=1` 释放 vLLM
cache。colocate 模式不传 `vllm_server_host/port`，因此不会尝试连接 `127.0.0.1:8000`；8000 只属于
显式 `VLLM_MODE=server` 的独立 rollout 服务。该档位面向 80GB H100 的较高利用率，不建议未经
实测直接提高到 0.8–0.9。

24 小时是这组 4B/4×H100 参数的目标预算，不是跨驱动、模型和数据环境的硬保证。先完成第 1 轮并
读取 `round_001/logs/ms_swift.log` 的实际耗时；若超过 7 小时，不要原样启动后两轮，应优先将
`questioner_episodes` 降到 128、`opponent_samples` 降到 2，仍不足时再把 `rounds` 降到 2。
不要先压缩 Solver base/archive/new 的 256 条预算，以免速度提升来自削弱 Solver 训练。

首轮同时用 `nvidia-smi dmon` 观察 GPU 0–2。若生成阶段 OOM，先把
`vllm_gpu_memory_utilization` 从 0.6 降到 0.5；若反向传播阶段 OOM，则改为
`micro_batch_size: 2`、`gradient_accumulation_steps: 4`，保持训练有效 batch 仍为 24。若训练和
生成阶段都长期有较多余量，可试验 `micro_batch_size: 6`，但这会把训练有效 batch 提到 36，需将其
作为单独实验重新核对学习率和收敛曲线。不要通过缩短 16384 总上下文来掩盖正常样本超长，除非
重新做过真实模板长度审计。

## 6. val 选模与提升判定

所有候选 checkpoint 都使用同一份
`kqapro_graphscript_v03_solver_rl_val.parquet`，报告至少以下指标：

- answer exact match/F1；
- GraphScript parse rate；
- certified execution rate；
- 按 KoPL/GraphScript operator family 分桶的准确率；
- completion tokens、operator count 和执行 latency；
- Questioner acceptance/rejection reason 与 Solver reward components。

判断 self-play 是否有效时，至少比较同一起始 adapter 下的 `0 round`、每个 self-play round 和
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
工具失败不回退；SFT、self-play 和可选 GRPO 都生成 GraphScript v0.3，由同一个有界执行器
访问图，并在 GraphScript
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

## 附录 A：可选的 Solver-only GRPO warm-up

默认主线不运行本附录。Self-play 自身仍通过 mixed-role GRPO 更新共享 LoRA；这里可选的
是 self-play 前额外的 Solver-only 强化学习作业。仅当 SFT Solver 的 GraphScript
parse/execution/F1 不稳定，或需要单独测量 Solver RL 增益时使用。如果使用，应在第 5 节
self-play 之前运行。

### A.1 数据与训练

GRPO train rows 只来自 KQAPro train；val rows 只供 rollout evaluation/checkpoint selection：

```bash
python -m graphtask_r1.cli data export-rl \
  --input "$KQAPRO_DIR/train/training_tasks.parquet" \
  --output "$KQAPRO_TRAINING/kqapro_graphscript_v03_grpo_train.parquet" \
  --roles solver --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog "$KQAPRO_DIR/relation_catalog.json" --seed 42

python -m graphtask_r1.cli data export-rl \
  --input "$KQAPRO_DIR/val/training_tasks.parquet" \
  --output "$KQAPRO_TRAINING/kqapro_graphscript_v03_solver_rl_val.parquet" \
  --roles solver --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog "$KQAPRO_DIR/relation_catalog.json" --seed 42

export MS_SWIFT_SFT_ADAPTER=$PWD/outputs/sft/qwen3-4b-kqapro-v03/checkpoint-last
export SOLVER_RL_TRAIN_DATA=$KQAPRO_TRAINING/kqapro_graphscript_v03_grpo_train.parquet
export SOLVER_RL_VAL_DATA=$KQAPRO_TRAINING/kqapro_graphscript_v03_solver_rl_val.parquet
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

运行完后，将第 5 节的 `INITIAL_ADAPTER` 指向
`outputs/grpo/qwen3-4b-kqapro-v03/checkpoint-last`。

### A.2 GRPO batch 设置

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
3 个 prompt；评测 batch 也是 12。launcher 会拒绝不合法组合：`steps_per_generation` 必须是
`gradient_accumulation_steps` 的整数倍，采样 batch 和评测 batch 都必须能被 `rollout_n` 整除。

单 GPU smoke 可使用 `ROLLOUT_N=2`；若把 `EVAL_BATCH_SIZE` 改为 2，也应保持
`ROLLOUT_N=2`。GRPO 的 LR 只随 `micro_batch_size` 缩放，不随 `num_gpus`、
`gradient_accumulation_steps` 或 `rollout_n` 自动变化。确认 parse、execution、reward 分量和显存
后，再增加 GPU、rollout 数与 completion 上限。正式 server 模式见 [训练手册](TRAINING.md)。

### A.3 可选：合并 LoRA checkpoint

SFT 和 GRPO 默认产出 LoRA adapter。需要独立模型目录时，应合并实际需要部署的 checkpoint；
GRPO 合并时不能误用其 SFT 初始 adapter：

```bash
export BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
export GRPO_ADAPTER=$PWD/outputs/grpo/qwen3-4b-kqapro-v03/checkpoint-last
export GRPO_MERGED=$PWD/outputs/merged/qwen3-4b-kqapro-grpo

CUDA_VISIBLE_DEVICES=0 swift export \
  --model "$BASE_MODEL" --adapters "$GRPO_ADAPTER" \
  --merge_lora true --output_dir "$GRPO_MERGED"
```

完整的产物检查、磁盘注意事项和 adapter/merged 等价性 smoke 见
[训练手册：合并 SFT/GRPO LoRA 权重](TRAINING.md#6-合并-sftgrpo-lora-权重)及
[评测与可视化 README：合并权重后的等价性检查](KQAPRO_EVAL_VIS_README.md#7-合并权重后的等价性检查)。
