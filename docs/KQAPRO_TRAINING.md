# KQAPro 训练流程

本流程只覆盖 KQAPro：先用官方 train 分别导出 Solver 与定量 Questioner SFT，再混合训练共享
adapter，随后直接做 Questioner/Solver self-play，
最后只在官方 val 上选择 checkpoint。独立的 Solver-only GRPO 是可选 warm-up/消融，不是
self-play 前置依赖。KILT/OpenQA 使用 GraphScript v0.2 与独立
checkpoint，不进入本流程。

```text
默认：Solver + Questioner SFT → Questioner/Solver self-play → KQAPro val 选模
可选：Solver + Questioner SFT → Solver-only GRPO warm-up → self-play → KQAPro val 选模
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
27/27 个 KoPL 函数；两份 task parquet 的 deep audit 均为 0 invalid、0 duplicate。Solver SFT 导出
行数分别为 19,941 和 11,768；按本手册另取 2,048 条 Questioner train 后，mixed train 应为
21,989 条，实际仍以 preflight 与 `combine-sft` metrics 为准。

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
ms-swift 3.10.3 的安装见 [ms-swift 环境说明](MS_SWIFT_CUDA_12_4.md)。

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

## 4. Solver + Questioner SFT

Solver 与 Questioner 使用独立导出入口和独立 Parquet。由于两种角色的 prompt 合约已同步更新，
不要复用旧 SFT Parquet；Solver train/val 也需要重新生成。现在使用一个入口完成 audit、catalog、
两种角色导出、比例计算、混合及模板预检：

```bash
export TRAIN_TASKS="$KQAPRO_DIR/train/tasks.parquet"
export VAL_TASKS="$KQAPRO_DIR/val/tasks.parquet"
export WORK_DIR=$PWD/outputs/sft-data
export GRAPH_DB_PATH="$KQAPRO_DIR/graph.sqlite"
export MODEL_PATH=/path/to/local/model
export SOLVER_RATIO=1
export QUESTIONER_RATIO=1

bash scripts/prepare_mixed_sft_data.sh
source "$WORK_DIR/sft_data.env"
```

默认 `1:1` 让 Solver 与 Questioner 获得相同训练曝光。脚本先导出全部 Solver 候选行，
混合时再从较多的角色无放回下采样。若实验要求固定 Questioner 目标数量，可在开头设置：

```bash
export QUESTIONER_COUNT_OVERRIDE=2048
```

Questioner SFT 不再只选单 seed，而是对所有具有显式 entity roots 的认证任务做固定 seed
的随机无放回抽样。单根、双根和更高阶多根任务都可进入；只有 0-root、依赖 `all_entities`
起步的任务暂不进入 Questioner。若 1:1 目标超过唯一 explicit-root 容量，脚本使用全部
合格 Questioner，再无放回下采样 Solver 到相同数量；不重复数据，且最终角色比例仍为 1:1。

脚本同时生成 `questioner-seeds.parquet` 供 self-play 使用。它保留真实多 entity 分布，不设置默认
root 数硬上限；每个 root 只暴露 ID、label、type 和有界 incoming/outgoing relation IDs，不包含
source program、相邻实体或 gold answer。每个 root 都必须使用
`resolve_entity(query=<entity_id>, match="id", limit=1)`，SSA 输出 handle 可按实际程序依赖自由分配。
自然语言 prompt 不再使用“v0.3”作为说明性术语，只在机器可解析 JSON 的 `"version":"0.3"` 字段中
保留版本。32K 模板预检和 GRPO context 负责拦截真正过长的样本。

导出 metrics 会统计真实 train 与最终 Questioner 行的精确 entity 数、terminal、路径长度、
答案类型、operator 覆盖率、strata 占比和
`distribution_total_variation`。Self-play reward 使用 seed metadata 中隐藏的 `source_stratum`，按
root bucket、terminal、程序长度、operator Jaccard 和答案类型计算 `target_alignment`；它不向模型
暴露 source program，但会把合成路径拉回真实分布，并作为 Questioner 曲线写入每轮 report。

训练集混合后一次预检，rejected Parquet 仍保留每行角色，便于区分 Solver/Questioner 问题；
`--require-all` 会在任何一行失败时停止，避免比例静默漂移。val 始终保持 Solver-only，用于选择 QA
checkpoint，不让验证指标随 Questioner 混合比例变化。底层独立命令只在排障时使用。

先 dry-run 核对实际训练脚本和环境，再启动：

```bash
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

Questioner seeds 从 certified train tasks 的显式 roots 导出，与 SFT 共用真实结构分布采样器。
必须用当前代码重新生成旧 seed Parquet，因为新格式保留多 roots，还额外记录
answer-free 的 label、type、局部 relation IDs 与隐藏 `source_stratum`。该命令只读 train，不打开
val 文件，也不把 source program 或 gold answer 写入 prompt。

```bash
python -m graphtask_r1.cli data export-questioner-seeds \
  --input "$KQAPRO_DIR/train/training_tasks.parquet" \
  --output "$KQAPRO_TRAINING/kqapro_questioner_seeds.parquet" \
  --count 4096 --seed 42 \
  --max-seed-neighbor-facts 200 --max-seed-relations 64 \
  --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog "$KQAPRO_DIR/relation_catalog.json"
```

4096 是可复用的 seed pool，不代表每轮全部进入训练。若请求数超过 unique explicit-root
容量，导出器直接按唯一容量截断并记录 `shortfall`。裸 `Q...` ID 对图后端有意义，但对模型缺少
语义，因此 prompt 同时提供实体 label/type 和最多 64 个有界局部 relation IDs，不暴露邻居实体或
答案。默认 4×H100/24 小时预算配置每轮只按显式 seed 和 round index 确定性抽取 256 条
Questioner rows；完整 pool 用于跨轮保持覆盖度。

默认直接用 mixed-role SFT adapter 初始化 self-play。Self-play val 必须使用 RL row schema，因此
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
questioner_episodes: 192
solver_episodes: 320
questioner_reward_weight: 1.0
solver_reward_weight: 1.0
opponent_samples: 4
actor_gpus: "0,1,2"
opponent_gpus: "3"
max_completion_tokens: 4096
vllm_max_model_len: 32768
vllm_gpu_memory_utilization: 0.6
vllm_sleep_level: 1
deepspeed: zero2
rl_algorithm: reinforce_plus_plus
micro_batch_size: 4
eval_batch_size: 4
validation_samples: 256
gradient_accumulation_steps: 2
steps_per_generation: 4
rollout_n: 4
```

GPU 0–2 同时承担共享 actor 的 LoRA 训练和 colocate rollout；GPU 3 只运行冻结 Solver opponent。
GPU 3 会先完成本轮 LoRA 合并，合并进程退出后再启动 SGLang；合并日志位于
`round_NNN/logs/merge.log`，SGLang 日志位于 `round_NNN/logs/sglang.log`。每轮保留一份完整合并
模型，因此三轮运行需额外预留约三份 4B 模型权重的磁盘空间。

Self-play 的 ms-swift 训练输出会同时实时打印到当前终端并完整写入
`round_NNN/logs/ms_swift.log`；merge、SGLang 和 frozen opponent 仍只写各自日志，避免多进程输出
相互干扰。每次训练 attempt 的逐 batch reward 分量按 rank 保存在
`round_NNN/logs/metrics_attempt_NNN/reward_components.rank-*.jsonl`，不会在重试时覆盖旧记录。
每轮完成后会更新根目录下的 `logs/selfplay_metrics.json`、`logs/round_metrics.jsonl`、
`logs/training_history.jsonl` 和 `logs/selfplay_curves.png`。图中包含 Questioner/Solver 未加权
score、两者较小值定义的 cooperation bottleneck、Questioner validity/frontier/novelty、Solver
F1/EM、frozen Solver 在新任务上的 success rate，以及 loss、gradient norm、KL、train/eval
reward 和 completion clipped ratio；聚合 JSON 还记录每轮 archive 增量。原始 JSONL 是审计依据，
PNG 只用于快速观察趋势。
若出现 `loss=0`、`reward_std>0`、`grad_norm=0` 或角色 reward 长期固定，按
[Self-play reward / zero-gradient 服务器排查手册](SELFPLAY_REWARD_DEBUG_README.md)逐项检查，尤其要
区分真实 action variance 与 frozen-opponent 随机噪声。
两组 GPU 必须非空、无重复且互不重叠，配置加载时会提前校验。每个训练 prompt 生成 4 条
completion。每轮最多包含 512 prompts、2048 条 actor
completions；只有通过 Questioner 基础认证的 completion 才会触发 opponent，理论上限为 3072 条
opponent completions。三轮上限分别为 6144 和 9216。`--dry-run` 的 `rollout_budget` 会打印这些
上限，便于在启动 GPU 作业前复核。

训练全局 batch 为 `3 × 4 × 2 = 24 completions`；采样全局 batch 为
`3 × 4 × 4 = 48 completions`，即每次采样包含 12 个 prompt；评测 batch 为
`3 × 4 = 12 completions`。相比 micro batch 2 的配置，训练有效 batch 保持 24 不变，但每卡
micro batch、单次采样和评测吞吐均扩大。4B 模型在 80GB H100 上默认使用 micro batch 4。
colocate vLLM 使用 60% 显存上限、32768 总上下文，并在反向阶段启用 `sleep_level=1` 释放 vLLM
cache。colocate 模式不传 `vllm_server_host/port`，因此不会尝试连接 `127.0.0.1:8000`；8000 只属于
显式 `VLLM_MODE=server` 的独立 rollout 服务。该档位面向 80GB H100 的较高利用率，不建议未经
实测直接提高到 0.8–0.9。

`deepspeed` 只控制 actor 训练状态的分布式管理，不改变 `train_type=lora`，也不作用于 GPU 3 的
SGLang opponent。正式配置默认使用 `zero2`。训练环境需要安装与当前 PyTorch/CUDA 兼容的
DeepSpeed；设为 `none` 时 launcher 不传 `--deepspeed`。支持的档位为：

| 配置 | 行为 | 建议用途 |
|---|---|---|
| `none` | 普通 DDP | 基线或排查 DeepSpeed 兼容性 |
| `zero0` | DeepSpeed engine，不做 ZeRO 分片 | 验证 DeepSpeed 链路 |
| `zero1` | 分片 optimizer state | 轻量节省 |
| `zero2` | 再分片 gradient | LoRA self-play 默认档位 |
| `zero3` | 再分片 model parameters | actor 参数显存仍不足时试验，通信更重 |
| `zero2_offload` | ZeRO-2 并向 CPU offload | GPU 紧张且 CPU 内存充足 |
| `zero3_offload` | ZeRO-3 并向 CPU offload | 最低 GPU 参数状态占用，速度最慢 |

LoRA 的 optimizer/gradient 本来就较小，因此 ZeRO-2 的节省通常不如 full fine-tuning 明显；长
prompt 的 activation、colocate vLLM KV cache 和 opponent 并发显存不会被 DeepSpeed 消除。生成阶段
OOM 仍应优先降低 `vllm_gpu_memory_utilization`/completion 上限；GPU 3 OOM 仍应降低
`opponent_samples` 或限制 opponent 并发。

24 小时是这组 4B/4×H100 参数的目标预算，不是跨驱动、模型和数据环境的硬保证。先完成第 1 轮并
读取 `round_001/logs/ms_swift.log` 的实际耗时；若超过 7 小时，不要原样启动后两轮，应优先将
`questioner_episodes` 降到 128、`opponent_samples` 降到 2，仍不足时再把 `rounds` 降到 2。
不要先压缩 Solver base/archive/new 的 256 条预算，以免速度提升来自削弱 Solver 训练。

首轮同时用 `nvidia-smi dmon` 观察 GPU 0–2。若生成阶段 OOM，先把
`vllm_gpu_memory_utilization` 从 0.6 降到 0.5；若反向传播阶段 OOM，则改为
`micro_batch_size: 2`、`gradient_accumulation_steps: 4`，保持训练有效 batch 仍为 24。若训练和
生成阶段都长期有较多余量，可试验 `micro_batch_size: 6`，但这会把训练有效 batch 提到 36，需将其
作为单独实验重新核对学习率和收敛曲线。不要通过缩短 16384 总上下文来掩盖正常样本超长，除非
重新做过真实模板长度审计。若 ZeRO-2 后仍在 actor 反向阶段 OOM，可以按 `zero3`、
`zero2_offload`、`zero3_offload` 的顺序分别测试；每个档位都应单独记录吞吐和峰值显存，不要把
offload 带来的速度下降误判为数据或 reward 变慢。

`rl_algorithm` 可设为 `grpo` 或 `reinforce_plus_plus`。launcher 始终使用同一个 GRPO trainer，分别
传入 `advantage_estimator=grpo`（组内归一化）或
`advantage_estimator=reinforce_plus_plus`（batch 归一化并把 KL 纳入 reward）。主线默认后者。
REINFORCE++ 不能凭空修复完全相同的 Solver reward，因此 Solver reward 同时改为分阶段信号：非法
JSON、schema/结构错误、执行失败、可执行但答错、部分正确和完全正确严格递增。两个角色默认使用
相同的 1.0 reward 尺度，再通过 192:320 的 prompt 预算让已经开始收敛的 Questioner 减少曝光，给
Solver 更多起步样本；如果切回 GRPO，其他数据和 reward contract 不变。

正式 val 不再逐轮跑完整的数千条集合。`validation_samples: 256` 使用固定 seed 从源 Parquet
确定性抽样一次，所有 round 共用同一子集；`validation_sample.json` 保存源文件哈希、抽样下标和
行数以便复现。`ms-swift==3.10.3` 的评测生成数沿用 `rollout_n: 4`；该版本尚不支持单独设置
`num_generations_eval`，不要向 launcher 添加这个参数。要恢复完整验证可设
`validation_samples: null`；最终候选 checkpoint 的正式比较仍应离线跑完整 val，而不是把频繁
训练期 val 当最终结果。

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
