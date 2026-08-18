# ms-swift 训练手册

仓库只有一条训练运行时：ms-swift。默认主线是 **SFT → self-play → val 选模**；独立的
Solver-only GRPO 只是可选 warm-up/消融。SFT、GRPO 和 self-play 共享本地数据适配器
`graphtask_r1/training/ms_swift_plugin.py`；训练 Parquet 保持 GraphTask 自身的中立 schema，
加载时才转换为 ms-swift 的 `messages`、`tools` 和 reward 输入。

## 1. 训练前质量门

开始 GPU 作业前必须满足：

- train/val 均通过一次 `data audit --kind task --deep --training-view-output`；
- bounded diagnostic 的 canonical trace replay 通过；正式 SFT 数据无需存 trace；
- gold answer 全部由 certified program 执行产生；
- SFT 使用真实 tokenizer/template 完成长度预检；
- SFT、GRPO 与评测均记录 `interaction_mode=graphscript`、`graphscript_version=0.3`；
- KQAPro train 用于 SFT/GRPO/self-play，KQAPro val 只用于模型选择；
- KILT v0.2 passage-search 路线与 KQAPro v0.3 完全隔离；
- 先用 ToyGraph 或 bounded KQAPro 数据跑通，再扩大数据和 GPU 数。

推荐目录：

```text
data/training/
├── kqapro_graphscript_v03_solver_sft_train.parquet
├── kqapro_graphscript_v03_solver_sft_val.parquet
├── kqapro_graphscript_v03_questioner_sft_train.parquet
├── kqapro_graphscript_v03_solver_rl_val.parquet
└── kqapro_graphscript_v03_grpo_train.parquet  # 仅可选 Solver GRPO 需要
```

Qwen3-8B 使用相同的配置驱动入口，无需修改训练代码。仓库已提供
`configs/experiments/qwen3_8b_sft_ms_swift_cuda124.yaml` 和
`configs/experiments/qwen3_8b_solver_grpo_ms_swift_cuda124.yaml`；参数含义、纯 `--dry-run`
检查和显存边界见 [KQAPro 训练流程：Qwen3-8B 配置示例](KQAPRO_TRAINING.md#21-qwen3-8b-配置示例)。
这些配置不会自行下载模型，只有去掉 `--dry-run` 后才会进入 ms-swift。
8B 示例当前使用 `micro_batch_size=2`；`learning_rate` 记录 micro 为 1 时的基准，launcher 在
`scale_learning_rate_with_micro_batch: true` 时线性放大最终 LR。显式环境变量 `LR` 可覆盖自动
结果，且不会被二次缩放。

## 2. SFT 长度预检

预检会调用与训练相同的 model type、Qwen3 template、Hermes agent template 和
`truncation_strategy=raise`。超过长度的样本不会被静默截断，而是写入带 reason code 的独立
Parquet。

预检同样按 row batch 流式处理，accepted/rejected 文件边检查边写，不会同时持有全量 token IDs
或整张 Arrow table。若长时间没有吞吐，优先检查模型/tokenizer 是否仍在下载，而不是继续增加
内存。

Solver 与 Questioner prompt 合约都已更新，必须从 certified tasks 重新导出，不能复用旧 Solver
Parquet 或旧 Questioner seed。主路径统一使用一键脚本；先修改脚本开头配置区，或用同名环境变量
覆盖：

```bash
export TRAIN_TASKS=$PWD/data/processed/kqapro/kqapro-v1/train/tasks.parquet
export VAL_TASKS=$PWD/data/processed/kqapro/kqapro-v1/val/tasks.parquet
export WORK_DIR=$PWD/outputs/sft-data
export GRAPH_DB_PATH=$PWD/data/processed/kqapro/kqapro-v1/graph.sqlite
export SOLVER_RATIO=1
export QUESTIONER_RATIO=1
export MODEL_PATH=/path/to/local/model

bash scripts/prepare_mixed_sft_data.sh
source "$WORK_DIR/sft_data.env"
```

比例按最终训练行数定义：默认 `1:1` 让 Solver 与 Questioner 获得相同训练曝光。脚本先导出全部
Solver 候选行，再根据实际可用数量无放回下采样较多的角色；设置正整数
`QUESTIONER_COUNT_OVERRIDE` 可改为固定的 Questioner 目标数量。
脚本会将角色隔离的原始文件、拒绝原因、summary 和最终 accepted 文件全部保存在 `WORK_DIR`，并以
`--require-all` 预检保证比例不会因静默过滤而漂移。底层 `data export-sft`、
`data export-questioner-sft` 和 `data combine-sft` 仍可用于单步排障。

Questioner SFT 不限定单 seed，但只对真实 explicit-root tasks 做固定 seed 的随机无放回抽样。
无显式实体根的任务仍被拒绝，因为 Questioner 禁止无界 `all_entities` 起步。若目标数超过
唯一容量，导出器使用全部合格 Questioner 并记录 `shortfall`；混合器同步下采样 Solver，
使最终行数严格满足 `SOLVER_RATIO:QUESTIONER_RATIO`。整个 SFT 不重复任何行。

脚本还生成 `questioner-seeds.parquet`；self-play seed 才按 root 数、operator set、程序节点数、
terminal operator 和答案类型做比例分层。它只保留 root metadata，不泄漏 source program 或
gold answer，默认不裁剪多 entity root 数。真实模板和 32K GRPO context 负责长度质量门。

sampling metrics 直接统计精确 entity 数、root bucket、terminal、路径长度、答案类型与
operator 覆盖率；同时为每个联合 stratum 写入 `source_fraction` 与 `final_fraction`，并用
`distribution_total_variation` 汇总整体偏差。Self-play seed 的 `source_stratum` 只进入 reward metadata，
不会拼进 prompt；认证后的生成程序得到 `target_alignment`，该分量参与 Questioner reward，并由
self-play report 单独绘图。这样保留 novelty/frontier 探索，同时避免合成路径长期漂离真实 train。

该接口不依赖 KQAPro 名称：任意后端只要实现 `GraphBackend`、提供 certified explicit-root tasks 与
relation catalog，即可复用同一个 Questioner 导出器。运行时 prompt 只描述通用 GraphScript 合约；
数据集/快照名称留在数据配置中。Questioner seed metadata 只包含 ID、label、type 和有界 relation
IDs，不包含邻居实体或答案。

`--max-length` 支持 1–40960。若从 32K 提高到 40K，应重新运行预检并记录新的 summary；不要仅
修改训练参数后继续使用旧 accepted 文件。

## 3. SFT

```bash
source "${WORK_DIR:-$PWD/outputs/sft-data}/sft_data.env"
export SFT_OUTPUT_DIR=$PWD/outputs/sft/qwen3-4b-kqapro-v03
export NUM_GPUS=4
export MAX_LENGTH=32768

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml --dry-run

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft_ms_swift_cuda124.yaml
```

脚本使用 LoRA、BF16、SDPA 和显式 seed。显存不足时依次降低 `MAX_LENGTH`、
`MICRO_BATCH_SIZE`，再提高 `GRADIENT_ACCUMULATION_STEPS`；不要让模板自动截断程序尾部。
上述 batch 参数已经是 experiment YAML 的正式字段，环境变量可临时覆盖；完整计算例子见
[KQAPro 训练流程](KQAPRO_TRAINING.md#sft-batch-设置)。

## 4. 可选：KQAPro Solver-only GRPO warm-up

默认流程可跳过本节，直接进入第 5 节。Self-play 内部仍使用 mixed-role GRPO 更新共享 LoRA；
这里“可选”的只是 self-play 之前额外运行的 Solver-only GRPO 作业。仅当 SFT Solver 的
GraphScript parse/execution/F1 不稳定，或需要单独测量 Solver RL 增益时，再运行本节。

从认证的 KQAPro train 导出 Solver GRPO 数据，并复用主线的只读 Solver RL val：

```bash
python -m graphtask_r1.cli data export-rl \
  --input data/processed/kqapro/kqapro-v1/train/training_tasks.parquet \
  --output data/training/kqapro_graphscript_v03_grpo_train.parquet \
  --roles solver --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog data/processed/kqapro/kqapro-v1/relation_catalog.json

python -m graphtask_r1.cli data export-rl \
  --input data/processed/kqapro/kqapro-v1/val/training_tasks.parquet \
  --output data/training/kqapro_graphscript_v03_solver_rl_val.parquet \
  --roles solver --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog data/processed/kqapro/kqapro-v1/relation_catalog.json
```

### colocate smoke test

```bash
export MS_SWIFT_SFT_ADAPTER=$PWD/outputs/sft/qwen3-4b-kqapro-v03/checkpoint-last
export SOLVER_RL_TRAIN_DATA=$PWD/data/training/kqapro_graphscript_v03_grpo_train.parquet
export SOLVER_RL_VAL_DATA=$PWD/data/training/kqapro_graphscript_v03_solver_rl_val.parquet
export SOLVER_GRPO_OUTPUT_DIR=$PWD/outputs/grpo/kqapro-v03-smoke
export NUM_GPUS=1
export VLLM_MODE=colocate
export ROLLOUT_N=2
export MAX_COMPLETION_LENGTH=4096

python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo_ms_swift_cuda124.yaml --dry-run

python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo_ms_swift_cuda124.yaml
```

确认 smoke test 后再提高 completion length、generation 数和 GPU 数。正式 server 模式先在独立
GPU 上运行 `scripts/rollout_ms_swift.sh`，再以 `VLLM_MODE=server` 启动 GRPO。GraphScript 是单次
程序生成，不启用多轮 scheduler；只有显式 `INTERACTION_MODE=tool` 的消融实验才启用它。
launcher 只在 server 模式传入 `VLLM_SERVER_HOST/PORT`；colocate 模式不会连接默认的
`127.0.0.1:8000`，而是在训练 GPU 上创建本地 vLLM engine。
GRPO 的训练、采样与评测 batch 约束见
[KQAPro 训练流程](KQAPRO_TRAINING.md#grpo-batch-设置)；不合法组合会在启动训练前报错。

这里的 rollout server 和 eval README 中的单个 `sglang.launch_server` 都不是 SGLang PD 分离。
`tp-size`/tensor parallel 也不等于 prefill/decode 分离。PD 需要独立 prefill、decode GPU 实例和
router，主要用于多 GPU 高吞吐服务；单 GPU 4B eval 应优先使用低 concurrency、bounded limit 和
足够的 request timeout 排查问题。

## 5. Self-play（默认接在 SFT 后）

默认配置是 KQAPro + GraphScript v0.3。Self-play 的验证集使用 RL row schema，因此即使跳过
Solver-only GRPO，也需要从冻结的 KQAPro val 导出一份只读 Solver RL val；这一步只做格式转换，
不会训练模型：

```bash
python -m graphtask_r1.cli data export-rl \
  --input data/processed/kqapro/kqapro-v1/val/training_tasks.parquet \
  --output data/training/kqapro_graphscript_v03_solver_rl_val.parquet \
  --roles solver --interaction-mode graphscript --graphscript-version 0.3 \
  --relation-catalog data/processed/kqapro/kqapro-v1/relation_catalog.json

export INITIAL_ADAPTER=$PWD/outputs/sft/qwen3-4b-kqapro-v03/checkpoint-last
export BASE_TASKS=$PWD/data/processed/kqapro/kqapro-v1/train/training_tasks.parquet
export VAL_DATA=$PWD/data/training/kqapro_graphscript_v03_solver_rl_val.parquet
export QUESTIONER_SEEDS=$PWD/data/training/kqapro_questioner_seeds.parquet
export KQAPRO_RELATION_CATALOG=$PWD/data/processed/kqapro/kqapro-v1/relation_catalog.json

python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay --dry-run

python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay
```

若有意运行了第 4 节，可选地把 `INITIAL_ADAPTER` 改为 Solver-only GRPO checkpoint；self-play
其余配置不变。

真实运行每轮会先用 ms-swift 合并基础模型与当前 LoRA，然后让 SGLang 直接加载合并后的冻结
opponent；SGLang 不使用动态 LoRA 参数。随后组装 questioner/solver mixed Parquet、调用 ms-swift
GRPO、查找新 LoRA adapter，并写入 round manifest。合并模型保存在
`round_NNN/opponent_merged/`，日志保存在 `round_NNN/logs/merge.log`。配置 hash、dataset hash、
adapter 路径和 ms-swift 版本用于恢复；修改配置后不能从旧 manifest 继续。外部图调用保留 timeout、
retry、cache 和 trace ID。

合并命令使用 `--load_args false` 并显式传入模型类型、LoRA 类型和精度，不恢复 checkpoint
`args.json` 中可能已经失效的 `external_plugins` 绝对路径。因此移动项目目录或复制 checkpoint 后
无需修改 `args.json`，也不需要为旧项目目录建立软链接。

默认 4×H100 布局为 actor GPU `0,1,2`、frozen opponent GPU `3`；actor rollout 使用 colocate，
不另占 GPU。每轮确定性抽取 192 条 Questioner rows 和 320 条 Solver rows，使用
`rollout_n=4`、`opponent_samples=4`、4096 completion 上限。三轮理论上限为 6144 条 actor
completions 和 9216 条 opponent completions。详细的一天预算、首轮外推和降级顺序见
[KQAPro 训练流程：Questioner/Solver self-play](KQAPRO_TRAINING.md#5-questionersolver-self-play)。

4B/80GB 默认采用 `micro_batch_size=4`、`eval_batch_size=4`、
`gradient_accumulation_steps=2`、`vllm_gpu_memory_utilization=0.6`、
`vllm_max_model_len=32768` 和 `vllm_sleep_level=1`。三张 actor GPU 的训练有效 batch 为 24，
采样 batch 为 48。训练期 val 固定抽样 256 条且每题只生成一次；REINFORCE++ 为默认 advantage
estimator，也可切回 GRPO。首轮显存监控以及生成/反向传播 OOM 的分别退档方式见上面的 KQAPro
小节。

### 单卡 Qwen3-0.6B smoke

`configs/training/selfplay_qwen3_0_6b_smoke.yaml` 用于 ToyGraph 上的两轮最小链路检查。它显式使用
`use_vllm=false` 和进程内 Transformers frozen opponent，因此不需要安装 vLLM/SGLang。单卡重叠
只有在 `allow_gpu_overlap=true`、`opponent_backend=transformers`、`use_vllm=false` 同时成立时才会
通过配置验证；该 opponent 是确定性生成，所以 `opponent_samples` 必须为 1。此配置只验证 LoRA
合并、mixed-role reward、checkpoint 传递和恢复，不替代 KQAPro v0.3 的正式训练或模型质量评测。

运行前把 `SMOKE_MODEL_PATH` 指向本地 `Qwen/Qwen3-0.6B`，并设置配置中其余数据与 adapter 环境
变量。GRPO 启动脚本会预检 `math_verify`；缺失时会在加载模型前快速失败。

## 6. 合并 SFT/GRPO LoRA 权重

训练脚本使用 LoRA，因此 `checkpoint-last` 默认只包含增量 adapter；它不是可以脱离基础模型单独
加载的完整权重。需要生成可独立部署、复制或归档的 Hugging Face 模型目录时，使用当前固定版本
`ms-swift==3.10.3` 的 `swift export --merge_lora true`。v3.x 使用 `--adapters`，不要使用已移除的
v2.x `--ckpt_dir` 参数。

### 6.1 合并 SFT checkpoint

```bash
export BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
export SFT_ADAPTER=$PWD/outputs/sft/qwen3-4b-kqapro-v03/checkpoint-last
export SFT_MERGED=$PWD/outputs/merged/qwen3-4b-kqapro-sft

test -f "$SFT_ADAPTER/adapter_config.json"
test -f "$SFT_ADAPTER/adapter_model.safetensors"
test ! -e "$SFT_MERGED"
mkdir -p "$(dirname "$SFT_MERGED")"

CUDA_VISIBLE_DEVICES=0 swift export \
  --model "$BASE_MODEL" \
  --adapters "$SFT_ADAPTER" \
  --merge_lora true \
  --output_dir "$SFT_MERGED"
```

### 6.2 合并 GRPO checkpoint

GRPO 必须合并最终 GRPO adapter，不能误用它的 SFT 初始化 adapter：

```bash
export BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
export GRPO_ADAPTER=$PWD/outputs/grpo/qwen3-4b-kqapro-v03/checkpoint-last
export GRPO_MERGED=$PWD/outputs/merged/qwen3-4b-kqapro-grpo

test -f "$GRPO_ADAPTER/adapter_config.json"
test -f "$GRPO_ADAPTER/adapter_model.safetensors"
test ! -e "$GRPO_MERGED"
mkdir -p "$(dirname "$GRPO_MERGED")"

CUDA_VISIBLE_DEVICES=0 swift export \
  --model "$BASE_MODEL" \
  --adapters "$GRPO_ADAPTER" \
  --merge_lora true \
  --output_dir "$GRPO_MERGED"
```

`--output_dir` 应指向一个尚不存在的新目录，避免把完整权重混入 adapter checkpoint。合并不会修改
原 adapter，但需要同时读取基础模型和 adapter；应预留至少一份完整模型的磁盘空间，并保留原始
adapter、训练 `args.json` 和 checkpoint manifest，以便追溯。

### 6.3 检查合并产物

```bash
test -f "$SFT_MERGED/config.json"
test -f "$SFT_MERGED/tokenizer_config.json"
find "$SFT_MERGED" -maxdepth 1 \
  \( -name 'model*.safetensors' -o -name 'model.safetensors.index.json' \) -print

test -f "$GRPO_MERGED/config.json"
test -f "$GRPO_MERGED/tokenizer_config.json"
find "$GRPO_MERGED" -maxdepth 1 \
  \( -name 'model*.safetensors' -o -name 'model.safetensors.index.json' \) -print
```

不要仅根据导出命令退出码判断合并正确。正式评测前，对同一个 checkpoint 做一次 adapter 与 merged
的固定 prompt、`temperature=0` 对照；至少确认输出格式和 GraphScript 行为一致。然后按
[评测与可视化 README](KQAPRO_EVAL_VIS_README.md#7-合并权重后的等价性检查)在同一组 KQAPro
indices 上跑 bounded evaluation。合并后的目录部署时作为完整模型传给 `--model-path`，不再传
`--enable-lora` 或 `--lora-paths`。

合并命令依据 ms-swift 3.10.3 的
[命令行参数说明](https://swift.readthedocs.io/en/v3.6/Instruction/Command-line-parameters.html)和
[v3 迁移说明](https://swift.readthedocs.io/en/v3.6/Instruction/ReleaseNote3.0.html)。当前项目是普通
Transformers Qwen3-4B LoRA；若以后切换到 Megatron/MCore、MoE 或混合全参训练，不应直接套用本节，
需使用对应训练后端的专用 export 流程。

## 7. KQAPro val 验证

本阶段只用 `kqapro_graphscript_v03_solver_rl_val.parquet` 的冻结 val 行评测并选择 checkpoint；它由
官方 `val.json` 的认证程序产生，不从隐藏 test 构造标签。KILT/HotpotQA 的检索型评测等独立路线
完成设计后再启用，不能替代当前 KQAPro val checkpoint 选择。

除 EM/F1 外必须报告 program parse rate、execution rate、operator count、passage search count 和
latency，避免把“代码格式失败”“执行失败”和“程序语义错误”混为一类。

## 8. 入口索引

| 用途 | 文件 |
| --- | --- |
| SFT | `scripts/train_ms_swift_sft.sh` |
| Self-play 内部 GRPO / 可选 Solver-only GRPO | `scripts/train_ms_swift_grpo.sh` |
| rollout server | `scripts/rollout_ms_swift.sh` |
| 合并 LoRA | `swift export --adapters ... --merge_lora true` |
| SFT 模板预检 | `scripts/preflight_ms_swift_sft.py` |
| 数据加载与字段转换 | `graphtask_r1/training/ms_swift_data.py` |
| dataset/reward/scheduler 注册 | `graphtask_r1/training/ms_swift_plugin.py` |
| mixed-role round orchestration | `graphtask_r1/training/selfplay.py` |
