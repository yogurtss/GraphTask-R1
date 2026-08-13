# KQAPro 模型评测与路径可视化 README

本文说明如何独立运行以下四种评测模式：

1. `base`：原始基础模型，严格直接回答问题；
2. `base_tool`：同一个原始基础模型，用带函数说明与 few-shot 示例的 prompt 生成 GraphScript；
3. `sft`：SFT checkpoint，优先生成并执行 GraphScript，失败时回退为直接回答；
4. `grpo`：GRPO checkpoint，优先生成并执行 GraphScript，失败时同样回退为直接回答。

每条评测或可视化命令只连接一个模型服务。每种模式分别运行、分别保存输出；离线汇总时传入任意
两个或更多 `metrics.json`，不要求四种模式全部完成。例如 GRPO 尚未训练时，可以只比较
`base`、`base_tool` 和 `sft`。评测与可视化不需要部署 Web 前后端；可视化产物是静态 HTML。

## 1. 运行链路

```text
一个 OpenAI-compatible 模型服务
              │
              ▼
  evaluate kqapro-val                 visualize kqapro
  全量或 bounded 精度评测             浏览/测试少量指定样本
              │                              │
              ▼                              ▼
 metrics.json + predictions.parquet   CLI JSON + paths.html
              │
              ▼
 evaluate kqapro-compare
 任意多次兼容运行的精度对比
```

模型阶段决定推理协议：

| `--model-stage` | 首次推理 | 工具路径失败后 | 是否执行 KQAPro 图 |
| --- | --- | --- | --- |
| `base` | 直接回答 prompt | 不适用 | 否 |
| `base_tool` | 带函数说明/few-shot 的 GraphScript v0.3 | 不回退，保留工具失败 | 是 |
| `sft` | GraphScript v0.3 | 同一模型直接回答 | 是 |
| `grpo` | GraphScript v0.3 | 同一模型直接回答 | 是 |

这里的“工具调用”是模型生成 GraphScript、GraphTask 在本地有界执行该程序并查询 KQAPro SQLite
图。模型服务本身只需提供 OpenAI-compatible `/v1/chat/completions`，不需要加载 GraphTask 工具。

## 2. 前置条件

### 2.1 Python 与 GPU 环境

在项目根目录运行：

```bash
conda activate graphtask-swift-cu124
export PYTHONPATH=$PWD

python -m pip install -r requirements.txt
python -c "import aiohttp, pyarrow, pydantic, yaml; print('evaluation dependencies OK')"
python -m sglang.launch_server --help >/dev/null
```

如果最后一条失败，需要在独立的模型服务环境安装与当前 CUDA/PyTorch 兼容的 SGLang。项目的
self-play 也使用 `python -m sglang.launch_server`。环境建议见
[MS_SWIFT_CUDA_12_4.md](MS_SWIFT_CUDA_12_4.md)，SGLang 安装和 OpenAI-compatible 服务参数以
[SGLang 官方文档](https://docs.sglang.ai/)为准。

### 2.2 KQAPro 数据与图

评测需要以下三个文件：

```text
data/processed/kqapro/kqapro-v1/
├── graph.sqlite
├── relation_catalog.json
└── val/
    └── tasks.parquet
```

设置路径：

```bash
export KQAPRO_DIR=$PWD/data/processed/kqapro/kqapro-v1
export GRAPHTASK_KQAPRO_DB=$KQAPRO_DIR/graph.sqlite

test -f "$GRAPHTASK_KQAPRO_DB"
test -f "$KQAPRO_DIR/relation_catalog.json"
test -f "$KQAPRO_DIR/val/tasks.parquet"
```

如果尚未生成这些文件，请先完成
[KQAPRO_TRAINING.md](KQAPRO_TRAINING.md) 中的数据准备、审计和 relation catalog 构建步骤。
gold answer 来自 certified program 的实际执行结果，不从模型输出或原始 answer 字段重新推断。

## 3. 评测配置

默认模板是 [configs/evaluation/kqapro_val.yaml](../configs/evaluation/kqapro_val.yaml)：

```yaml
input_path: ${KQAPRO_DIR}/val/tasks.parquet
relation_catalog: ${KQAPRO_DIR}/relation_catalog.json
graph_snapshot: kqapro-v1
graphscript_version: "0.3"

model:
  model_url: ${KQAPRO_MODEL_URL}
  model: ${KQAPRO_MODEL}
  max_completion_tokens: 4096

seed: 42
concurrency: 2
request_timeout_s: 600
request_retries: 1
max_follow_limit: 100
max_edge_visits: 200
max_returned_entities: 1000
```

配置中只放当前正在评测的一个模型。`KQAPRO_MODEL` 必须等于服务的 `/v1/models` 返回的模型
ID。切换 base checkpoint 与 SFT/GRPO checkpoint 时，修改当前服务和这一个环境变量即可；
`base` 与 `base_tool` 使用同一个原模型服务，只是评测 prompt/protocol 不同。

这些默认值针对一个 4B 模型部署在单 GPU 上。首次测试仍建议用 `concurrency: 1`，确认显存、单条
延迟和输出格式后再调到 2；只有服务端确有更多并行容量时才继续提高。SGLang 的队列等待也计入
`request_timeout_s`。随机 seed、GraphScript limit 和图执行预算均显式记录在 `metrics.json` 中。

8B 单模型评测/可视化可复制
[configs/evaluation/kqapro_val_qwen3_8b.yaml](../configs/evaluation/kqapro_val_qwen3_8b.yaml)。它只把
客户端初始值改得更保守：`concurrency: 1`、`request_timeout_s: 900`，并保持输出上限 4096。
配置本身不会下载或启动模型；`KQAPRO_MODEL_URL` 仍指向用户已经部署的一个 OpenAI-compatible
服务，`KQAPRO_MODEL` 必须使用该服务 `/v1/models` 返回的 ID。base、base_tool、SFT 和 GRPO
继续分别调用，一次只传一个 `--model-stage`。

如果服务加载官方 `Qwen/Qwen3-8B`，第 4–6 节的部署方式不变，只需将 `BASE_MODEL` 或 adapter 的
基础模型路径换成 8B，并按显存调整 `--tp-size`。仓库不会根据 YAML 自动部署、下载或合并 8B
权重；首次连接仍应先跑 bounded smoke，再决定是否提高 concurrency 或 token 上限。

### 3.1 当前 SGLang 部署不是 PD 分离

本文的启动命令只运行一个 `python -m sglang.launch_server`。prefill 和 decode 位于同一服务进程
及同一组 `--tp-size` GPU 上，因此是普通统一部署，不是 prefill/decode（PD）分离。`--tp-size 2`
表示 tensor parallel，也不等于 PD 分离。

真正的 SGLang PD 分离需要至少两组独立 GPU/实例，分别以 disaggregation prefill/decode 模式
启动，并在前面配置负责把请求在两侧路由的 router。它适合多 GPU、高并发、prefill 与 decode
负载明显不平衡的服务；对于单 GPU 上的 4B 离线 eval，PD 无法成立，也通常不是解决 timeout 的
第一选择。应先降低客户端 concurrency、观察实际每条生成长度、提高合理的总超时，并检查是否有
请求在服务端排队。

如果确实要采用 PD 分离，应按照所安装 SGLang 版本的
[PD Disaggregation 官方文档](https://docs.sglang.ai/backend/pd_disaggregation.html)部署。该功能的
CLI 参数和 router 配置可能随 SGLang 版本变化，不要把本文的普通 `launch_server` 命令直接解释为
PD 配置。无论服务端是否 PD 分离，GraphTask 客户端仍只配置 router 暴露的一个
`KQAPRO_MODEL_URL`。

### 3.2 Token 上限与 timeout 是两件事

默认 `max_completion_tokens: 4096` 是模型**输出**上限，不是总上下文长度。对只生成 JSON
GraphScript 的 SFT/GRPO 来说通常已经充足；盲目增到 8192 或更大，会允许异常样本持续 decode
更久，反而更容易触发 timeout。建议先从 `predictions.parquet` 的 `completion_tokens` 分布判断：

- 大多数成功样本远低于 4096：保持 4096；
- 大量输出恰好停在 4096 且 JSON 尾部被截断：再提高到 6144/8192，并同步核对延迟和显存；
- prompt 本身过长或服务报 context length：应提高 SGLang 的总 context length，或缩短 relation
  catalog；只改 `max_completion_tokens` 不能扩大总上下文。

模型服务的总上下文必须至少容纳“system/user prompt（含 relation catalog）+ completion 上限”。具体
的 context-length 参数应以当前安装的 SGLang 版本为准。完整 eval 前建议先跑 20 条，记录实际
prompt/completion token 与 p95 延迟，再决定是否调整 token 上限。

## 4. 部署原模型（base）

原模型不加载 LoRA。以下命令使用一个 GPU 和端口 `18100`；如果模型需要多卡，可相应调整
`--tp-size` 和 `CUDA_VISIBLE_DEVICES`。

在终端 A 启动服务：

```bash
export BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
export CUDA_VISIBLE_DEVICES=0

python -m sglang.launch_server \
  --model-path "$BASE_MODEL" \
  --host 127.0.0.1 \
  --port 18100 \
  --tp-size 1
```

在终端 B 检查服务：

```bash
curl -f http://127.0.0.1:18100/health
curl -s http://127.0.0.1:18100/v1/models
```

根据 `/v1/models` 的实际返回值配置评测客户端。默认情况下通常是 `--model-path` 的值，但应以
接口返回的 `data[].id` 为准：

```bash
export KQAPRO_MODEL_URL=http://127.0.0.1:18100
export KQAPRO_MODEL=Qwen/Qwen3-4B-Instruct-2507
```

`base` 评测不会读取 relation catalog，也不会执行 GraphScript。它通过三个格式示例要求模型的整个
响应严格为 `<answer>["最终答案"]</answer>`；解析器使用 full-match，标签前后出现解释、推理或
其他文字都会记为 `DIRECT_INFERENCE_FAILED`，不会从长文本中抽取一个看似正确的答案。

`base_tool` 使用同一个 base 服务，但会读取 relation catalog，并给原模型提供全部 GraphScript
函数的签名/语义、handle 规则，以及 follow、query attribute、all entities + count、literal filter
四类示例。它执行模型生成的代码，但失败时不回退直接回答，以便单独测量“未训练原模型生成代码并
使用工具”的能力。`--model-stage` 控制的是评测协议，不只是结果标签。

## 5. 部署 SFT 模型

### 5.1 SFT 是 LoRA adapter

训练脚本默认输出 LoRA checkpoint。先确认 adapter 目录：

```bash
export BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
export SFT_ADAPTER=$PWD/outputs/sft/qwen3-4b-kqapro-v03/checkpoint-last

test -d "$SFT_ADAPTER"
find "$SFT_ADAPTER" -maxdepth 2 \
  \( -name adapter_config.json -o -name adapter_model.safetensors \) -print
```

停止上一节的 base 服务，然后在终端 A 启动基础模型和命名 adapter：

```bash
export CUDA_VISIBLE_DEVICES=0

python -m sglang.launch_server \
  --model-path "$BASE_MODEL" \
  --host 127.0.0.1 \
  --port 18100 \
  --tp-size 1 \
  --enable-lora \
  --lora-paths "kqapro-sft=$SFT_ADAPTER"
```

在终端 B 验证并选择 adapter 名：

```bash
curl -f http://127.0.0.1:18100/health
curl -s http://127.0.0.1:18100/v1/models

export KQAPRO_MODEL_URL=http://127.0.0.1:18100
export KQAPRO_MODEL=kqapro-sft
```

`KQAPRO_MODEL=kqapro-sft` 对应 `--lora-paths` 等号左侧的名字，不是 adapter 的文件系统路径。

### 5.2 SFT 已合并为完整模型

如果已经把 adapter 合并为完整 Hugging Face 模型目录，则与 base 一样直接加载：

```bash
export SFT_MERGED=/path/to/qwen3-4b-kqapro-sft-merged
export CUDA_VISIBLE_DEVICES=0

python -m sglang.launch_server \
  --model-path "$SFT_MERGED" \
  --host 127.0.0.1 \
  --port 18100 \
  --tp-size 1
```

然后从 `/v1/models` 获取 ID 并赋给 `KQAPRO_MODEL`。完整模型不要再传 `--enable-lora`。

## 6. 部署 GRPO 模型

GRPO checkpoint 的部署方式与 SFT 相同，但必须指向 GRPO 输出，而不是初始化 GRPO 时使用的
SFT adapter。

### 6.1 GRPO 是 LoRA adapter

```bash
export BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
export GRPO_ADAPTER=$PWD/outputs/grpo/qwen3-4b-kqapro-v03/checkpoint-last

test -d "$GRPO_ADAPTER"
find "$GRPO_ADAPTER" -maxdepth 2 \
  \( -name adapter_config.json -o -name adapter_model.safetensors \) -print
```

停止 SFT 服务，然后启动 GRPO adapter：

```bash
export CUDA_VISIBLE_DEVICES=0

python -m sglang.launch_server \
  --model-path "$BASE_MODEL" \
  --host 127.0.0.1 \
  --port 18100 \
  --tp-size 1 \
  --enable-lora \
  --lora-paths "kqapro-grpo=$GRPO_ADAPTER"
```

配置当前模型：

```bash
curl -f http://127.0.0.1:18100/health
curl -s http://127.0.0.1:18100/v1/models

export KQAPRO_MODEL_URL=http://127.0.0.1:18100
export KQAPRO_MODEL=kqapro-grpo
```

如果 GRPO 是合并后的完整模型，也应像 base 一样直接用 `--model-path` 加载，并把
`KQAPRO_MODEL` 设置为 `/v1/models` 返回的 ID。

## 7. 合并权重后的等价性检查

SFT/GRPO 默认是 LoRA adapter。如果希望用单一模型目录部署，可先按
[训练手册的合并权重章节](TRAINING.md#6-合并-sftgrpo-lora-权重)分别生成：

```text
outputs/merged/qwen3-4b-kqapro-sft/
outputs/merged/qwen3-4b-kqapro-grpo/
```

合并后的模型应像 base 一样部署，不要再叠加原 adapter：

```bash
export MERGED_MODEL=$PWD/outputs/merged/qwen3-4b-kqapro-sft
export CUDA_VISIBLE_DEVICES=0

python -m sglang.launch_server \
  --model-path "$MERGED_MODEL" \
  --host 127.0.0.1 \
  --port 18100 \
  --tp-size 1
```

从 `/v1/models` 读取实际模型 ID，然后对 adapter 部署和 merged 部署分别使用相同阶段、seed、输入
indices 和执行预算做小规模对照。例如检查 SFT：

```bash
# 第一次：部署 kqapro-sft adapter 后运行。
export KQAPRO_MODEL_URL=http://127.0.0.1:18100
export KQAPRO_MODEL=kqapro-sft
python -m graphtask_r1.cli visualize kqapro \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage sft --indices 0,12,41 \
  --output-dir outputs/merge-check/sft-adapter

# 第二次：停止 adapter 服务，部署 SFT_MERGED；把 KQAPRO_MODEL 改成
# /v1/models 返回的 merged model ID 后运行。
python -m graphtask_r1.cli visualize kqapro \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage sft --indices 0,12,41 \
  --output-dir outputs/merge-check/sft-merged
```

GRPO 使用同样流程，但两次命令都传 `--model-stage grpo`。重点比较两侧的：

- `predictions.parquet` 中 `raw_response`、`predicted_answers` 和 GraphScript path；
- `metrics.json` 中 EM/F1、tool success 和 fallback；
- 模型服务使用的基础模型 revision、tokenizer、chat template 和 dtype。

浮点计算和推理引擎可能造成细微 logits 差异，因此不要求输出文件逐字节相同；但相同 greedy
配置下若大量样本的 GraphScript、最终答案或工具成功率发生系统性变化，应停止完整评测，检查是否
合并了错误 adapter、基础模型 revision 是否一致，以及 merged 服务是否又重复加载了 LoRA。

## 8. 部署后的最小 API 检查

在运行正式评测前，先确认 OpenAI-compatible chat endpoint 可用。把 `model` 保持为当前阶段的
`KQAPRO_MODEL`：

```bash
curl -f http://127.0.0.1:18100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$KQAPRO_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK\"}],\"max_tokens\":8,\"temperature\":0}"
```

常见错误：

- HTTP 404：`KQAPRO_MODEL_URL` 不应包含 `/v1`，只填 `http://host:port`；
- model not found：`KQAPRO_MODEL` 与 `/v1/models` 返回值不一致；
- adapter load error：基础模型与 LoRA adapter 不匹配，或 adapter 路径不是实际 checkpoint；
- CUDA OOM：降低配置中的 `concurrency`，降低服务显存占用，或增加 tensor parallel GPU；
- context length error：确认模型服务上下文窗口能容纳 relation catalog 和最多 4096 completion
  tokens。
- `TimeoutError`：先把 `concurrency` 降到 1，查看 SGLang 日志中的排队和生成速度；确认仍是单请求
  本身超过 600 秒后，再增加 `request_timeout_s`。不要先增加 concurrency；客户端 timeout 后的
  retry 可能使尚未结束的服务端请求与重试请求同时占用队列。

### 8.1 Eval 进度输出

`evaluate kqapro-val` 默认每 5 秒向 stderr 写一条结构化进度日志，不需要安装 `tqdm`。开始、等待
和完成时会看到类似：

```text
operation="evaluate.kqapro_val" phase="progress" completed=84 total=11768 percent=0.7 \
elapsed_s=312.4 pending=11684 correct=51 tool_successes=73 fallbacks=11 \
terminal_failures=0 cache_hits=0 bar="[░░░░░░░░░░░░░░░░░░░░]"
```

字段含义：

- `completed/total/percent`：已经完成的样本和总进度；
- `pending`：尚未完成（包含正在请求和等待 semaphore）的样本；
- `correct`：当前最终回答正确数；
- `tool_successes`：GraphScript 首次成功执行数；
- `fallbacks`：SFT/GRPO 已进入直接回答回退的样本数；
- `terminal_failures`：主路径和回退都没有产生答案的样本数；
- `cache_hits`：从当前输出目录响应 cache 复用的样本数。
- `bar`：20 格终端文本进度条；精确进度仍以 `completed/total/percent` 为准。

即使前一批请求仍在等待，heartbeat 也会每 5 秒打印，因此可以区分“进程卡死”和“模型仍在生成”。
日志使用 CLI 默认的 `--log-level INFO`；如果显式设成 `WARNING` 或 `ERROR`，进度不会显示。需要
保存日志时可运行：

```bash
python -m graphtask_r1.cli --log-level INFO evaluate kqapro-val \
  --config configs/evaluation/kqapro_val.yaml --model-stage grpo --limit 20 \
  2>&1 | tee outputs/evaluation/kqapro-grpo-smoke.log
```

## 9. 浏览 KQAPro val 数据

可先只查看数据，不调用模型：

```bash
python -m graphtask_r1.cli visualize kqapro \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage base \
  --indices 0,12,41 \
  --inspect-only
```

CLI 会打印：

- `task_id` 和 source ID；
- question；
- certified gold answers；
- operator tags。

`--indices` 是 `val/tasks.parquet` 中从 0 开始的行号。省略时默认读取前三条，可用 `--limit`
修改数量：

```bash
python -m graphtask_r1.cli visualize kqapro \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage sft \
  --limit 10 \
  --inspect-only
```

`--inspect-only` 不请求模型；这里的 `--model-stage` 只是保持 CLI 调用形式一致。

## 10. 小规模评测

每次换模型后，先跑 10–20 条 bounded smoke：

### 10.1 原模型直接回答

```bash
python -m graphtask_r1.cli evaluate kqapro-val \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage base \
  --limit 20 \
  --output-dir outputs/evaluation/kqapro-base-smoke
```

### 10.2 原模型生成 GraphScript

不需要重启 base 服务，只切换评测模式和输出目录：

```bash
python -m graphtask_r1.cli evaluate kqapro-val \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage base_tool \
  --limit 20 \
  --output-dir outputs/evaluation/kqapro-base-tool-smoke
```

### 10.3 SFT 模型

```bash
python -m graphtask_r1.cli evaluate kqapro-val \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage sft \
  --limit 20 \
  --output-dir outputs/evaluation/kqapro-sft-smoke
```

### 10.4 GRPO 模型（训练完成后可选）

```bash
python -m graphtask_r1.cli evaluate kqapro-val \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage grpo \
  --limit 20 \
  --output-dir outputs/evaluation/kqapro-grpo-smoke
```

每次只运行与当前已部署 checkpoint 对应的命令。不要在部署 SFT adapter 时传
`--model-stage grpo`；stage 决定 prompt 和回退协议，不能代替正确的 checkpoint。

## 11. 完整 val 评测

smoke 正常后去掉 `--limit`。每条命令独立运行；`base` 与 `base_tool` 连接同一个原模型服务。

### 11.1 Base

```bash
python -m graphtask_r1.cli evaluate kqapro-val \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage base \
  --output-dir outputs/evaluation/kqapro-base
```

### 11.2 Base + tool

```bash
python -m graphtask_r1.cli evaluate kqapro-val \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage base_tool \
  --output-dir outputs/evaluation/kqapro-base-tool
```

### 11.3 SFT

```bash
python -m graphtask_r1.cli evaluate kqapro-val \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage sft \
  --output-dir outputs/evaluation/kqapro-sft
```

### 11.4 GRPO（可选）

```bash
python -m graphtask_r1.cli evaluate kqapro-val \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage grpo \
  --output-dir outputs/evaluation/kqapro-grpo
```

如果省略 `--output-dir`，CLI 会自动使用：

```text
outputs/evaluation/kqapro-base/
outputs/evaluation/kqapro-base-tool/
outputs/evaluation/kqapro-sft/
outputs/evaluation/kqapro-grpo/
```

每个目录包含：

```text
metrics.json             # 当前阶段的聚合指标和 operator 分桶
predictions.parquet      # 每条样本的回答、路径、失败和回退信息
cache/<stage>.json       # 可重放模型响应缓存
```

重点查看 `metrics.json` 中的：

- `overall.exact_match`、`overall.f1`：最终答案精度；
- `overall.tool_success_rate`：`base_tool`/SFT/GRPO 首次 GraphScript 成功率；
- `overall.fallback_rate`：进入直接回答回退的比例；
- `overall.fallback_exact_match`：回退样本的准确率；
- `overall.primary_failure_rate`：GraphScript 或首次模型请求的失败比例；
- `overall.terminal_failure_rate`：包括回退后仍没有答案的最终失败比例；
- `by_operator`：按 certified gold program operator family 分桶的结果。

`predictions.parquet` 还保留结构化 `rejection_reason`、原始模型响应、GraphScript steps、执行
support triples、图预算和 trace 相关信息。SFT/GRPO 回退后仍会保留主路径的失败原因；如果主程序
已经解析成功，其尝试过的 operator path 也会保留。

## 12. 任意汇总两个或更多模式

汇总不连接模型，也不要求四种模式全部存在。GRPO 尚未训练时，例如只比较 base direct、base tool
和 SFT：

```bash
python -m graphtask_r1.cli evaluate kqapro-compare \
  --metrics outputs/evaluation/kqapro-base/metrics.json \
            outputs/evaluation/kqapro-base-tool/metrics.json \
            outputs/evaluation/kqapro-sft/metrics.json \
  --output outputs/evaluation/kqapro-comparison.json
```

也可以只比较两份：

```bash
python -m graphtask_r1.cli evaluate kqapro-compare \
  --metrics outputs/evaluation/kqapro-base-tool/metrics.json \
            outputs/evaluation/kqapro-sft/metrics.json \
  --baseline-stage base_tool
```

命令会验证所有结果的 dataset、split、graph snapshot、输入文件和样本数一致，并拒绝重复 stage。
默认传入 `base` 时以 base 为基线；没有 base 时以第一份 metrics 为基线。也可用
`--baseline-stage base|base_tool|sft|grpo` 显式指定已传入的模式。输出包括：

- 每个已传入模式的 EM、F1、precision、recall、工具成功率和回退率；
- `baseline_stage`；
- 其他模式相对基线的 `delta_vs_baseline` EM/F1；
- 基线是 `base` 时额外保留兼容字段 `delta_vs_base`。

如果比较命令报告输入或样本数不一致，应重新用同一个 `input_path`、相同的 `--limit`（或均不传）
运行，不应直接比较不等价的结果。

## 13. 生成路径可视化

可视化和完整评测完全分开。它只测试显式选择的少量样本，并生成单模型 HTML。

### 13.1 Base 可视化

部署 base 后运行：

```bash
python -m graphtask_r1.cli visualize kqapro \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage base \
  --indices 0,12,41 \
  --output-dir outputs/visualization/kqapro-base
```

base 的 HTML 会显示直接回答及正确性，不会伪造图路径。

### 13.2 Base tool 可视化

仍连接 base 服务：

```bash
python -m graphtask_r1.cli visualize kqapro \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage base_tool \
  --indices 0,12,41 \
  --output-dir outputs/visualization/kqapro-base-tool
```

### 13.3 SFT 可视化

部署 SFT 后运行：

```bash
python -m graphtask_r1.cli visualize kqapro \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage sft \
  --indices 0,12,41 \
  --output-dir outputs/visualization/kqapro-sft
```

### 13.4 GRPO 可视化（可选）

部署 GRPO 后运行：

```bash
python -m graphtask_r1.cli visualize kqapro \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage grpo \
  --indices 0,12,41 \
  --output-dir outputs/visualization/kqapro-grpo
```

每次命令会在 CLI 输出 JSON，包括：

- 数据集 question 与 gold answer；
- 当前模型 prediction 和 correct；
- `graphscript`、`direct` 或 `direct_fallback` 推理模式；
- 是否发生回退；
- GraphScript operator path；
- 每一步有界的输入、输出、获取、保留、过滤实体和新增证据；
- 结构化失败原因。

同时生成：

```text
outputs/visualization/kqapro-<stage>/
├── paths.html
├── metrics.json
├── predictions.parquet
└── cache/<stage>.json
```

直接用浏览器打开 `paths.html`，不需要启动 HTTP 服务：

```bash
# Linux 桌面环境可选；也可以在文件管理器中双击。
xdg-open outputs/visualization/kqapro-grpo/paths.html
```

HTML 展示当前模型的：

- question、gold answer、predicted answer 和正确性；
- 左侧 `resolve_entity`、`follow`、filter、query、count、emit 等 operator/handle 数据流；
- 中间随步骤累积的知识图子图；当前步骤新增边会高亮；
- 节点采用整条 trace 的固定选择与固定初始布局：一旦出现，后续步骤不会删除或换掉它；
- 输入、本步获取、选中/输出、过滤掉、数值/答案、延迟集合六种节点角色；
- 右侧先用自然语言说明“本步实际做了什么”，再展示参数、输入/输出规模、实体
  ID/label/type/alias、耗时和累计预算；
- 上一步/下一步、拖动节点、重置布局和节点/关系点击详情；
- GraphScript 失败与直接回答回退状态。

所有交互均由 HTML 内嵌的原生 JavaScript 和 SVG 完成，不依赖 CDN，也不需要部署前后端。

### 13.5 大实体集合如何展示

执行本身仍受 `max_returned_entities`、`max_edge_visits` 等评测预算约束；展示层另做更严格的
有界预览，避免某次 `follow`、`all_entities` 或 filter 产生几百个节点后把图挤满：

- 每个 handle、每步获取/选中/过滤列表最多写入 8 个实体，同时保留真实 `total_count` 和
  `truncated` 标记，例如 `8 / 327（截断）`；
- trace 在执行完成后反向分析后续步骤，优先保留最终被 filter、intersect、select 或 emit 使用的
  实体，而不是简单取集合前 8 个；
- 相应的关键 evidence edge 同样优先保留；
- HTML 为整条 trace 固定选择最多 18 个真实实体节点，并明确显示“固定显示 18 / N”；步骤切换
  只增加已经到达的节点，不会重新选择 18 个节点，因此先前结果不会消失；
- 完整预测、最终答案、评测分数和执行预算不因可视化截断而改变。截断只影响 CLI/HTML 展示。

因此，一个步骤即使选择了大量实体，也会显示少数代表实体以及后续真正操作到的实体，而不会把
全部候选都塞入 HTML。

### 13.6 Operator 的执行语义与展示

页面区分“操作结果”“新增 KG 证据”和“延迟查询”，三者不能用同一个空列表表示：

| Operator 类别 | 执行器实际行为 | 页面展示 |
| --- | --- | --- |
| `resolve_entity` / `start` | 物化实体 handle | 解析到的实体节点、真实数量与 label/ID |
| `all_entities` | 只建立有 `max_results` 上限的 `AllEntities` program，不立即枚举全部实体 | 黄色 `All entities ≤ N` 延迟集合节点，明确标注“不是空结果” |
| `follow` | 调用 bounded neighbors，物化目标实体并记录访问到的 triples | 输入实体、获取实体、关系边、方向、证据数 |
| filter | 对实体 handle 或延迟 program 执行后端过滤；延迟输入可在同一次查询中物化 | 保留节点、已知的过滤节点；延迟集合到结果使用虚线 dataflow edge |
| `intersect` / `union` | 对已经物化的多个实体集合求交或合并 | 多输入 handle、输出集合和被排除的代表节点 |
| query / `count` / `verify` | 产生 literal、relation、count 或布尔答案，不一定产生新实体 | 橙色答案节点及实际值；`0` 是有效计数，不再当作空结果 |
| `select_between` / `select_among` | 查询比较属性并物化被选实体 | 候选输入、选择属性/mode、最终选中节点 |
| `require_unique` / `emit` | 验证唯一性或复用输入 handle 输出答案，不执行新的图遍历 | 复用并保留已有节点，说明验证/输出效果；不会清空前序图 |

`new_evidence_total == 0` 只表示本步没有新增 support triple，不代表本步结果为空。例如 `emit` 复用
已有 handle，`count` 产生数值，`all_entities` 建立延迟 program，它们都可能没有新增 KG edge。

最终图包含两层连线，并在切到后续步骤时持续保留：

1. 紫色流程层：`producer operator --handle--> consumer operator`，以及
   `operator --output handle--> result`。即使 operator 没有 KG evidence，流程仍然连通；
2. 灰色/绿色证据层：实体之间的 relation，以及实体到属性值的 attribute edge；当前步骤新增的
   evidence 为绿色。比如 `filter_literal(age > 30)` 会同时在 operator 节点显示 `age > 30`，并在
   候选实体旁显示 `entity --age--> actual value`。

实体节点下方直接显示 ID 和最多两个 type；点击节点可看完整 type、alias，以及本条执行实际读取的
`observed_properties`。属性展示也遵循 trace 的有界规则，只展示本次程序涉及和读取到的相关属性，
不会为每个实体无界加载整个知识库属性表。

不传 `--indices` 时默认只测试前三条；也可以使用 `--limit 5`。为保证案例可比，推荐所有待比较
模式都使用相同的 `--indices`。

## 14. 推荐的完整操作顺序

```text
1. 检查 graph.sqlite、val/tasks.parquet、relation_catalog.json
2. 部署 base
3. base --limit 20 smoke
4. base_tool --limit 20 smoke（复用同一个 base 服务）
5. base、base_tool 分别完整评测 + 相同 indices 可视化
6. 停止 base，部署 SFT
7. SFT --limit 20 smoke
8. SFT 完整评测 + 相同 indices 可视化
9. 如果 GRPO 已训练：部署 GRPO，完成 smoke/完整评测/可视化
10. kqapro-compare 汇总任意两个或更多 metrics.json
```

## 15. 可复现性与结果管理

- 所有待比较模式必须使用同一个 `val/tasks.parquet` 和 graph snapshot；工具模式还应使用同一个
  relation catalog；
- 正式对比时所有模式应使用相同样本数，不要只给某个模式传 `--limit`；
- 不要删除或手工修改 `metrics.json`、`predictions.parquet` 和响应 cache；
- 不要用 KQAPro val 生成训练样本、计算训练 reward 或回填 self-play archive；
- 记录基础模型版本、adapter 路径、checkpoint step、GPU、SGLang 版本和配置文件；
- 如果更换 prompt、预算或 relation catalog，应重新运行所有待比较模式；
- 首次大规模运行前，始终先使用 `--limit` 和低 concurrency。

## 16. 开发检查

修改评测或可视化代码后，在发布前运行：

```bash
make lint
make typecheck
make test
```

当前主要入口：

| 用途 | 入口 |
| --- | --- |
| 单模型评测 | `python -m graphtask_r1.cli evaluate kqapro-val` |
| 任意多模式指标汇总 | `python -m graphtask_r1.cli evaluate kqapro-compare` |
| 数据浏览/路径可视化 | `python -m graphtask_r1.cli visualize kqapro` |
| 配置模板 | `configs/evaluation/kqapro_val.yaml` |
| 核心实现 | `graphtask_r1/evaluation/kqapro_val.py` |
