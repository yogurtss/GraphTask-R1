# KQAPro 模型评测与路径可视化 README

本文说明如何依次部署并评测以下三个模型阶段：

1. `base`：原始基础模型，只直接回答问题；
2. `sft`：SFT checkpoint，优先生成并执行 GraphScript，失败时回退为直接回答；
3. `grpo`：GRPO checkpoint，优先生成并执行 GraphScript，失败时同样回退为直接回答。

每条评测或可视化命令只连接一个模型服务。三个阶段应分别运行、分别保存输出，最后再离线汇总
三份 `metrics.json`。评测与可视化不需要部署 Web 前后端；可视化产物是一个可直接用浏览器打开的
静态 HTML 文件。

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
 三次独立运行的精度对比
```

模型阶段决定推理协议：

| `--model-stage` | 首次推理 | 工具路径失败后 | 是否执行 KQAPro 图 |
| --- | --- | --- | --- |
| `base` | 直接回答 prompt | 不适用 | 否 |
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
concurrency: 8
request_timeout_s: 180
request_retries: 2
max_follow_limit: 100
max_edge_visits: 200
max_returned_entities: 1000
```

配置中只放当前正在评测的一个模型。`KQAPRO_MODEL` 必须等于服务的 `/v1/models` 返回的模型
ID。切换 base、SFT、GRPO 时，修改当前服务和这一个环境变量即可。

首次测试建议临时把 `concurrency` 改为 `1` 或 `2`，确认显存和输出格式后再提高。随机 seed、
GraphScript limit 和图执行预算均显式记录在 `metrics.json` 中。

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

base 评测不会读取 relation catalog，也不会执行 GraphScript。`--model-stage base` 不能省略，
因为它控制直接回答协议，而不只是给结果加标签。

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

## 7. 部署后的最小 API 检查

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

## 8. 浏览 KQAPro val 数据

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

## 9. 小规模评测

每次换模型后，先跑 10–20 条 bounded smoke：

### 9.1 原模型

```bash
python -m graphtask_r1.cli evaluate kqapro-val \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage base \
  --limit 20 \
  --output-dir outputs/evaluation/kqapro-base-smoke
```

### 9.2 SFT 模型

```bash
python -m graphtask_r1.cli evaluate kqapro-val \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage sft \
  --limit 20 \
  --output-dir outputs/evaluation/kqapro-sft-smoke
```

### 9.3 GRPO 模型

```bash
python -m graphtask_r1.cli evaluate kqapro-val \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage grpo \
  --limit 20 \
  --output-dir outputs/evaluation/kqapro-grpo-smoke
```

每次只运行与当前已部署 checkpoint 对应的命令。不要在部署 SFT adapter 时传
`--model-stage grpo`；stage 决定 prompt 和回退协议，不能代替正确的 checkpoint。

## 10. 完整 val 评测

smoke 正常后去掉 `--limit`。以下三条命令不是同时运行，而是在依次部署相应模型后分别执行。

### 10.1 Base

```bash
python -m graphtask_r1.cli evaluate kqapro-val \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage base \
  --output-dir outputs/evaluation/kqapro-base
```

### 10.2 SFT

```bash
python -m graphtask_r1.cli evaluate kqapro-val \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage sft \
  --output-dir outputs/evaluation/kqapro-sft
```

### 10.3 GRPO

```bash
python -m graphtask_r1.cli evaluate kqapro-val \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage grpo \
  --output-dir outputs/evaluation/kqapro-grpo
```

如果省略 `--output-dir`，CLI 会自动使用：

```text
outputs/evaluation/kqapro-base/
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
- `overall.tool_success_rate`：SFT/GRPO 首次 GraphScript 成功率；
- `overall.fallback_rate`：进入直接回答回退的比例；
- `overall.fallback_exact_match`：回退样本的准确率；
- `overall.primary_failure_rate`：GraphScript 或首次模型请求的失败比例；
- `overall.terminal_failure_rate`：包括回退后仍没有答案的最终失败比例；
- `by_operator`：按 certified gold program operator family 分桶的结果。

`predictions.parquet` 还保留结构化 `rejection_reason`、原始模型响应、GraphScript steps、执行
support triples、图预算和 trace 相关信息。SFT/GRPO 回退后仍会保留主路径的失败原因；如果主程序
已经解析成功，其尝试过的 operator path 也会保留。

## 11. 汇总 base、SFT、GRPO 精度

三个阶段均完成后，离线读取三份指标：

```bash
python -m graphtask_r1.cli evaluate kqapro-compare \
  --metrics outputs/evaluation/kqapro-base/metrics.json \
            outputs/evaluation/kqapro-sft/metrics.json \
            outputs/evaluation/kqapro-grpo/metrics.json \
  --output outputs/evaluation/kqapro-comparison.json
```

该命令不会连接模型服务。它会先验证三份结果的 dataset、split、graph snapshot、输入文件和样本数
一致，再打印并保存：

- 三个阶段各自的 EM、F1、precision、recall；
- SFT/GRPO 的工具成功率和回退率；
- SFT、GRPO 相对 base 的 EM/F1 增量。

如果比较命令报告输入或样本数不一致，应重新用同一个 `input_path`、相同的 `--limit`（或均不传）
运行，不应直接比较不等价的结果。

## 12. 生成路径可视化

可视化和完整评测完全分开。它只测试显式选择的少量样本，并生成单模型 HTML。

### 12.1 Base 可视化

部署 base 后运行：

```bash
python -m graphtask_r1.cli visualize kqapro \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage base \
  --indices 0,12,41 \
  --output-dir outputs/visualization/kqapro-base
```

base 的 HTML 会显示直接回答及正确性，不会伪造图路径。

### 12.2 SFT 可视化

部署 SFT 后运行：

```bash
python -m graphtask_r1.cli visualize kqapro \
  --config configs/evaluation/kqapro_val.yaml \
  --model-stage sft \
  --indices 0,12,41 \
  --output-dir outputs/visualization/kqapro-sft
```

### 12.3 GRPO 可视化

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
- `resolve_entity`、`follow`、filter、query、count、emit 等 operator 顺序；
- 每一步参数和 handle；
- 最多 20 条执行 support triples；
- GraphScript 失败与直接回答回退状态。

不传 `--indices` 时默认只测试前三条；也可以使用 `--limit 5`。为保证案例可比，推荐三个阶段都
使用相同的 `--indices`。

## 13. 推荐的完整操作顺序

```text
1. 检查 graph.sqlite、val/tasks.parquet、relation_catalog.json
2. 部署 base
3. base --limit 20 smoke
4. base 完整评测 + base 小样本可视化
5. 停止 base，部署 SFT
6. SFT --limit 20 smoke
7. SFT 完整评测 + 相同 indices 可视化
8. 停止 SFT，部署 GRPO
9. GRPO --limit 20 smoke
10. GRPO 完整评测 + 相同 indices 可视化
11. kqapro-compare 汇总三份 metrics.json
```

## 14. 可复现性与结果管理

- 三个阶段必须使用同一个 `val/tasks.parquet`、graph snapshot 和 relation catalog；
- 正式对比时三个阶段应使用相同样本数，不要只给某个模型传 `--limit`；
- 不要删除或手工修改 `metrics.json`、`predictions.parquet` 和响应 cache；
- 不要用 KQAPro val 生成训练样本、计算训练 reward 或回填 self-play archive；
- 记录基础模型版本、adapter 路径、checkpoint step、GPU、SGLang 版本和配置文件；
- 如果更换 prompt、预算或 relation catalog，应重新运行全部三个阶段；
- 首次大规模运行前，始终先使用 `--limit` 和低 concurrency。

## 15. 开发检查

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
| 三阶段指标汇总 | `python -m graphtask_r1.cli evaluate kqapro-compare` |
| 数据浏览/路径可视化 | `python -m graphtask_r1.cli visualize kqapro` |
| 配置模板 | `configs/evaluation/kqapro_val.yaml` |
| 核心实现 | `graphtask_r1/evaluation/kqapro_val.py` |
