# Self-play reward / zero-gradient 服务器排查手册

本手册用于排查以下典型现象：

```text
loss = 0
reward_std > 0
grad_norm = 0
completions/clipped_ratio = 0
Questioner score = -0.35 或 Solver score = 0
```

所有命令均为只读检查。不要在正在训练的 run 上手工调用 Questioner reward 或 opponent
`/evaluate` 接口：该接口会写入 self-play archive，改变后续训练状态。

## 1. 指定本次 run

在项目根目录执行，并将路径替换为服务器上的实际路径：

```bash
export SELFPLAY_RUN=/absolute/path/to/outputs/selfplay/kqapro-v03
export SELFPLAY_ROUND_DIR="$SELFPLAY_RUN/round_001"
```

列出当前 round 的 Trainer 目录和 reward attempt：

```bash
find "$SELFPLAY_ROUND_DIR" -maxdepth 3 -name logging.jsonl -print
find "$SELFPLAY_ROUND_DIR/logs" -maxdepth 2 \
  -name 'reward_components.rank-*.jsonl' -print
```

从上面选择当前训练实际使用的目录：

```bash
export SELFPLAY_TRAINER_DIR=/absolute/path/to/round_001/v0-YYYYMMDD-HHMMSS
export SELFPLAY_METRICS_DIR="$SELFPLAY_ROUND_DIR/logs/metrics_attempt_001"
```

如果 round 中存在多个 `v*` 或 `metrics_attempt_*`，不要混读旧 retry。正在运行时通常选择修改时间
最新的目录；完成后以 `round_001/manifest.json` 中记录的 `adapter` 和 `reward_metrics_dir` 为准。

## 2. 检查 base model 和 SFT adapter 是否正确加载

查看 self-play 实际计划：

```bash
jq '{
  base_model: .train_environment.MODEL_PATH,
  adapter: .train_environment.LORA_ADAPTER_PATH,
  adapter_in: .adapter_in,
  graphscript_version: .graphscript_version,
  interaction_mode: .interaction_mode
}' "$SELFPLAY_ROUND_DIR/plan.json"
```

查看 SFT LoRA 记录的基础模型：

```bash
export SELFPLAY_ADAPTER=$(jq -r '.train_environment.LORA_ADAPTER_PATH' \
  "$SELFPLAY_ROUND_DIR/plan.json")
jq '{base_model_name_or_path, peft_type, r, lora_alpha}' \
  "$SELFPLAY_ADAPTER/adapter_config.json"
```

检查 ms-swift 实际启动命令：

```bash
rg -n -- '--model |--adapters |--model_type |--num_generations ' \
  "$SELFPLAY_ROUND_DIR/logs/ms_swift.log" | head -n 10
```

正确关系应为：

```text
MODEL_PATH            = 原始基础模型
LORA_ADAPTER_PATH     = Solver SFT LoRA checkpoint
base_model_name_or_path ≈ MODEL_PATH
```

以下配置会重复应用 SFT，属于错误配置：

```text
MODEL_PATH        = 已合并 SFT 模型
LORA_ADAPTER_PATH = 同一个 SFT LoRA
```

若这里不正确，先修正模型加载，不要调整 reward。

## 3. 确认 GRPO 的真实运行参数

```bash
jq '{
  learning_rate,
  num_generations,
  steps_per_generation,
  gradient_accumulation_steps,
  scale_rewards,
  beta,
  dynamic_sample,
  max_resample_times,
  overlong_filter,
  max_completion_length,
  temperature,
  top_p,
  seed
}' "$SELFPLAY_TRAINER_DIR/args.json"
```

注意：`steps_per_generation > 1` 时，一批 rollout 会被多个 optimizer step 复用。连续两次
`loss=0` 不一定代表完成了两次独立采样；判断持续时间时应按新的 generation batch 计数。

## 4. 查看 Trainer 核心指标

```bash
jq -c 'select(.loss != null or .reward_std != null) | {
  step: ."global_step/max_steps",
  loss,
  reward,
  reward_std,
  grad_norm,
  learning_rate,
  mean_length: ."completions/mean_length",
  clipped_ratio: ."completions/clipped_ratio",
  kl
}' "$SELFPLAY_TRAINER_DIR/logging.jsonl"
```

优先按下表判断：

| 观测 | 初步结论 |
|---|---|
| `reward_std=0, grad_norm=0` | 同 prompt rollout 全部同分，无 GRPO advantage |
| `reward_std>0, grad_norm>0` | 有任务梯度；`loss=0` 本身不代表没有学习 |
| `reward_std>0, grad_norm=0, clipped_ratio=0` | reward 差异未传到参数，继续执行第 5–8 节 |
| `clipped_ratio≈1` | completion 截断；先修输出长度或 thinking 模式 |
| `mean_length≈0` | completion mask 可能为空，即使 `clipped_ratio=0` |
| `reward_std=0, KL>0` | 可能只有 KL 正则更新，没有任务 reward 梯度 |

GRPO 在同一 prompt 的 `num_generations` 个 completion 内计算：

```text
advantage_i = (reward_i - group_mean) / (group_std + 1e-4)
```

Questioner 与 Solver 属于不同 prompt group；两个角色之间的 reward 差异不会互相产生 advantage。

## 5. 检查“不同 reward 是否来自不同 completion”

这是 `reward_std>0, grad_norm=0` 时最有判别力的检查。

```bash
python - <<'PY'
import json
import os
from pathlib import Path

trainer_dir = Path(os.environ["SELFPLAY_TRAINER_DIR"])
path = trainer_dir / "completions.jsonl"
if not path.is_file():
    raise SystemExit(f"missing {path}")

for line_number, line in enumerate(path.read_text().splitlines(), start=1):
    if not line.strip():
        continue
    row = json.loads(line)
    completions = [str(value) for value in row.get("completion", [])]
    rewards = [float(value) for value in row.get("GraphTaskReward", [])]
    prompts = row.get("prompt", [])
    prompt = str(prompts[0]) if prompts else ""
    role = "questioner" if "Questioner" in prompt else "solver"
    print({
        "line": line_number,
        "step": row.get("step"),
        "role": role,
        "samples": len(completions),
        "unique_completions": len(set(completions)),
        "rewards": rewards,
        "unique_rewards": len(set(rewards)),
        "completion_prefixes": [value[:120].replace("\n", "\\n") for value in completions],
    })
PY
```

判定：

| `unique_completions` | `unique_rewards` | 结论 |
|---:|---:|---|
| `1` | `>1` | 同一 action 得到不同 reward；优先怀疑 stochastic opponent / archive 顺序噪声 |
| `>1` | `>1` | 存在真实 action-dependent variance；继续检查梯度和 checkpoint |
| `>1` | `1` | reward 过粗，不能区分不同 completion |
| `1` | `1` | actor 采样坍缩；该组理论上 `reward_std=0` |

Questioner 的 frozen Solver 当前使用随机采样；因此同一 GraphScript 可能得到不同 pass rate。若
`unique_completions=1, unique_rewards>1` 主要发生在 Questioner，`reward_std` 很可能是环境噪声而非
可学习信号。相同 token 序列对应相同策略梯度，正负 advantage 会抵消，可能得到
`grad_norm=0`。

## 6. 统计 GraphScript 解析失败原因

以下检查只解析 completion，不执行程序，也不会访问或修改 archive：

```bash
PYTHONPATH=. python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

from graphtask_r1.graphscript import GraphScriptError, parse_graphscript

path = Path(os.environ["SELFPLAY_TRAINER_DIR"]) / "completions.jsonl"
counts = Counter()
examples = {}

for line in path.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    prompts = row.get("prompt", [])
    prompt = str(prompts[0]) if prompts else ""
    role = "questioner" if "Questioner" in prompt else "solver"
    for completion in row.get("completion", []):
        text = str(completion)
        try:
            parse_graphscript(text, max_follow_limit=100)
            reason = "PARSE_OK"
        except GraphScriptError as exc:
            reason = exc.reason_code
        counts[(role, reason)] += 1
        examples.setdefault((role, reason), text[:300].replace("\n", "\\n"))

for key, count in sorted(counts.items()):
    print({"role": key[0], "reason": key[1], "count": count, "example": examples[key]})
PY
```

常见原因：

| reason | 含义 |
|---|---|
| `NON_JSON` | 输出不是 JSON，常见于 `<think>`、自然语言或空输出 |
| `EXTRA_TEXT` | JSON 前后有解释、Markdown 或其他文本 |
| `UNSUPPORTED_VERSION` / `VERSION_MISMATCH` | SFT 与 self-play GraphScript 版本不一致 |
| `INVALID_SCHEMA` / `UNKNOWN_OP` | JSON 合法但 GraphScript schema 错误 |
| `INVALID_SHAPE` | 操作链不符合当前版本约束 |
| `LIMIT_EXCEEDED` | follow limit 超出训练配置 |
| `PARSE_OK` | 只代表解析成功；仍可能在 relation、执行或认证阶段失败 |

## 7. 查看按角色 reward components

```bash
for file in "$SELFPLAY_METRICS_DIR"/reward_components.rank-*.jsonl; do
  echo "$file"
  tail -n 5 "$file" | jq '{sequence, rank, samples, roles}'
done
```

解释：

- Questioner `score=-0.35`：进入 GraphScript/格式拒绝路径，即 `-1 × 0.35`。
- Solver 出现 `format` 或 `reject_*`：解析或执行失败。
- Solver 有 `f1=0, exact_match=0` 且无格式拒绝：程序可执行，但答案错误。
- `opponent_success_rate`：frozen Solver 对 Questioner 新任务的采样成功率。

当前日志的 component `means` 只在该 component 出现的样本上聚合。因此 `reject_* = 1` 表示该拒绝
发生过，不应直接解释为整个 batch 的拒绝率。

## 8. 检查 LoRA 是否真的更新

先确认有可训练参数：

```bash
rg -n "Trainable|trainable|model_parameter_info" \
  "$SELFPLAY_ROUND_DIR/logs/ms_swift.log" | tail -n 10
```

先比较连续 checkpoint 的 adapter hash：

```bash
find "$SELFPLAY_TRAINER_DIR" -path '*/checkpoint-*/adapter_model.safetensors' \
  -print0 | sort -z | xargs -0 -r sha256sum
```

判定：

- 连续 checkpoint hash 完全相同：参数没有变化，`grad_norm=0` 是真实 no-op。
- hash 不同：只能说明文件字节不同，不能单独证明 tensor 已更新；继续执行第 14 节的逐 tensor 比较。
- 没有 trainable parameter 或 adapter checkpoint：优先检查 LoRA 加载和 `target_modules`。

## 9. 最终决策表

| 排查结果 | 主要问题 | 后续修复方向 |
|---|---|---|
| model/adapter 关系错误 | SFT 加载问题 | 使用 base model + 单独 SFT LoRA，避免重复应用 |
| Solver 单独也大量解析失败 | SFT/checkpoint 问题 | 检查 SFT 数据、版本、catalog、checkpoint 选择 |
| Solver 正常，Questioner 大量 `-0.35` | Solver-only SFT 导致 Questioner 冷启动 | Questioner warm-up/curriculum + 分阶段 reward |
| 相同 completion 得到不同 reward | stochastic reward 噪声 | opponent 显式 seed；按 task/signature 缓存同轮评估 |
| 不同 completion 全部同分 | reward 过粗 | 增加 JSON/schema/执行/认证的稠密分阶段 reward |
| `reward_std>0`、completion 不同、checkpoint 不变 | 梯度/训练器问题 | 单 GPU bounded smoke；检查 trainable LoRA、mask、DDP 切分 |
| `reward_std>0`、checkpoint 改变 | 实际有更新 | `loss=0` 可能只是中心化 advantage 的标量表现 |

不要用增大 `0.35/0.65` 权重解决 zero-gradient：GRPO 先在每个 prompt group 内中心化 reward，正比例
缩放不能创造组内 action variance。也不要仅靠提高 `num_generations` 掩盖系统性格式失败；先确认
completion 合法、reward 可复现且与 action 质量相关。

## 10. 回传诊断信息

完成检查后，建议回传以下五组输出即可定位：

1. 第 2 节 `plan.json` 的 model/adapter 摘要；
2. 第 3 节 GRPO 参数摘要；
3. 第 4 节最近两个 generation batch 的 Trainer 指标；
4. 第 5 节 `unique_completions/unique_rewards` 输出；
5. 第 6 节解析原因计数和第 8 节 checkpoint hash。

路径、主机名和模型私有目录可以脱敏，但请保留 GraphScript version、数值指标、reject reason 和各组
completion/reward 的唯一值数量。

## 11. 补充诊断：奇数 step 有 reward、偶数 step 为 null

如果日志呈现：

```text
step 1, 3, 5, ...: reward 有值
step 2, 4, 6, ...: reward=null, reward_std=null
```

先检查实际参数：

```bash
jq '{
  steps_per_generation,
  gradient_accumulation_steps,
  num_generations,
  temperature,
  scale_rewards
}' "$SELFPLAY_TRAINER_DIR/args.json"
```

当 `steps_per_generation=2` 时，通常只有 1、3、5 等 step 生成新 rollout 并记录 reward；偶数 step
复用上一批 rollout，Trainer 日志不重复写 generation/reward 指标。因此偶数 step 的 `null` 不是
reward 为 0，也不是 Questioner/Solver 交替；它本身属于预期日志行为。

真正需要处理的是 generation step 同时满足：

```text
reward_std=0, loss=0, grad_norm=0
```

这表示同一 prompt 的 `num_generations` 个 completion 得到相同 reward，组内 advantage 为 0。

## 12. 必须按 prompt group 统计 completion 和 reward

第 5 节的快速检查会汇总一整行 generation batch。在 mixed-role 或一个 batch 含多个 prompt 时，
`unique_rewards=2` 可能只来自：

```text
Questioner reward = -0.35
Solver reward = 0
```

两个角色或两个不同 prompt 之间的 reward 差异不能产生 GRPO advantage。使用下面的检查按完整 prompt
精确分组：

```bash
python - <<'PY'
import json
import os
from collections import defaultdict
from pathlib import Path

path = Path(os.environ["SELFPLAY_TRAINER_DIR"]) / "completions.jsonl"
if not path.is_file():
    raise SystemExit(f"missing {path}")

for line_number, line in enumerate(path.read_text().splitlines(), start=1):
    if not line.strip():
        continue
    row = json.loads(line)
    prompts = row.get("prompt", [])
    completions = row.get("completion", [])
    rewards = row.get("GraphTaskReward", [])
    steps = row.get("step", [])
    if isinstance(prompts, str):
        prompts = [prompts]
    if not isinstance(steps, list):
        steps = [steps] * len(completions)
    if not (len(prompts) == len(completions) == len(rewards) == len(steps)):
        raise ValueError(
            f"unaligned completion row {line_number}: "
            f"prompts={len(prompts)}, completions={len(completions)}, "
            f"rewards={len(rewards)}, steps={len(steps)}"
        )

    groups = defaultdict(list)
    for prompt, completion, reward, step in zip(
        prompts, completions, rewards, steps, strict=True
    ):
        groups[str(prompt)].append((str(completion), float(reward), step))

    for group_number, (prompt, values) in enumerate(groups.items(), start=1):
        group_completions = [value[0] for value in values]
        group_rewards = [value[1] for value in values]
        role = "questioner" if "Questioner" in prompt else "solver"
        print({
            "line": line_number,
            "step": values[0][2],
            "group": group_number,
            "role": role,
            "samples": len(values),
            "unique_completions": len(set(group_completions)),
            "unique_rewards": len(set(group_rewards)),
            "rewards": sorted(set(group_rewards)),
        })
PY
```

如果结果主要是：

```text
questioner: unique_completions > 1, unique_rewards = 1, rewards = [-0.35]
solver:     unique_completions > 1, unique_rewards = 1, rewards = [0.0]
```

则不同 action 全部落入同一失败档位。此时 `reward_std=0` 与 Trainer 行为一致，问题是输出契约失败
和 reward 过粗，不是增加 `rollout_n` 就能解决的采样数量问题。

## 13. 定位 EXTRA_FIELD 的准确字段路径

Questioner 的 `score=-0.35` 表示 GraphScript 解析、schema、shape 或执行阶段被拒绝。若
`EXTRA_FIELD` 占主导，不要直接把 parser 改成忽略额外字段；先确认模型究竟添加了什么：

```bash
PYTHONPATH=. python - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

from pydantic import ValidationError
from graphtask_r1.graphscript.schema import GraphScript

path = Path(os.environ["SELFPLAY_TRAINER_DIR"]) / "completions.jsonl"
counts = Counter()
examples = {}

for line in path.read_text().splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    prompts = row.get("prompt", [])
    completions = row.get("completion", [])
    for prompt, completion in zip(prompts, completions, strict=True):
        role = "questioner" if "Questioner" in str(prompt) else "solver"
        try:
            raw = json.loads(str(completion))
            GraphScript.model_validate(raw)
        except json.JSONDecodeError:
            continue
        except ValidationError as exc:
            for error in exc.errors():
                if error["type"] != "extra_forbidden":
                    continue
                location = ".".join(map(str, error["loc"]))
                key = (role, location)
                counts[key] += 1
                examples.setdefault(key, error.get("input"))

for (role, location), count in counts.most_common():
    print({
        "role": role,
        "field_location": location,
        "count": count,
        "example_value": examples[(role, location)],
    })
PY
```

常见结果的解释：

| 路径 | 含义 | 优先处理 |
|---|---|---|
| 顶层 `question`、`answer`、`explanation` | 模型不知道顶层只能有 `version`、`ops` | 强化 prompt，并加入 Questioner SFT |
| `ops.*.input`、`ops.*.output` | 模型未使用 GraphScript 字段别名 `in`、`out` | prompt 给出严格 JSON 示例 |
| `ops.*` 下其他未知参数 | operation signature 不匹配 | 对照 v0.3 grammar 和 SFT target |
| 标准 SFT target 本身包含该字段 | parser、版本或数据不一致 | 核对数据和运行时 GraphScript 版本 |

当前 KQAPro 主线 SFT 只训练 Solver。若 Questioner 的额外字段远多于 Solver，这更符合
Questioner 未经 SFT 的冷启动，而不是 SFT adapter 没有加载。

## 14. 区分 Solver 格式失败与可执行但答案错误

Solver 的 reward 为 0 有两种不同含义：GraphScript 被拒绝，或者程序成功执行但答案 F1 为 0。
按 rank 查看 component：

```bash
for file in "$SELFPLAY_METRICS_DIR"/reward_components.rank-*.jsonl; do
  echo "$file"
  jq -c '
    .roles.solver
    | select(. != null)
    | {
        samples,
        f1: .means.f1,
        exact_match: .means.exact_match,
        format: .means.format,
        graph_calls: .means.graph_calls,
        rejects: (.means | with_entries(select(.key | startswith("reject_"))))
      }
  ' "$file"
done
```

- 有 `reject_*`：优先修 GraphScript 输出契约。
- 无 `reject_*`、`graph_calls>0`、`f1=0`：程序合法可执行，但语义错误。
- `f1` 偶尔大于 0，但 Trainer 仍报告 `reward_std=0`：回到第 12 节，确认成功和失败是否发生在同一
  prompt group；跨 prompt 的差异不产生 advantage。

## 15. SHA 不同后逐 tensor 比较 LoRA

不要仅比较初始 SFT adapter 与第一个 GRPO checkpoint：保存格式、key 或 dtype 变化也可能改变文件
SHA。优先选择两个连续 GRPO checkpoint：

```bash
export CKPT_A="$SELFPLAY_TRAINER_DIR/checkpoint-1/adapter_model.safetensors"
export CKPT_B="$SELFPLAY_TRAINER_DIR/checkpoint-2/adapter_model.safetensors"

python - <<'PY'
import os

import torch
from safetensors.torch import load_file

a = load_file(os.environ["CKPT_A"], device="cpu")
b = load_file(os.environ["CKPT_B"], device="cpu")

print("same_keys:", set(a) == set(b))
print("tensor_count:", len(a))

changed = 0
max_abs_delta = 0.0
sum_sq = 0.0
for name in sorted(set(a) & set(b)):
    left = a[name].float()
    right = b[name].float()
    delta = left - right
    if not torch.equal(left, right):
        changed += 1
        max_abs_delta = max(max_abs_delta, delta.abs().max().item())
        sum_sq += delta.square().sum().item()

print({
    "changed_tensors": changed,
    "max_abs_delta": max_abs_delta,
    "global_l2_delta": sum_sq ** 0.5,
})
PY
```

判定：

- `changed_tensors=0`：SHA 差异来自文件层，模型参数实际未改变。
- tensor 改变量极小：同时检查未四舍五入的 `grad_norm`、`kl` 和当步 learning rate。
- tensor 明显变化：查找是否有少数 prompt group 存在非零 reward variance；如果所有组都同分，则需
  继续检查 Trainer 指标采集位置或其他 loss（例如 KL）。

## 16. 当前已观测模式的处理顺序

若观测到 Questioner 的 `EXTRA_FIELD` 数百条，另有少量 `INVALID_SCHEMA`、`INVALID_SHAPE`、
`NON_JSON`，Solver 错误更少，应按以下顺序处理：

1. 暂停完整规模 self-play，保留当前 bounded run 作为诊断样本。
2. 用第 13 节确定额外字段路径，不要直接放宽 parser。
3. 在 Questioner v0.3 prompt 中明确顶层只能是 `{"version":"0.3","ops":[...]}`，并提供最小合法
   JSON 示例。
4. 从 KQAPro train 的认证 program 导出 Questioner SFT 行，对共享 LoRA 做 Questioner/Solver
   混合或分阶段 warm-up；当前 Solver-only SFT 无法保证 Questioner 冷启动成功。
5. 将 reward 改为可区分的阶段信号：非 JSON、额外字段、schema、shape/handle、可执行、认证/F1。
   保留最终严格认证目标，不把无效程序当作成功。
6. 重新运行小规模 smoke；只有同一 prompt group 出现 `unique_rewards>1`，且 generation step 出现
   `reward_std>0`、`grad_norm>0` 后，再恢复完整训练。
