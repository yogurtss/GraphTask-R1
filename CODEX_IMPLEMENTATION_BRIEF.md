# Codex Implementation Brief：GraphTask-R1

## Mission

实现一个可重放、可测试的研究代码库，用于：

1. 在可执行知识图谱上构造类型化查询程序；
2. 将程序语言化为自然问题；
3. 通过程序执行、反事实干预和 shortcut 检测验证任务质量；
4. 把程序编译为图工具 Solver 的冷启动轨迹；
5. 先完成静态数据 utility 实验；
6. 第一版即接入 `verl`，让同一个 3B–4B 共享策略分别扮演 Questioner 和 Solver，并完成最小 self-play 共进化闭环。

完整研究规格见：`GraphTask-R1_PROJECT_EXECUTION_PLAN.md`。

---

## Non-negotiable Rules

- 先实现环境与 verifier，最后实现 RL。
- 训练热路径使用纯 Python `reset/step` 状态机，不使用 LangGraph。
- 所有环境状态可 JSON 序列化、恢复和 replay。
- 所有随机性显式传入 seed。
- 所有图调用有 timeout、retry、cache、trace id。
- Gold answer 只能来自程序执行，不能信任 LLM 自报答案。
- Reward 必须返回各分量，不能只返回 scalar total。
- 被拒绝样本保留 reason codes。
- 每个 PR 只处理一个 issue，并附测试、CLI 示例和验收命令。
- 首版允许进行 mini self-play，但在 mini gate 未通过前不得扩大到 Freebase 或长周期训练。

---

## First 10 Issues

### Issue 00 — Bootstrap

Deliver:

- `pyproject.toml`、`uv.lock`、lint/type/test；
- Hydra config；
- `src/` layout；
- `AGENTS.md`；
- CI；
- manifest utility。

Acceptance:

```bash
uv sync
make lint
make typecheck
make test
```

全部通过。

### Issue 01 — Schemas

实现：

- EntityInfo、Triple、Answer、AnswerSet；
- Program AST discriminated union；
- TaskCertificate；
- ToolCall / Observation / Trajectory；
- VerifierResult / RewardBreakdown。

Acceptance:

- JSON round-trip；
- invalid schema 有清晰错误；
- unit tests。

### Issue 02 — ToyGraph Backend

实现 `GraphBackend` 和 `InMemoryGraphBackend`。

Acceptance:

- golden graph cases；
- deterministic sorting；
- overlay 不修改 base graph。

### Issue 03 — DSL Executor and Signatures

实现：

- Entity、Hop、Intersect、FilterType、FilterLiteral、Count；
- execute；
- canonical signature；
- weighted cost。

Acceptance:

- property tests；
- canonicalize 幂等；
- 1000 随机合法 AST 无崩溃。

### Issue 04 — SPARQL Compiler

实现 AST → SPARQL。

Acceptance:

- escaping tests；
- ToyGraph 参考执行结果一致；
- snapshot tests。

### Issue 05 — Interventions

实现：

- drop filter；
- drop intersection branch；
- bypass hop；
- relation replacement；
- graph triple removal overlay；
- necessity mean/min。

Acceptance:

- 冗余条件得低分；
- 必要条件得高分；
- 原 AST 和图保持不变。

### Issue 06 — Shortcut Detector

实现有界低成本程序枚举与 denotation equality cache。

Acceptance:

- toy shortcut case 检出；
- 无 shortcut case 不误报；
- 超预算时安全退出并标记 `unknown`，不能默认为 false。

### Issue 07 — Static Program Sampler

实现 typed constrained sampler。

Acceptance:

- 支持 operator quotas；
- 记录 rejected partial programs；
- 相同 seed 完全复现；
- 输出 Parquet + manifest。

---


### Issue 08 — Shared Role Policy

实现一个共享模型封装：

- 单一 `policy_name_or_path`；
- 单一 LoRA adapter 与 optimizer；
- `questioner` / `solver` role prompt；
- role-specific tool registry 与 output parser；
- checkpoint 中保存统一参数和角色配置。

Acceptance:

- 两个角色参数对象 identity 相同；
- 任一 optimizer step 后两个角色看到同一新权重；
- role prompt 和工具权限严格隔离；
- 可在 3B–4B 模型上完成双角色 inference smoke test。

### Issue 09 — Mini Self-Play Loop

实现 round-based self-play：

```text
Questioner rollout
→ deterministic verification
→ frozen snapshot Solver evaluation
→ current Solver rollout
→ role-wise advantage normalization
→ joint GRPO update of shared policy
→ archive and snapshot update
```

Acceptance:

- ToyGraph 上连续运行 3 rounds；
- 每轮可完整 resume；
- 输出 questioner/solver 独立 reward breakdown；
- 记录共享参数梯度范数与角色梯度余弦相似度；
- 能通过配置将 Questioner/Solver loss weight 设为 0 做单角色消融。

## Required Core Interfaces

```python
class GraphBackend(Protocol):
    def neighbors(...) -> list[Triple]: ...
    def execute_program(self, program: Program) -> AnswerSet: ...
    def execute_sparql(self, sparql: str) -> AnswerSet: ...
    def entity_info(self, entity_id: str) -> EntityInfo: ...
    def extract_witness(self, program: Program, answers: AnswerSet) -> list[Witness]: ...
    def with_overlay(self, overlay: GraphOverlay) -> "GraphBackend": ...

class ToolEnvironment(Protocol):
    def reset(self, sample: EpisodeInput, seed: int) -> Observation: ...
    def step(self, action: AgentAction) -> StepResult: ...
    def snapshot(self) -> dict: ...
    def restore(self, state: dict) -> None: ...
```

Do not introduce global singletons for graph backends, RNGs, configs, models, or archives.

---

## Initial Repository Tree

```text
src/graphtask_r1/
  schema/
  graph/
  dsl/
  envs/
  generation/
  verification/
  rewards/
  archive/
  training/
  evaluation/
tests/
  fixtures/
  unit/
  property/
  integration/
  e2e/
```

---

## Initial Milestone

第一个里程碑仍先完成确定性数据管线；紧接着的第一版验收目标是共享策略 mini self-play，而不是停留在静态数据生成：

```text
ToyGraph/KQA Pro mini
→ sample executable programs
→ compute answers
→ run interventions
→ reject shortcuts/redundancy
→ serialize TaskCertificate
→ compile canonical Solver tool trace
→ replay trace and recover exact answer
```

Milestone acceptance command:

```bash
python -m graphtask_r1.cli e2e mini-pipeline \
  --graph toy \
  --num-programs 100 \
  --seed 42 \
  --output-dir outputs/e2e-mini
```

The command must produce:

- `programs.parquet`；
- `tasks.parquet`；
- `traces.parquet`；
- `rejections.parquet`；
- `metrics.json`；
- `manifest.json`；
- zero unrecoverable errors；
- 100% replay accuracy for accepted canonical traces。


## First-Version Self-Play Acceptance

```bash
python -m graphtask_r1.cli train mini-self-play \
  --graph toy \
  --model <3B-4B-instruct-checkpoint> \
  --shared-policy true \
  --rounds 3 \
  --questioner-groups 16 \
  --solver-episodes 64 \
  --seed 42 \
  --output-dir outputs/mini-self-play
```

必须生成：

- `round_*/questioner_rollouts.parquet`；
- `round_*/solver_rollouts.parquet`；
- `round_*/reward_breakdown.json`；
- `round_*/gradient_diagnostics.json`；
- `round_*/task_archive.parquet`；
- `checkpoints/shared_policy_round_*`；
- 可从任意 round 恢复的 manifest。
