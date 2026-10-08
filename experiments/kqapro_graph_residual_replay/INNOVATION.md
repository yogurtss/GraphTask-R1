# GRR：图认证的当前策略残差回放

> 历史机制文档：文末所述 execution-boundary 方向后来已实现为 ECDB v1/v2，但两轮均未获得经程序
> 语义验证的净精度提升。当前可保留的窄 novelty 主线、失败证据与下一轮可证伪方案见
> [ECDB_FAILURE_ANALYSIS.md](ECDB_FAILURE_ANALYSIS.md)。

> GRR mini 实验状态（2026-08-31）：128-pair dev A/B 的 EM 与 F1 均持平，仅 tool success `9→10`，硬门禁
> 失败；本文档描述的是可证伪机制，不构成已提高精度的主张。实证边界见
> [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md)。

## 一句话创新

GRR 将当前 Solver 在真实 train 问题上的**原始失败响应**视为当前策略的 task-local residual；它用
同一知识图谱重新认证该题的 gold GraphScript，然后直接形成“同题 certified gold / 当前策略真实
错误”的偏好对。模型更新后重新采样新策略，错误分布随策略一起变化，从而形成 graph-grounded
self-evolution。

```text
当前策略错误 -> 同图执行分类 -> 同题 gold 认证 -> 残差偏好更新 -> 新策略重新采样
```

这里的关键不是“Graph + DPO”的简单组合，而是训练负例与三项状态的同时绑定：当前模型快照、具体
task、可重放图执行状态。旧 ERE 的历史 completion 混合多个训练 step，且抽出的全局槽优先级会用于
无关问题；在当前仓库数据上，它的有效 task-local residual 几乎为零。GRR 修正的是这一条因果断点。

## 机制

给定固定图快照 `G`、执行预算 `B`、train 问题 `x_i`、certified program `P_i+` 与进入本轮的固定
incumbent `pi_t`：

\[
\hat P_i \sim \pi_t(\cdot\mid x_i; T=0), \qquad
y_i^+=\operatorname{Exec}(P_i^+,G;B).
\]

首先要求 `y_i+` 与数据任务的 gold answer 完全一致。随后对模型原始响应依次做 GraphScript parse
和 graph execution，得到结构化残差：

\[
r_i\in\{\text{parse},\text{execution},\text{terminal-semantic},\text{correct}\}.
\]

只有 `r_i != correct` 的样本进入训练。偏好对为：

\[
D_t=\{(x_i,P_i^+,\hat P_i):
\operatorname{Exec}(P_i^+,G;B)=y_i,\;r_i\ne\text{correct}\}.
\]

其中 rejected 保留 `pi_t` 的实际响应；解析失败不会被清洗成一个看似合法的程序，执行失败也不会被
替换为从别题槽统计合成的 negative。标准 DPO 更新为：

\[
\mathcal L_{\mathrm{GRR}}=-\mathbb E_{D_t}\log\sigma\left(\beta\left[
\log{\pi_\theta(P_i^+\mid x_i)\over\pi_t(P_i^+\mid x_i)}-
\log{\pi_\theta(\hat P_i\mid x_i)\over\pi_t(\hat P_i\mid x_i)}
\right]\right).
\]

得到 `pi_(t+1)` 后必须重新 rollout；不能继续把 `D_t` 宣称为新模型的 on-policy residual。完整迭代
由每轮模型 artifact hash、probe task IDs、sample seed、graph/data hash、response hash 和 execution
certificate 重放。

## 为什么有算法价值

当前 KQAPro dev 的主要损失不是“合法程序里某一个 relation 选错”，而是 NON_JSON、ENTITY、SCHEMA
等 parse/执行失败。对这些错误强行构造 executable single-slot counterfactual，会改变问题分布并丢掉
当前策略真正的高频失败。GRR 对不同阶段不设人工 quota，直接优化固定快照的经验错误分布。

它只替换训练 pair 的生成原则：

- graph executor 同时充当 gold 认证器与错误阶段测量器；
- task-local 约束阻止全局错误统计被错误施加到无关题；
- fixed-snapshot 约束避免跨训练 step 的旧错误冒充当前策略残差；
- raw-rejection 约束保留模型真正需要消除的格式、schema、entity 和语义错误。

这四项是同一个数据生成合同的组成条件，没有新增神经模块。推理路径、prompt、relation catalog、
解码和执行预算均保持不变，因此可把精度变化归因到 residual replay。

## 与常见路线的边界

GRR 不声称首创 learning from mistakes、DPO 或 execution-guided learning。它与相邻路线的实质边界是：

- 与离线 error replay 相比：negative 必须由已登记的当前固定策略在同一轮直接生成；
- 与合成 hard-negative 相比：不编辑 gold 程序，也不需要假设合成 candidate 对模型足够困难；
- 与 outcome-only preference 相比：图执行保留 parse、execution、terminal semantic 的结构化残差和
  证书，而不是只记录一个 0/1 标签；
- 与 self-training 相比：chosen 不是模型伪标签，而是 certified program 的再执行结果；
- 与额外 critic/refiner 相比：不增加新的推理模型调用；LoRA 可合并回 base，但未合并 adapter 的
  serving 开销仍需单独报告。

本实验的可发表组合主张应严格写成：**fixed-snapshot on-policy failure capture + same-task graph-certified
positive + raw residual preference replay + policy-refresh evolution**。正式论文仍需完成系统相关工作
检索，不能把这份工程设计文档当成新颖性证明。

## 可证伪假设与最小实验

主假设只有一个：在相同模型、128 pairs、token budget、optimizer、prompt 和评测任务下，GRR 比
incumbent 严格提高未见 dev EM，同时 F1/tool success 不退化。

诊断量用于解释结果，不扩展主张：

1. 256 个固定 train probe 中的 gold certification yield 与真实 failure yield；
2. parse / execution / terminal-semantic 的 pair 分布；
3. DPO chosen/rejected reward margin；
4. 同一 dev 的 candidate-only 与 baseline-only task、exact McNemar；
5. 失败原因从 baseline 到 candidate 的配对迁移。

第一轮不加入 stage-balanced sampling、多个 rollout、counterfactual fallback 或 curriculum；这些都会让
无法判断提升来自 on-policy task-local residual 还是额外采样策略。若 256 probe 不能自然得到 128 个
错误，停止而不是自适应扩样。若 dev 不严格提升，停止且不打开 blind。

## 论文与专利角度的可保护链条

论文价值在于把“self-evolution 的反馈是否属于当前策略、当前问题”变成可审计实验变量，并与历史
混合快照/跨题合成负例做等预算消融。若有稳定提升，可进一步研究刷新频率与 residual distribution
drift，而不是继续叠模块。

专利角度可讨论的技术链条是：

1. 固定待更新模型快照，对确定的图问答任务集合生成带 seed/trace 的程序响应；
2. 在同一图快照、allowlist 和预算下，对认证程序与预测响应分别执行并记录阶段化残差；
3. 仅将同一任务的认证程序和该模型快照的原始失败响应绑定为带 hash 的偏好证书；
4. 依据证书更新模型，并在下一轮强制刷新模型 hash 与响应集合；
5. 用配对未见任务门禁决定是否允许模型晋级。

可量化技术效果包括：消除跨题 negative 错配、减少已过时 residual 的训练占比、让每个偏好对可由
图执行重放、且不增加新的推理模型调用。合并后的延迟是否持平必须另测，不能由结构直接推出。上述
只是潜在权利要求链与实验记录建议，不是专利新颖性、创造性或 FTO 意见；申请前仍需代理人结合检索
和正向实验确定范围。

## 停止条件

- certified gold 少于 probe：按固定规则跳过，不能用模型答案补 gold；
- 真实失败少于 128：停止，不改变 probe 或采样次数；
- candidate dev correct count 未严格增加，或 F1/tool success 下降：停止，不打开 blind；
- 请求失败不为零：本轮 A/B 无效，修复运行条件后重跑同一模型，不能算算法结果；
- dev 胜出但 blind 未胜：不晋级，只保留负结果与配对误差分析。

## 本轮负结果后的唯一后续方向

本轮说明 task-local、current-snapshot 约束是必要的数据合同，但**完整响应级 DPO**仍把 parse、执行和
语义错误压成同一个序列偏好。训练中 mean preference accuracy 仅 50.78%、mean reward margin 仅
0.0041；dev 上修复一个正确题的同时损失另一个正确题，表现为错误类型迁移而非净减少。

如果以后开启新实验，优先检验“graph execution-boundary credit assignment”：保留同一 current-policy
rollout 和 same-task certified gold，在 parser/SSA executor 判断的最早失败 operation 处对齐 gold 与
prediction prefix，只对该 operation 到下一个可执行边界施加偏好损失，已执行正确的前缀保持不动。
它仍是一条 graph + self-evolution 因果链，不需要新增模型；核心可证伪点是能否在相同 pair/token
预算下减少 `NON_JSON/TYPE_MISMATCH/INVALID_SCHEMA`，同时不破坏原本正确的程序。

该方向目前只作为失败后的算法假设：本轮没有实现、训练或评测，也不能写成精度提升或新增实验结果。
