# Certified Evidence Self-Play for KILT

## Research position

The mainline starts directly from online RL. SFT is not a prerequisite and is retained only
as a diagnostic ablation. The implementation stays on ms-swift 3.10.3:

- Solver: GRPO over four typed, multi-turn evidence-tool rollouts per question.
- Questioner: REINFORCE++ over certified candidate challenges.
- Shared-policy order: Solver update, then Questioner update.
- Gold answers and provenance: derived only by executing a hidden ProofScript.

This follows the direct self-play principle of
[CoEvoKG](https://arxiv.org/abs/2608.01904) and its
[official implementation](https://github.com/lazzy1225/CoEvoKG), but does not copy its
heuristic path reward or unrestricted memory write-back.

## Main algorithmic difference

Each episode has two programs with asymmetric information:

1. The hidden `ProofScript v0.4-proof` addresses the bridge page, target page, exact
   paragraphs and answer-producing operation. Executing it is the only source of gold.
2. The Solver calls bounded `text_search`, `expand_evidence` and `select_evidence` tools,
   then emits a JSON answer. The reward worker records the calls as a replayable
   `SearchScript v0.4` trajectory and executes each retrieval against the KILT graph.

The Solver reward is decomposed into answer F1, KILT-F1, provenance R-precision, recovered
proof prefix and grounded search efficiency. Failures are localized into:

- retrieval residual: at least one certified proof passage was not recovered;
- reasoning residual: all proof passages were recovered but the answer was wrong;
- success: the answer and proof evidence were both recovered.

The Questioner is rewarded on a two-dimensional competence frontier rather than only on
answer pass rate. For a candidate task, let `r` be certified evidence recovery and let `a`
be answer accuracy conditional on full recovery. Its central term is

`exp(-((r-t)^2 + (a-t)^2) / (2 sigma^2))`.

This avoids assigning the same curriculum signal to “cannot find the second hop” and
“found both hops but cannot reason over them.” Proof validity and task novelty are retained
as separate auditable reward components.

## CoEvoKG comparison

| Dimension | CoEvoKG | This implementation |
| --- | --- | --- |
| Main training | Direct joint self-play RL | Direct alternating self-play RL |
| Solver policy | Multi-turn free-form search | Typed, bounded SearchScript execution |
| Gold/path check | Answer plus path-support scoring | Executed hidden proof with exact provenance |
| Failure credit | Scalar answer/path/process reward | Retrieval and reasoning residuals |
| Curriculum | Bell reward on Solver accuracy | Two-dimensional recovery/conditional-answer frontier |
| Memory | Successful trajectory KG write-back | Planned proof-aligned overlay; base KILT graph remains immutable |
| Framework | veRL/SGLang | ms-swift 3.10.3 only |

## Small-scale baseline protocol

Freeze the validation split before any policy update. Compare:

1. single BM25 retrieval plus answer generation;
2. hyperlink EvidenceFlow without RL;
3. Solver-only GRPO;
4. Solver GRPO plus Questioner REINFORCE++;
5. the same method with the dual-frontier reward replaced by scalar answer difficulty.

Report answer EM/F1, provenance R-precision/recall, KILT-EM/F1, full-proof recovery, and the
retrieval/reasoning residual distribution. A self-play round counts as an improvement only
when the frozen validation KILT-F1 or proof recovery improves; train challenge closure alone
is treated as memorization.

## Current smoke artifacts

The frozen 16-challenge audit was converted to direct RL rows under
`outputs/kilt-evidence-selfplay-0.6b/direct_rl_round_001`: 12 Solver train examples, four
fixed validation examples, and ten Questioner candidate pools. These rows contain no SFT
demonstrations. The original official Qwen3-0.6B checkpoint is available at
`outputs/selfplay-qwen3-0.6b-smoke/model`. It contains a standalone 596,049,920-parameter
`Qwen3ForCausalLM` with no PEFT configuration and predates the first project adapter and RL
run. This checkpoint, rather than any `sft-merged` directory, is the no-task-SFT starting
policy for the direct-RL smoke.

## First direct-RL smoke result

The no-SFT run in `direct_rl_updates_001` completed on ms-swift 3.10.3. Solver GRPO ran
12 optimizer steps and Questioner REINFORCE++ ran 10 steps. Eight of 12 Solver steps and
seven of 10 Questioner steps had non-zero task-reward variance; their maximum gradient
norms were 7.00 and 5.06, respectively. The final shared LoRA adapter is the Questioner
`checkpoint-10`, which was initialized from the Solver `checkpoint-12`.

The frozen four-example validation comparison is deliberately reported as a smoke result,
not as an accuracy claim. Before self-play, single BM25 retrieval obtained KILT-F1 0.00,
while hyperlink EvidenceFlow obtained 0.25. Loading the final self-play adapter left the
EvidenceFlow KILT-F1 at 0.25, with answer F1 0.5714 and provenance recall 0.625 in both
cases. Thus the retrieval method improves over the baseline on this tiny split, while one
22-step self-play round is neutral on held-out accuracy. Larger seed runs and a rollout
evaluator matched to the trained tool policy are required before claiming an RL gain.

## Role-separated frontier archive experiment

The second-round experiment separates the two policies: the Solver keeps its first-round
LoRA, while the Questioner is trained from the base model with REINFORCE++. Its 140 online
rollouts are parsed with the same strict proposal schema used by the reward worker. Within
each private bridge pool, the highest-reward legal proposal is promoted; pools without a
legal proposal fall back to their executed-proof frontier score. The resulting archive is
therefore policy-dependent but remains certified and replayable. It is not an SFT dataset.

On the 500-page graph, 22 of 140 online outputs were legal positive-reward proposals,
covering 15 of 34 pools. The other 19 pools used the certified fallback. There was no
train/validation task overlap, and the 16 validation rows were equivalent at the decoded-row
level to the frozen first-round validation set.

Two Solver-only GRPO variants were run from the same first-round Solver checkpoint:

| Policy | Train rows / unique tasks | Answer F1 | Provenance R-precision | Provenance recall | KILT-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base EvidenceFlow | 0 / 0 | 0.4554 | 0.4688 | 0.5000 | 0.0625 |
| Round-1 Solver | 48 / 48 | 0.3929 | 0.5000 | 0.5312 | 0.0625 |
| Frontier archive, checkpoint 48 | 48 / 34 | 0.4554 | 0.4688 | 0.4688 | 0.0625 |
| Coverage + frontier archive, checkpoint 60 | 64 / 48 | 0.3929 | 0.4688 | 0.5000 | 0.0625 |
| Coverage + frontier archive, checkpoint 64 | 64 / 48 | 0.3929 | 0.4688 | 0.4688 | 0.0625 |

The pure frontier update recovered one held-out answer relative to the round-1 Solver, but
lost one evidence-recall event and did not exceed the base policy or improve KILT-F1. The
64-row coverage mixture increased rollout-signal density but did not improve the frozen
metrics. Thus the archive mechanism is operational and produces a measurable reasoning
trade-off, while this one-seed smoke result does not yet support a general accuracy claim.
The next algorithmic change should make archive promotion Pareto-aware over answer and
provenance residuals instead of scaling epochs or model size first.

## Executable counterfactual proof self-play experiment

The ECP implementation adds three elements that are not present in the scalar frontier
baseline: proof-potential traces for action-level credit, certified one-edge
counterfactual proof pairs for retriever positives and hard negatives, and a staged reward
gate. ECP-v1 exposed a process-only shortcut: all 196 rollouts were answer-incorrect, yet
176 received positive reward by selecting bridge evidence. ECP-v2 gates proof credit by
answer F1 and removed most of that shortcut. ECP-v3 requires the retrieve/expand/select
protocol and exact executed-proof answer; a cold start produced zero positive rollouts,
whereas initializing v3 from the best v2 checkpoint produced three positive rollouts out
of 196. This is evidence for gate annealing as an exploration mechanism, not yet evidence
for a general KILT improvement.

The final evaluator executes the same Hermes tool semantics as ms-swift, including every
ordered tool call emitted in one assistant generation. On the frozen 16-example split,
both the base model and staged-v3 fail under the four-generation budget. With one shared
recovery turn added, the paired result is:

| Policy (five generations) | Completed | Answer EM | Answer F1 | Provenance R-precision | KILT-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B base | 6/16 | 0.1250 | 0.1458 | 0.1562 | 0.0000 |
| ECP staged-v3, checkpoint 48 | 13/16 | 0.3125 | 0.3802 | 0.3438 | 0.0000 |

Thus the small run improves protocol completion, answer accuracy, and provenance hit rate,
but does not yet recover both proof passages on a held-out example. The next controlled
change is to make `select_evidence` causally follow `expand_evidence` and train the
retriever on the exported counterfactual pairs; increasing epochs without increasing
strict-positive rollout density is not justified by this experiment.

## Open-source Search-R1 baseline

The controlled open-source baseline is
[Search-R1](https://github.com/PeterGriffinJin/Search-R1), whose published training signal
is final-answer normalized exact match without a format or process reward. This repository
implements that reward as `search_r1_em` inside the existing ms-swift 3.10.3 path. It is a
protocol-matched reproduction rather than an exact artifact reproduction: both sides use
Qwen3-0.6B, the local KILT hyperlink retriever, the GraphTask evidence tools, seed 71, 48
updates, four GRPO rollouts, learning rate 1e-6, and the same 48/16 train/test manifest.
Neither side uses SFT. Only the reward differs.

The frozen 16-example evaluation allows five policy generations for both systems because
the required retrieve/expand/select/answer sequence needs four actions plus one shared
recovery turn. The final checkpoint is used rather than selecting on this test set.

| 48-step policy | Completed | Answer EM | Answer F1 | Provenance R-precision | KILT-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Search-R1-EM, checkpoint 48 | 5/16 | 0.1250 | 0.1250 | 0.1250 | 0.0000 |
| ECP-v2, checkpoint 48 | 13/16 | **0.3125** | **0.3802** | **0.3438** | 0.0000 |

ECP-v2 has three answer-EM candidate-only examples and no Search-R1-only examples. The
exact two-sided McNemar p-value is 0.25, so this seed passes the small-run engineering gate
but is not a statistical generalization claim. Search-R1 obtained usable within-group
advantages in 14/48 training batches; the remaining 70.8% were zero-variance groups. ECP's
denser answer-gated evidence residuals therefore improve answer accuracy and protocol
completion under matched optimization compute. The KILT-F1 tie remains a failed criterion:
most completed trajectories select the bridge before expansion and never revise the evidence
set to contain both gold pages. A paper-level claim requires the causal-selection change,
multiple seeds, and a larger held-out KILT/HotpotQA-derived split.

Artifacts:

- Search-R1 rows: `outputs/kilt-evidence-selfplay-0.6b/search_r1_em_round_500_seed71/`
- Search-R1 update: `outputs/kilt-evidence-selfplay-0.6b/direct_rl_updates_500_seed71_search_r1_em/`
- Search-R1 test report: `outputs/kilt-evidence-selfplay-0.6b/eval_protocol_exact_search_r1_em_ckpt48_turn5_seed71.json`
- ECP-v2 test report: `outputs/kilt-evidence-selfplay-0.6b/eval_protocol_exact_ecp_v2_ckpt48_turn5_seed71.json`

## Causal counterfactual ECP-v4 result

ECP-v4 implements the controlled change identified above. A lightweight reranker is
learned from executed counterfactual proof pairs: the certified target passage is the
positive and proof-exclusive linked passages are hard negatives. The reranker is not an
LLM judge and does not use validation answers. At rollout time it reranks hyperlink
expansions using the learned query/passage/type token interactions. A causal certificate
then requires the final `select_evidence` action to occur after expansion and to include
observed passages from both retrieval stages. Answer and joint KILT credit remain separate
components, so retrieving only the easy bridge cannot satisfy the certificate.

On the frozen validation split, the learned reranker increased second-hop Recall@3 from
13/16 (0.8125) to 14/16 (0.8750) and MRR from 0.6875 to 0.8021. The direct GRPO run again
starts from the unadapted Qwen3-0.6B checkpoint, uses no SFT, and uses the same seed, 48
examples, 48 optimizer updates, four rollouts, and learning rate as Search-R1-EM. Of the
48 batches, 32 had non-zero within-group reward variance; two had a positive group-mean
reward and the maximum group mean was 0.3473. This confirms that the binary certificate is
learnable at this scale, although its positive events are sparse.

Five policy generations are insufficient for this policy: seven examples finish evidence
selection only on the fifth generation after two corrected early-answer attempts, leaving
no answer generation. Consequently, both systems were also evaluated under one shared
additional generation. This changes inference compute equally and does not change or
select a training checkpoint.

| 48-step policy (six generations) | Completed | Answer EM | Answer F1 | Provenance R-precision | KILT-EM | KILT-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Search-R1-EM, checkpoint 48 | 8/16 | **0.2500** | **0.2812** | 0.1875 | 0.0000 | 0.0000 |
| ECP-v4 + counterfactual reranker, checkpoint 48 | 7/16 | 0.0625 | 0.1302 | **0.3438** | **0.0625** | **0.1094** |

ECP-v4 is therefore the first matched run in this study to exceed the open-source baseline
on the primary KILT-F1 metric and to produce non-zero joint answer/evidence correctness.
It also improves provenance R-precision by 0.1563 absolute. The result is not uniformly
better: Search-R1 retains a 0.1510 Answer-F1 advantage. Pairwise, ECP-v4 has two KILT-F1
successes unique to the candidate and none unique to the baseline (two-sided exact McNemar
`p=0.50`); answer EM has one candidate-only and four baseline-only successes (`p=0.375`).
The split is intentionally small, so these are exploratory effect estimates rather than
statistical claims. The next experiment should anneal from proof-potential differences to
the binary causal certificate to recover answer accuracy without reopening bridge-only
reward hacking.

Artifacts:

- Counterfactual reranker: `outputs/kilt-evidence-selfplay-0.6b/ecp_counterfactual_seed71/reranker.json`
- ECP-v4 rows: `outputs/kilt-evidence-selfplay-0.6b/ecp_v4_round_500_seed71/`
- ECP-v4 update: `outputs/kilt-evidence-selfplay-0.6b/direct_rl_updates_500_seed71_ecp_v4/`
- Six-generation Search-R1 report: `outputs/kilt-evidence-selfplay-0.6b/eval_protocol_exact_search_r1_em_ckpt48_turn6_seed71.json`
- Six-generation ECP-v4 report: `outputs/kilt-evidence-selfplay-0.6b/eval_protocol_exact_ecp_v4_ckpt48_turn6_seed71.json`

## KQAPro curriculum transfer smoke: ECP-v5

The first KQAPro-to-KILT transfer experiment isolates an algorithmic mechanism rather
than importing extra supervision. Although the local workspace contains the full KQAPro
graph, training data, and 0.6B KQAPro checkpoints, ECP-v5 deliberately loads none of those
weights or examples. It starts from the same raw Qwen3-0.6B model as Search-R1 and transfers
only Curriculum-v3's staged execution principle.

ECP-v5 defines a monotone causal potential over three typed milestones:

1. `0.20`: `retrieve` recovers the certified bridge passage;
2. `0.35`: a later `expand` recovers the remaining proof passage(s);
3. `0.45`: a still-later `select_evidence` selects bridge and target evidence.

Only the final potential is rewarded, so repeating a tool call cannot accumulate credit.
Process shaping is capped at 0.15 of total reward. The remaining objective is 0.15 causal
proof certification, 0.40 exact answer, and 0.30 joint KILT-F1. Thus the curriculum supplies
early exploration signal while the strict ECP-v4 certificate remains the terminal target.

The run uses the same 48/16 manifest, seed 71, 48 GRPO updates, four rollouts, learning rate
1e-6, counterfactual reranker, and final-checkpoint evaluation as ECP-v4. It uses no SFT.
The curriculum materially changes optimization: positive-mean batches increase from 2/48
to 41/48, non-zero-variance batches increase from 32/48 to 35/48, and mean training reward
changes from -0.0214 to 0.0305.

| 48-step policy (six generations) | Completed | Answer EM | Answer F1 | Provenance R-precision | KILT-EM | KILT-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Search-R1-EM | 8/16 | **0.2500** | **0.2812** | 0.1875 | 0.0000 | 0.0000 |
| ECP-v4 | **7/16** | 0.0625 | **0.1302** | **0.3438** | 0.0625 | **0.1094** |
| ECP-v5 KQA curriculum transfer | 3/16 | 0.0625 | 0.1094 | 0.1875 | 0.0625 | **0.1094** |

ECP-v5 retains ECP-v4's two KILT-positive examples and still exceeds Search-R1 on KILT-F1,
but it does not improve over ECP-v4. It expands evidence on 12/16 examples and completes
selection on only 3/16, compared with 15/16 and 7/16 for v4. The fixed dense/terminal mixture
therefore improves exploration on the training distribution but weakens held-out protocol
completion. The simple transfer hypothesis is not accepted as an accuracy improvement.
A proper follow-up should use temporal annealing in separate phases—dense grounding first,
then remove its weight and optimize the terminal certificate—rather than mixing both rewards
at every update. KQAPro checkpoint initialization remains a separate, explicitly
compute-unmatched transfer ablation.

Artifacts:

- ECP-v5 config: `configs/training/evidence_selfplay_qwen3_0_6b_ecp_v5_solver.yaml`
- ECP-v5 rows: `outputs/kilt-evidence-selfplay-0.6b/ecp_v5_kqacurr_round_500_seed71/`
- ECP-v5 update: `outputs/kilt-evidence-selfplay-0.6b/direct_rl_updates_500_seed71_ecp_v5_kqacurr/`
- ECP-v5 report: `outputs/kilt-evidence-selfplay-0.6b/eval_protocol_exact_ecp_v5_kqacurr_ckpt48_turn6_seed71.json`

## Temporal certificate annealing smoke: 16 dense + 32 strict updates

The follow-up replaces ECP-v5's fixed reward mixture with a genuine time curriculum while
holding the total budget at 48 updates. A seed-71 permutation assigns 16 disjoint training
questions to the dense ECP-v5 potential, then 32 different questions to the strict ECP-v4
certificate. Stage two loads the stage-one LoRA adapter and removes the dense potential.
The model still starts from raw Qwen3-0.6B and uses no SFT, KQAPro examples, or KQAPro
checkpoint. The split manifest records all UIDs and verifies zero overlap.

The dense phase supplies useful group contrast, but the signal collapses after the switch:

| Phase | Updates | Positive-mean batches | Non-zero-variance batches | Mean reward |
| --- | ---: | ---: | ---: | ---: |
| ECP-v5 dense potential | 16 | 14/16 | 14/16 | 0.0400 |
| ECP-v4 strict certificate | 32 | 1/32 | 20/32 | -0.0218 |

Under the same frozen 16-example, six-generation evaluation protocol, the annealed policy
recovers one completion and 0.0625 provenance R-precision relative to fixed ECP-v5, but it
does not improve joint KILT accuracy or exceed ECP-v4.

| 48-step policy (six generations) | Completed | Answer EM | Answer F1 | Provenance R-precision | KILT-EM | KILT-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Search-R1-EM | **8/16** | **0.2500** | **0.2812** | 0.1875 | 0.0000 | 0.0000 |
| ECP-v4 | 7/16 | 0.0625 | **0.1302** | **0.3438** | 0.0625 | **0.1094** |
| ECP-v5 fixed mixture | 3/16 | 0.0625 | 0.1094 | 0.1875 | 0.0625 | **0.1094** |
| ECP 16/32 temporal annealing | 4/16 | 0.0625 | 0.1094 | 0.2500 | 0.0625 | **0.1094** |

This particular 16/32 schedule is therefore rejected as an improvement over ECP-v4. It
does preserve ECP-v4's advantage over Search-R1 on joint KILT-F1, but the first phase mostly
learns action progress rather than making terminal proof events less sparse. The abrupt
objective switch also restarts the optimizer and learning-rate schedule, so a matched
two-stage ECP-v4-to-ECP-v4 control is required before attributing the difference solely to
the curriculum. The stronger next hypothesis is coefficient annealing on one shuffled task
distribution, with a non-zero strict certificate weight from the first update and no
optimizer reset; scaling data or model size should follow only after that mechanism passes
a multi-seed small-scale test.

Artifacts:

- Split builder: `scripts/build_ecp_annealed_curriculum.py`
- Split manifest: `outputs/kilt-evidence-selfplay-0.6b/ecp_anneal_16_32_seed71/annealing_manifest.json`
- Stage configs: `configs/training/evidence_selfplay_qwen3_0_6b_ecp_anneal_dense.yaml` and `configs/training/evidence_selfplay_qwen3_0_6b_ecp_anneal_strict.yaml`
- Two-stage update: `outputs/kilt-evidence-selfplay-0.6b/direct_rl_updates_500_seed71_ecp_anneal_16_32/`
- Evaluation report: `outputs/kilt-evidence-selfplay-0.6b/eval_protocol_exact_ecp_anneal_16_32_ckpt48_turn6_seed71.json`
