# Technical Evidence and Inventor-Confirmation Items

A support state only indicates whether the current repository contains technical evidence for a feature. It does not establish novelty, inventiveness, ownership, or legal sufficiency.

## Stable evidence identifiers

| ID | Repository evidence | Supported technical content | State |
|---|---|---|---|
| C001 | README.md; docs/INTERACTION_MODES.md | One-shot GraphScript v0.3 is the principal mode; gold originates from certified execution | explicit |
| C002 | graphtask_r1/training/prompts.py | Joint question/program output; bounded graph program; operator and relation constraints | explicit |
| C003 | graphtask_r1/graphscript/schema.py | Versioned schema, typed operators, SSA handles, terminal emit, and limit validation | explicit |
| C004 | graphtask_r1/graphscript/executor.py | Relation whitelist, edge/return budgets, execution state, and evidence trace | explicit |
| C005 | graphtask_r1/generation/certify.py | Proposal validation, execution-derived gold, signature, witnesses, and certificate | explicit |
| C006 | graphtask_r1/rewards/curriculum.py; rewards/frontier.py | Staged dense reward, interface readiness, conditional semantic frontier | explicit |
| C007 | graphtask_r1/training/selfplay.py | Three curriculum phases, separate adapters, Questioner → archive → Solver order | explicit |
| C008 | graphtask_r1/archive/admission.py; archive/store.py | Deterministic pass-rate/novelty admission and structured rejection | explicit |
| C009 | configs/training/selfplay_curriculum_v3.yaml | Current Curriculum v3 budgets, stages, archive, sampling, and training settings | explicit |
| C010 | tests/ | Unit and integration evidence for implemented behaviors | explicit |

## Source-supported technical chains

| Technical chain | Evidence | Assessment |
|---|---|---|
| Joint question/program → bounded execution → execution-derived gold → certified task | C001–C005 | Strongest and most stable foundation |
| Interface readiness → gated conditional semantic frontier → staged reward | C006 | Distinctive when tied to the executable interface |
| Separate adapters → Questioner update → admission → same-round Solver update | C007–C009 | Strong combined self-evolution mechanism |
| Signature/text novelty/difficulty window → deterministic archive | C005, C008 | Suitable for dependent and state-artifact protection |
| Bounded budgets + trace + structured error | C003, C004 | Suitable for safety, replayability, and detectability |

## Feature-support ledger

| Candidate feature | State | Technical role | Drafting treatment |
|---|---|---|---|
| Typed executable graph DSL | explicit | Common model–graph interface | Core |
| Joint question and program output | explicit | Linked task proposal | Core |
| Gold only from certified execution | explicit | Answer authority | Core |
| Relation catalogue and execution budgets | explicit | Scope and resource control | Core or strong dependent |
| Interface readiness separated from semantics | explicit | Prevents reward distortion | Strong dependent or combined core |
| Frontier weight gated by readiness | explicit | Controls semantic signal | Strong dependent |
| Separate role adapters | explicit | Prevents role interference | Important dependent |
| Deterministic archive admission | explicit | Controls persistent task quality | Important dependent |
| Same-round consumption of new tasks | explicit | Shortens generation-to-learning loop | Important dependent |
| Replayable graph/seed/data/adapter state | explicit | Recovery and audit | System/state dependent |
| Accuracy, latency, or cost superiority | needs-confirmation | Potential experimental effect | Do not claim without data |
| Production external-graph deployment | needs-confirmation | Industrial embodiment | Present as applicability unless evidenced |

## Inventor-confirmation questions

1. [TO CONFIRM] Inventors, applicant/assignee, completion dates, and ownership history.
2. [TO CONFIRM] Earliest public disclosure, including papers, repositories, demos, or presentations.
3. [TO CONFIRM] Unpublished comparisons with prompt/tool baselines for accuracy, latency, cost, task validity, or training stability.
4. [TO CONFIRM] Whether protection should cover training only or also inference, task-data services, and archive-management services.
5. [TO CONFIRM] Minimum necessary functional operator classes; which operators are merely KQAPro examples.
6. [TO CONFIRM] Whether interface readiness must be a product or may use another monotonic combination.
7. [TO CONFIRM] Whether frontier reward must be Gaussian/target-centred or may use another proximity function.
8. [TO CONFIRM] Whether novelty may use alternative textual or semantic similarity measures.
9. [TO CONFIRM] Whether same-round consumption is mandatory or adjacent sub-round/micro-batch updates should be covered.
10. [TO CONFIRM] Whether a production-scale external graph backend has been tested; ToyGraph smoke tests are not production validation.

## Current drafting-risk assessment

- Executable program generation for KGQA is a known direction and should not be claimed in isolation.
- Questioner/Solver self-play and persistent memory are also known at a high level.
- The stronger technical story is the dependency chain: execution-certified gold, interface-gated frontier, role-separated adapters, deterministic admission, and same-round rebuilding.
- Current numerical settings are implementation examples, not automatically essential limitations.
- Formal claims require a dedicated prior-art search and qualified patent-professional review.
