# Curriculum v3 self-evolution experiment

`curriculum_v3` is an opt-in path. It does not change `legacy` or `frontier_v2`, so all
three variants can start from the same SFT adapter and run in separate output directories.

## Why this path exists

The previous GraphScript Questioner emitted only code; the natural-language question was
created later by the deterministic verbalizer. In addition, full certification gated every
useful Questioner signal, and unconditional Solver exact match mixed interface failures with
semantic task difficulty.

Curriculum v3 changes the learning problem:

1. The Questioner emits `{"question": ..., "program": ...}`. Gold answers are still derived
   only by executing the program.
2. Construction, grounding, and frontier quality are separate reward stages. Certification
   adds quality credit but no longer erases earlier progress.
3. Opponent parsing/execution readiness gates the frontier contribution. Semantic difficulty
   is measured conditional on successful execution.
4. Questioner and Solver use separate LoRA adapters. Questioner updates cannot overwrite the
   Solver used as the next frozen opponent.
5. Each round updates Questioner first, admits usable generated tasks, rebuilds the Solver
   dataset, and then updates Solver. New tasks can therefore be consumed in the same round.
6. Solver sampling begins in the easiest structural quantile, expands toward the full task
   distribution, and retains an easy replay fraction.

The default three-round schedule is:

| Round | Questioner objective | Solver objective | Archive policy |
| --- | --- | --- | --- |
| 1 | question + code production | syntax / easy base programs | no generated admission |
| 2 | seed, relation, handle, execution and grounding | process + answer shaping | relaxed difficulty |
| 3 | conditional semantic frontier | answer F1 / exact match | configured frontier window |

The formal config uses one-shot GraphScript v0.3. Here, `program_parse_rate` and
`program_execution_rate` are the interface-health metrics; literal function-call counts do not
apply. The optional `tool` mode selects `graphtask_curriculum_solver`, which passes cumulative
`calls`, `valid_calls`, `invalid_calls`, `edge_visits`, and `new_visible_entities` into the Solver
reward. This avoids treating a useful retrieval trajectory and an invalid first call as the same
outcome. Invalid calls return an error observation so both the actor and frozen judge can recover
on a later turn. The same scheduler lets the Questioner explore and execute a candidate program;
only the Solver rollout fields enter the Solver process reward.

## Reward logic

Questioner milestones cover the output envelope, question/code presence, schema and valid-prefix
progress, seed/relation/handle/type validity, execution, non-empty/cardinality-valid answers,
question-program alignment, leak avoidance, and certification.
Incomplete GraphScript JSON also receives capped credit for an already emitted question, schema
keys, valid operator names, and valid prefix, so length truncation does not collapse every rollout
to one reward.

At the frontier stage, the effective adversarial weight is:

```text
configured_frontier_weight * opponent_parse_rate * opponent_execution_rate_given_parse
```

The frontier itself uses semantic success conditional on executable opponent rollouts. A weak
Solver that cannot yet produce a valid program therefore does not label a Questioner task as
semantically impossible.

Solver milestones cover syntax/code progress and, in tool mode, call validity, successful calls,
evidence progress and budget compliance before answer F1 and exact match dominate.

## Run the A/B experiment

Use identical data, initial adapter, seed, and hardware, but distinct output directories:

```bash
python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay_frontier_v2.yaml \
  --output-dir outputs/selfplay-frontier-v2

python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay_curriculum_v3.yaml \
  --output-dir outputs/selfplay-curriculum-v3
```

The bounded three-round ToyGraph smoke is:

```bash
python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay_qwen3_0_6b_curriculum_v3_smoke.yaml \
  --output-dir outputs/selfplay-qwen3-0.6b-curriculum-v3-smoke
```

Before spending on a full run, require all of the following from the smoke:

- non-zero within-group reward variance in each role;
- increasing Questioner `milestone_program_executable` and Solver execution/F1;
- at least one round-2 or round-3 generated task admitted and present in that same round's Solver
  dataset (round 1 intentionally trains only the output contract);
- separate Questioner and Solver adapter paths in every round manifest;
- for a tool-mode ablation, non-zero `valid_calls` and `new_visible_entities`.

Do not compare the old `validity` curve directly with the new milestone curves. Sparse component
rates are recomputed from per-sample records, with missing components counted as zero, so rejection
and milestone rates use the full role sample count as their denominator.
