# Frontier v2 self-play A/B experiment

The original self-play path remains the default. Existing YAML files that omit
`selfplay_variant` resolve to `legacy` and retain mixed-role training, the additive
Questioner reward, stochastic frozen-opponent scoring, and immediate archive writes.

The opt-in config is `configs/training/selfplay_frontier_v2.yaml`. It changes four
coupled behaviors:

1. Train the Solver phase first and the Questioner phase second, with a separate
   REINFORCE++ normalization window for each role.
2. Use a frontier-gated Questioner reward. Certification remains a hard gate;
   novelty, necessity, alignment, and efficiency modulate rather than replace the
   difficulty signal.
3. Derive frozen-opponent sample seeds from the run seed, round, task signature,
   and sample index, and cache identical evaluations for the duration of the round.
4. Keep the persistent archive immutable during reward computation. Stage certified
   tasks per round, then admit them deterministically by difficulty and novelty.

The Solver reward, certified-program gold answers, graph snapshot, prompts, base/archive
sampling ratios, and model initialization are unchanged.

## Comparable runs

Use the same initial adapter, data, seed, and hardware, but distinct output directories:

```bash
python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay-legacy

python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay_frontier_v2.yaml \
  --output-dir outputs/selfplay-frontier-v2
```

Trainer evaluation stays disabled in both variants. Compare the generated round reports
using role-specific Questioner/Solver metrics rather than the mixed aggregate reward.
Production configs save every 20 optimizer steps and retain at most two checkpoints.
For frontier v2, every round additionally writes `logs/archive_admission.json`, and the
two optimizer phases write `logs/ms_swift_solver.log` and
`logs/ms_swift_questioner.log`.

## Lightweight smoke

The three-round single-GPU config uses the existing ToyGraph/Qwen3-0.6B fixtures:

```bash
python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay_qwen3_0_6b_frontier_v2_smoke.yaml \
  --output-dir outputs/selfplay-qwen3-0.6b-frontier-v2-smoke
```

Its archive thresholds are intentionally permissive because the deterministic local
opponent provides only one sample. This smoke validates isolation, phase checkpoint
handoff, reward execution, staged admission, and round-to-round resume; it is not a
quality comparison with the 4B experiment. Smoke configs save every step so their short
phases always produce an adapter for checkpoint handoff, while still retaining at most
two checkpoints.
