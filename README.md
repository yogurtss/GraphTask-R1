# GraphTask-R1

GraphTask-R1 is an offline-first research baseline for constructing, verifying, and replaying
knowledge-graph reasoning tasks. Every accepted task carries an executable program certificate;
gold answers are always obtained by executing that program.

The repository implements the complete ToyGraph milestone:

- typed Pydantic schemas and a JSON-serializable query DSL;
- deterministic in-memory execution and SPARQL compilation;
- program/graph interventions, necessity scoring, and bounded shortcut search;
- constrained seeded program sampling with structured rejection records;
- deterministic verbalization, task certificates, canonical Solver traces, and replay;
- a serializable tool environment and task archive;
- a CPU-friendly shared-parameter Questioner/Solver mini self-play harness with resumable rounds;
- Parquet artifacts, manifests, CLI commands, tests, linting, typing, and CI.

The mini self-play harness verifies orchestration and parameter-sharing semantics. It is not a
claim that a 3B model was trained. Production `verl`/LoRA and remote graph execution are explicit
extension boundaries and require external models, GPUs, and graph snapshots.

## Quick start

```bash
python -m pip install -e '.[dev,training]'
make lint
make typecheck
make test
make e2e
make selfplay
```

Without installation, prefix commands with `PYTHONPATH=src`.

## Main acceptance commands

```bash
python -m graphtask_r1.cli e2e mini-pipeline \
  --graph toy --num-programs 100 --seed 42 --output-dir outputs/e2e-mini

python -m graphtask_r1.cli train mini-self-play \
  --graph toy --model deterministic-shared-policy --shared-policy true \
  --rounds 3 --questioner-groups 16 --solver-episodes 64 \
  --seed 42 --output-dir outputs/mini-self-play
```

Generated manifests record the exact configuration, source revision, Python version, and lock
hash. See `GraphTask-R1_PROJECT_EXECUTION_PLAN.md` for the research roadmap.

