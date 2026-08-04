# GraphTask-R1

GraphTask-R1 trains one shared LoRA policy as both a privileged graph Questioner and a
tool-limited Solver. Questioner proposals are accepted only after program execution, bounded
witness materialization, counterfactual necessity checks, shortcut detection, and evaluation by
a frozen Solver snapshot. Gold answers always come from the certified program.

The repository now provides:

- a typed core DSL with entity roots, bounded scans, hops, intersection/union, type and typed
  literal filters, and count;
- deterministic in-memory, indexed KQA Pro SQLite, and Freebase Virtuoso backends;
- KQA Pro KoPL conversion with answer reconciliation and structured rejection records;
- WebQSP, ComplexWebQuestions, and GrailQA normalization with held-out entity denylists;
- verl v0.7.1 multi-turn SFT/RL data exporters and Qwen3-4B-Instruct-2507 LoRA launchers;
- candidate-specific asynchronous frontier rewards from a frozen tool-using Solver;
- a resumable 4-GPU mixed-role self-play orchestrator and benchmark evaluator;
- CPU fixtures, replay tests, manifests, audit commands, linting, typing, and CI.

No GPU result is claimed in this repository. Training and Freebase ingestion are intentionally
left for the user's server because they require model weights, licensed datasets, large storage,
and 4×80GB GPUs.

## Local verification

```bash
python -m pip install -e '.[dev,training]'
make lint
make typecheck
make test
make e2e
make scripted-selfplay
```

## Data and training entry points

```bash
# KQA Pro cold-start data
graphtask-r1 data fetch --dataset kqapro
graphtask-r1 data prepare --dataset kqapro \
  --raw-dir data/raw/kqa_pro --output-dir data/processed/kqapro/kqapro-v1

# Freebase endpoint check and leakage-safe Questioner seeds
graphtask-r1 graph preflight --snapshot freebase-v1
graphtask-r1 data sample-seeds --snapshot freebase-v1 \
  --exclude data/processed/freebase_heldout_entities.json \
  --output data/verl/freebase_questioner_seeds.parquet

# Inspect the exact 4-GPU self-play launch without starting GPUs
graphtask-r1 train self-play --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay --dry-run
```

Read [Data preparation](docs/DATA_PREPARATION.md) before downloading anything and
[Training](docs/TRAINING.md) before starting verl. Research motivation and experiment design are
in [RESEARCH_AND_TRAINING_GUIDE.md](docs/RESEARCH_AND_TRAINING_GUIDE.md).
