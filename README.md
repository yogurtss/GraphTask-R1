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

## Clone and run

GraphTask-R1 is a source repository rather than an installable Python package. The
`graphtask_r1/` package lives at the repository root, so after cloning you can run it directly;
do **not** run `pip install -e .` or install GraphTask-R1 into the environment.

Use Python 3.11 or newer. On a server where the Python dependencies, PyTorch, and verl are
already available, setup is only:

```bash
git clone <repository-url> GraphTask
cd GraphTask
python -m graphtask_r1.cli --help
```

For a new CPU/development environment, install the small project-level dependency set:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` deliberately excludes `torch`, `verl`, CUDA libraries, SGLang, and
FlashAttention. Install those separately using versions that match the server's CUDA stack;
cloning or updating this repository will therefore not replace a working GPU environment. The
training code is validated against verl v0.7.1 at commit
`bec9ef74768dd201881cd4e54cd0385e87caae27`.

The external verl checkout can live anywhere on the server. It does not need to be cloned into a
`third_party/` directory in this repository; it only needs to be importable from the active Python
environment.

All commands below assume the current directory is the repository root.

## Local verification

```bash
python -m pip install -r requirements-dev.txt
make lint
make typecheck
make test
make e2e
make scripted-selfplay
```

## Data and training entry points

```bash
# KQA Pro cold-start data
python -m graphtask_r1.cli data fetch --dataset kqapro
python -m graphtask_r1.cli data prepare --dataset kqapro \
  --raw-dir data/raw/kqa_pro --output-dir data/processed/kqapro/kqapro-v1

# Freebase endpoint check and leakage-safe Questioner seeds
python -m graphtask_r1.cli graph preflight --snapshot freebase-v1
python -m graphtask_r1.cli data sample-seeds --snapshot freebase-v1 \
  --exclude data/processed/freebase_heldout_entities.json \
  --output data/verl/freebase_questioner_seeds.parquet

# Inspect the exact 4-GPU self-play launch without starting GPUs
python -m graphtask_r1.cli train self-play --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay --dry-run
```

Read [Data preparation](docs/DATA_PREPARATION.md) before downloading anything and
[Training](docs/TRAINING.md) before starting verl. Research motivation and experiment design are
in [RESEARCH_AND_TRAINING_GUIDE.md](docs/RESEARCH_AND_TRAINING_GUIDE.md). The optional end-to-end
comparison between the existing multi-turn tool path and the restricted one-shot GraphScript path
is documented in [Interaction modes](docs/INTERACTION_MODES.md); the default remains tool use.
