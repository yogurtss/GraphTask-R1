# GraphTask-R1

GraphTask-R1 trains one shared LoRA policy in two roles: a privileged graph Questioner that
constructs executable tasks and a tool-limited Solver that answers them. Every accepted task is
verified by executing its certified program; gold answers are never copied from model output.

The current default is **KQA Pro-only**. KQA Pro supplies the cold-start tasks, validation set,
SQLite graph, and self-play seeds. Freebase, WebQSP, CWQ, and GrailQA are optional future
extensions and are not required by the commands below.

## Continue from an existing processed KQA Pro dataset

Use this path when KQA Pro has already been processed. You do **not** need to run `data fetch`,
`data prepare`, or `--rebuild-graph` again.

The following files are the reusable, expensive processing results:

```text
data/processed/kqapro/kqapro-v1/
├── graph.sqlite
├── train/tasks.parquet
└── val/tasks.parquet
```

Check that they exist and point the graph backend at the existing SQLite file:

```bash
ls -lh \
  data/processed/kqapro/kqapro-v1/graph.sqlite \
  data/processed/kqapro/kqapro-v1/train/tasks.parquet \
  data/processed/kqapro/kqapro-v1/val/tasks.parquet

export GRAPHTASK_KQAPRO_DB=$PWD/data/processed/kqapro/kqapro-v1/graph.sqlite
```

### 1. Export training Parquet files from the existing tasks

This is the missing step when SFT reports that `kqapro_sft_train.parquet` does not exist. These
commands read the existing accepted tasks and `graph.sqlite`; they do not process KQA Pro again and
do not modify anything under `data/processed/`.

```bash
python -m graphtask_r1.cli data export-sft \
  --input data/processed/kqapro/kqapro-v1/train/tasks.parquet \
  --output data/verl/kqapro_sft_train.parquet \
  --roles both

python -m graphtask_r1.cli data export-sft \
  --input data/processed/kqapro/kqapro-v1/val/tasks.parquet \
  --output data/verl/kqapro_sft_val.parquet \
  --roles both

python -m graphtask_r1.cli data export-verl \
  --input data/processed/kqapro/kqapro-v1/train/tasks.parquet \
  --output data/verl/kqapro_solver_rl.parquet \
  --roles solver

python -m graphtask_r1.cli data export-verl \
  --input data/processed/kqapro/kqapro-v1/val/tasks.parquet \
  --output data/verl/kqapro_solver_rl_val.parquet \
  --roles solver
```

The resulting lightweight training files are:

```text
data/verl/
├── kqapro_sft_train.parquet
├── kqapro_sft_val.parquet
├── kqapro_solver_rl.parquet
└── kqapro_solver_rl_val.parquet
```

They can be regenerated from the accepted tasks without rebuilding `graph.sqlite`.

### 2. Run shared Questioner/Solver SFT

```bash
export SFT_TRAIN_DATA=$PWD/data/verl/kqapro_sft_train.parquet
export SFT_VAL_DATA=$PWD/data/verl/kqapro_sft_val.parquet
export SFT_OUTPUT_DIR=$PWD/outputs/sft-qwen3-4b

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft.yaml --dry-run

python -m graphtask_r1.cli train sft \
  --config configs/experiments/qwen3_4b_sft.yaml
```

### 3. Run Solver-only GRPO

Use the LoRA adapter emitted by SFT:

```bash
export SFT_ADAPTER=/absolute/path/to/sft/lora_adapter
export SOLVER_RL_TRAIN_DATA=$PWD/data/verl/kqapro_solver_rl.parquet
export SOLVER_RL_VAL_DATA=$PWD/data/verl/kqapro_solver_rl_val.parquet
export SOLVER_GRPO_OUTPUT_DIR=$PWD/outputs/solver-grpo

python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo.yaml --dry-run

python -m graphtask_r1.cli train solver-grpo \
  --config configs/experiments/qwen3_4b_solver_grpo.yaml
```

### 4. Run KQA Pro self-play

Export a small seed batch first. The current orchestrator consumes every seed row in every round,
so 256 seeds is the safe starting point.

```bash
python -m graphtask_r1.cli data sample-seeds \
  --snapshot kqapro-v1 \
  --count 256 --pool-limit 100000 --seed 42 \
  --output data/verl/kqapro_questioner_seeds.parquet

export INITIAL_ADAPTER=/absolute/path/to/solver_grpo_or_sft_adapter
export BASE_TASKS=$PWD/data/processed/kqapro/kqapro-v1/train/tasks.parquet
export VAL_DATA=$PWD/data/verl/kqapro_solver_rl_val.parquet
export QUESTIONER_SEEDS=$PWD/data/verl/kqapro_questioner_seeds.parquet

python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay-kqapro --dry-run

python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay-kqapro
```

Resume an interrupted run without changing its existing config or round directories:

```bash
python -m graphtask_r1.cli train self-play \
  --config configs/training/selfplay.yaml \
  --output-dir outputs/selfplay-kqapro --resume
```

## Process KQA Pro only when processed files are absent

This is a one-time path for a new server. Skip it when the three files listed at the start of this
README already exist.

```bash
python -m graphtask_r1.cli data fetch --dataset kqapro

python -m graphtask_r1.cli data prepare --dataset kqapro \
  --raw-dir data/raw/kqa_pro \
  --output-dir data/processed/kqapro/kqapro-v1 \
  --splits train,val --seed 42 --workers 1
```

Do not use `--rebuild-graph` for a normal continuation. The converter intentionally keeps only
supported, executable, verified tasks; rejected records remain in `rejections.parquet` and must not
be added to training. KQA Pro `test.json` is not used for training because it lacks the required
gold program and answer.

## Environment

- Python 3.11 or newer.
- Run the repository directly from its root; do not install it with `pip install -e .`.
- Install lightweight project dependencies with `python -m pip install -r requirements.txt`.
- Install PyTorch, CUDA, SGLang, FlashAttention, and verl separately for the server's GPU stack.
- The validated verl revision is `v0.7.1` at commit
  `bec9ef74768dd201881cd4e54cd0385e87caae27`.
- The default model is `Qwen/Qwen3-4B-Instruct-2507`, non-thinking mode, with shared LoRA rank 32
  and alpha 64.

Before a full job, verify that `torch`, `verl`, and `sglang` import correctly and always inspect the
corresponding `--dry-run` output.

## Development checks

```bash
python -m pip install -r requirements-dev.txt
make lint
make typecheck
make test
make e2e
make scripted-selfplay
```

## Further documentation

- [Training details](docs/TRAINING.md)
- [KQA Pro data preparation and audits](docs/DATA_PREPARATION.md)
- [Research and experiment design](docs/RESEARCH_AND_TRAINING_GUIDE.md)
- [Tool and GraphScript interaction modes](docs/INTERACTION_MODES.md)

Freebase ingestion and Freebase-backed benchmarks are optional; their setup remains documented for
future experiments but is not part of the current default workflow.
