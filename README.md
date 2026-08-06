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

## GPU training environment

The following is the recommended NVIDIA/FSDP2 setup for this repository. GPU packages remain
external to GraphTask-R1: none of these commands modifies `requirements.txt`, and the verl source
checkout is kept outside this repository.

The pinned verl installer is the safest starting point because SGLang and vLLM have strict
PyTorch/CUDA compatibility requirements and may replace an existing PyTorch installation. Use a
fresh environment rather than running it inside another working training environment. See the
[official verl installation guide](https://verl.readthedocs.io/en/latest/start/install.html),
[PyTorch installation selector](https://pytorch.org/get-started/locally/), and
[SGLang installation guide](https://docs.sglang.io/docs/get-started/install) when adapting the
commands to different GPUs.

### 1. Check the NVIDIA host

Install the NVIDIA driver using the
[official driver guide](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/latest/),
then confirm that every training GPU is visible:

```bash
nvidia-smi
```

The verl installation guide currently recommends CUDA 12.8 or newer. The reproducible baseline
below uses CUDA 12.8 and Python 3.12; do not use it unchanged for AMD/ROCm or Ascend hardware.

### 2. Create an isolated Python environment

```bash
conda create -n graphtask python=3.12 -y
conda activate graphtask
python -m pip install --upgrade pip setuptools wheel
```

### 3. Install the pinned verl/SGLang stack outside GraphTask-R1

Set `GRAPHTASK_DEPS_ROOT` to an absolute directory on the server that is not inside this clone.
The pinned verl script installs its matching PyTorch, SGLang, vLLM, Ray, FlashAttention, and
FlashInfer dependencies. `USE_MEGATRON=0` avoids installing Megatron-specific packages because
the supplied launchers use FSDP2.

```bash
export GRAPHTASK_DEPS_ROOT=/absolute/path/to/external-dependencies
mkdir -p "$GRAPHTASK_DEPS_ROOT"
cd "$GRAPHTASK_DEPS_ROOT"

git clone https://github.com/verl-project/verl.git verl-v0.7.1
cd verl-v0.7.1
git checkout bec9ef74768dd201881cd4e54cd0385e87caae27

USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
python -m pip install --no-deps -e .
```

At the pinned commit, the installer is based on the Torch 2.8/CUDA 12 stack and explicitly pins
SGLang 0.5.2, vLLM 0.11.0, FlashAttention 2.8.1, and FlashInfer 0.3.1. Treat the script at that
commit as the source of truth. Do not substitute the newest SGLang or vLLM release without
revalidating verl's Hydra configuration, tool calling, LoRA updates, and rollout behavior.

If PyTorch must be installed manually, choose the wheel matching the server driver and the
remaining inference stack. For example, the official PyTorch 2.8 CUDA 12.8 wheel is:

```bash
python -m pip install \
  torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Do not run this manual command in addition to the pinned verl installer unless intentionally
repairing or rebuilding the environment: SGLang/vLLM installation may select a different Torch
build. For a fully custom stack, follow the exact
[installer at the pinned commit](https://github.com/verl-project/verl/blob/bec9ef74768dd201881cd4e54cd0385e87caae27/scripts/install_vllm_sglang_mcore.sh)
as a compatibility checklist.

### 4. Verify the environment and run GraphTask-R1

```bash
python -m pip check
python - <<'PY'
from importlib.metadata import version

import torch

for package in ("torch", "verl", "sglang", "vllm", "ray", "flash-attn"):
    print(f"{package}: {version(package)}")
print(f"torch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
assert torch.cuda.is_available(), "PyTorch cannot access CUDA"
PY

cd /absolute/path/to/GraphTask
python -m pip install -r requirements.txt  # skip when already satisfied
python -m graphtask_r1.cli --help
```

Before a full GPU job, first run the verl Qwen3 multi-turn example, then use GraphTask-R1's
`--dry-run` command shown below to inspect all paths and GPU allocations.

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

Long-running commands emit timestamped progress logs to stderr at `INFO` level while keeping the
final machine-readable result on stdout. Progress records include the operation, phase,
`completed`, `total`, percentage, elapsed time, and action-specific accepted/rejected counts.
Use the global option before the command group to change verbosity, for example
`python -m graphtask_r1.cli --log-level WARNING data prepare ...`.
`data prepare` can process independent records concurrently, but defaults to one worker because
multiple readers of the same SQLite graph are often slower. Benchmark before setting `--workers N`.
For KQA Pro, an existing `graph.sqlite` is reused when its source hash and converter metadata match;
pass `--rebuild-graph` only when a forced rebuild is required.

```bash
# KQA Pro cold-start data
python -m graphtask_r1.cli data fetch --dataset kqapro
python -m graphtask_r1.cli data prepare --dataset kqapro \
  --raw-dir data/raw/kqa_pro --output-dir data/processed/kqapro/kqapro-v1 --workers 1

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
