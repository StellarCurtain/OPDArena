# OPDArena

**Understanding Learning and Forgetting in On-Policy Distillation**

OPDArena compares target-domain learning and general capability retention during
language-model post-training. This repository provides training, accuracy
evaluation, and distribution/parameter analysis for the methods in the paper.

- Full-parameter SFT, GRPO-based RLVR, OPD from larger teachers, Self-OPD, and OPSD.
- Single- and multi-GPU training, with separate training and evaluation commands.
- Five mathematics benchmarks and seven general-capability benchmarks, with per-task scores and saved model responses.
- Fixed-prefix JS/KL measurements, weight interpolation/extrapolation, parameter distances, and the paper's teacher-source and first-token controls.

## Installation

Use **Linux x86_64, Python 3.11, and an NVIDIA driver compatible with CUDA 12.8**.
The supported runtime is pinned in [requirements.txt](requirements.txt): PyTorch
2.8.0, vLLM 0.11.0, Transformers 4.56.1, and Ray 2.48.0. The training engines are
included under `third_party/` because their OPD and raw-input changes are required.

```bash
git clone https://github.com/StellarCurtain/OPDArena.git
cd OPDArena
python3.11 -m venv .venv
source .venv/bin/activate
bash scripts/install.sh
```

Run commands from the repository root. Use a fresh virtual environment. The
installer uses the published FlashAttention wheel for this platform; pass
`--skip-flash-attn` and use `--attention sdpa` if that wheel is unsuitable.
See [installation and hardware](docs/installation.md) for details.

## Quick start

### 1. Prepare data and models

```bash
python -m opdarena.data.download --output ./data/openr1
python -m opdarena.data.benchmarks --output ./data/benchmarks
hf download Qwen/Qwen3-1.7B-Base --local-dir ./models/student-base
hf download Qwen/Qwen3-4B-Base --local-dir ./models/teacher-base
```

The training downloader reproduces the paper's 49,993-example OpenR1 selection,
including the original example order. Benchmarks and weights are downloaded
from their public sources and retain their own licenses. Custom data can be
prepared with `python -m opdarena.data.prepare --help`.

### 2. Training

```bash
# SFT: train a teacher from the student Base.
python -m opdarena.train --method sft \
  --model ./models/student-base --data ./data/openr1/train.parquet \
  --output ./outputs/sft --gpus 1

# Self-OPD: a fresh Base student learns from the frozen SFT endpoint.
python -m opdarena.train --method self-opd \
  --model ./models/student-base --teacher ./outputs/sft \
  --data ./data/openr1/train.parquet --output ./outputs/self-opd-sft --gpus 1

# Train with a larger frozen teacher on four GPUs.
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m opdarena.train --method opd \
  --model ./models/student-base --teacher ./models/teacher-base \
  --data ./data/openr1/train.parquet --output ./outputs/opd --gpus 4
```

`--batch-size` is the **global prompt batch size**, independent of GPU count.
Multi-GPU SFT uses FSDP by default; on-policy methods use Ray workers with
FSDP. `--tensor-parallel` controls vLLM rollout sharding and must divide `--gpus`.
`--dry-run` prints the launch configuration without downloads or training.

These are full-parameter runs. Memory needs depend on model size, teacher size,
sequence length, and GPU count; the examples are not a promise that every pair
fits on one GPU. CPU offloading is enabled for VERL by default. See
[training and checkpoint recovery](docs/training.md) for RLVR, OPSD, ablations,
multi-GPU behavior, and short smoke runs.

### 3. Export and evaluate

SFT outputs are already in Hugging Face format. Export a VERL checkpoint first:

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir ./outputs/opd/checkpoints/global_step_780/actor \
  --target_dir ./models/opd-step780

python -m opdarena.evaluate --model ./models/opd-step780 \
  --data-dir ./data/benchmarks --output ./outputs/eval-opd --gpus 1
```

Replace `global_step_780` with a checkpoint that exists. The evaluation command
saves `metrics.json` and an immutable `outputs.sqlite` record of prompts,
responses, token IDs, and scores. It rejects reuse with a different model or
protocol. Use `--tasks Math500` to select a subset and `--gpus 2` for two-way
tensor parallelism. See [evaluation and analysis](docs/evaluation.md).

## Repository structure

```text
opdarena/
  train.py, evaluate.py    Training and evaluation commands
  data/                   Public data preparation and OPSD inputs
  evaluation/             Benchmark protocols, scoring, and task adapters
  analysis/               Fixed-prefix distributions and parameter geometry
  methods/                Teacher-source and first-token controls
  utils/                  Checkpoints, provenance, and result storage
third_party/
  verl/                   Patched Apache-2.0 VERL training engine
  llamafactory/           Patched Apache-2.0 SFT engine
scripts/                  Installation, source checks, and test fixtures
examples/                 Training examples
tests/                    CPU unit tests
```

Run artifacts belong in `data/`, `models/`, or `outputs/`; these directories are
ignored by Git.

## Paper coverage

The main comparisons use SFT, RLVR, OPD, Self-OPD, and OPSD. OPSD exposes all three
teacher input formats discussed in the appendix. Its default is raw input with
the explicit answer boundary used in the accuracy comparison; select `raw` for
the plain raw/raw distribution comparison. Self-OPD reuses a frozen SFT/RLVR
endpoint and starts a new student from the released Base.

The controls include the normalized teacher training increment and masking
the first response-token feedback. Analysis tools measure distribution changes
and parameter geometry from checkpoints and saved evaluation outputs.

## Development and validation

```bash
python -m pytest -q
ruff check opdarena tests scripts
ruff format --check opdarena tests scripts
python scripts/check_release.py
```

See [validation](docs/validation.md) for tested configurations and how to run
the checks, and [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Citation

If you use OPDArena in your research, please cite:

```bibtex
@misc{xu2026opdarena,
  title  = {{OPDArena}: Understanding Learning and Forgetting in On-Policy Distillation},
  author = {Xu, Yuanze and Ye, Tianzhu and Sun, Hui and
            Wang, Wenhui and Dong, Li and Wei, Furu},
  year   = {2026},
  url    = {https://github.com/StellarCurtain/OPDArena}
}
```

## Acknowledgments and license

This work builds on [Rethinking On-Policy Distillation](https://github.com/thunlp/OPD)
and its student-Top-16 integration, [VERL](https://github.com/verl-project/verl),
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), and the
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
We also acknowledge the authors of the public models and benchmarks used here.

OPDArena is distributed under [Apache-2.0](LICENSE). Original notices and licenses
are retained in the vendored engines. See [NOTICE](NOTICE) and
[THIRD_PARTY.md](THIRD_PARTY.md) for attribution and modifications. Model and
dataset licenses are separate from the code license. Please also cite the
underlying methods and tools as appropriate.
