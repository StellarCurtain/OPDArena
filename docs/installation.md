# Installation and hardware

The supported environment is Linux x86_64, Python 3.11, CUDA 12.8, PyTorch 2.8.0,
vLLM 0.11.0, Transformers 4.56.1, Ray 2.48.0, and FlashAttention 2.8.3.
`requirements.txt` pins the main runtime and the exact evaluation-harness commit.
NumPy 1.26.4, PyArrow 19.0.1, and pandas 2.2.3 are pinned together for data compatibility.
The resolved dependency versions are in `requirements/constraints-cu128.txt` and
are applied by the requirements file.

Create a fresh virtual environment, activate it, and run `bash scripts/install.sh`
from the repository root. The script installs the GPU runtime, the included
training engines, and `opdarena`. The engines and project package use editable
installs, so keep the checkout available and run commands from its root.

`--skip-flash-attn` skips the platform-specific FlashAttention wheel. Use
`--attention sdpa` when launching training in that environment. Other platforms,
CUDA versions, and dependency versions require their own validation.

## GPU configuration

Select devices with `CUDA_VISIBLE_DEVICES` and pass their count as `--gpus`.
For example, `CUDA_VISIBLE_DEVICES=2,3` with `--gpus 2` uses those two devices.
With no visibility variable, the launcher selects the first requested devices.
Training runs on a single machine: SFT uses a standalone torchrun process group,
and VERL starts Ray workers on that machine.

SFT uses full-parameter FSDP when more than one GPU is requested; `--sft-sharding ddp`
uses replicated training instead. OPD shards the student and frozen teacher with
FSDP. `--tensor-parallel` controls the rollout engine and must divide the GPU count.
Evaluation uses all requested GPUs as one tensor-parallel inference engine.

GPU and host-memory requirements grow with model size, response length, rollout
count, and teacher size. The default CPU offloading flag applies to VERL, not SFT.
Use `--no-cpu-offload` when host/GPU memory allows. A low-memory card is not guaranteed
to fit full-parameter training even with microbatches of one.

For a small installation check, use the synthetic fixture workflow in
[validation.md](validation.md). It downloads no model weights or training data.
