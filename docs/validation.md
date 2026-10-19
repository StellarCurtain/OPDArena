# Validation

Checked on 2026-10-16 in a fresh Python 3.11 environment with the pinned CUDA 12.8
runtime, using NVIDIA H100 PCIe GPUs on one machine. Synthetic inputs were created
with `scripts/make_tiny_fixture.py`; the tiny Qwen2 model has 99,008 parameters.

## Checks performed

| Area | Result |
| --- | --- |
| Installation | Requirements resolve; `pip check` passes. NumPy/PyArrow/pandas compatibility is pinned. FlashAttention wheel imports and training work. |
| CPU tests | 19 tests pass, including global batch arithmetic, invalid launch arguments, complete multi-rank checkpoint selection, resume configuration protection, immutable results, directional KL, first-token masking, and weight scaling/geometry. Tests also pass in an environment without vLLM or datasets installed. |
| SFT | Two-GPU FSDP runs two steps, saves model/optimizer/RNG state, and resumes from checkpoint 2 to step 3. |
| OPD | Two-GPU FSDP runs two steps with nonzero gradients, saves all rank shards, and resumes from step 2 to step 3. A separate two-GPU run with rollout tensor parallelism 2 also completes with a nonzero gradient. |
| RLVR | Two-GPU GRPO runs two steps; the second step has a nonzero verification reward and gradient. |
| OPSD | Two-GPU raw-plus-answer-boundary training completes two steps with nonzero gradients. |
| First-token control | Single-GPU Self-OPD from the trained SFT teacher completes two steps; eight first-token feedback positions are zeroed per step and gradients remain nonzero. |
| Teacher increment | Single-GPU OPD with the normalized teacher-increment construction completes two steps. |
| Checkpoint export | A two-rank VERL actor is merged into a Hugging Face model and used by evaluation and analysis. |
| Accuracy | Two-GPU tensor-parallel inference generates and grades eight responses for two synthetic Math500-format questions. SQLite integrity passes. |
| Fixed-prefix analysis | Raw prompts are exported; two single-GPU prefix banks are generated and merged. GPU JS and both KL directions complete for Base and teacher references on all four entries. |
| Parameter analysis | Scaling produces a loadable Hugging Face checkpoint; the measured projection of a half-step model is approximately 0.5. |
| Packaging and source | Wheel build, documented CLI help, shell syntax, lint/format, English-comment checks, and hardcoded-path/credential-pattern checks pass. |

These checks validate execution paths and numerical invariants. Full-size models,
complete training schedules, all twelve benchmark datasets, the two other OPSD
input-format arms, arbitrary GPU memory configurations, and other GPU architectures
are outside the scope of these integration checks. The prefix smoke check uses two prompts per
source; the documented analysis default is sixteen per task.

## Repeating a short GPU check

```bash
python scripts/make_tiny_fixture.py --output ./data/tiny
CUDA_VISIBLE_DEVICES=0,1 python -m opdarena.train --method opd \
  --model ./data/tiny/base --teacher ./data/tiny/teacher \
  --data ./data/tiny/train.parquet --output ./outputs/smoke-opd \
  --gpus 2 --batch-size 4 --responses 2 --max-steps 2 --save-every 1 \
  --max-prompt-tokens 64 --max-response-tokens 16 --no-cpu-offload --rollout-memory 0.2
```

Repeat the same command with `--max-steps 3` to exercise automatic recovery from
step 2. Use a new output directory for each different method or control. For SFT,
use `--method sft`, omit teacher/rollout controls, and set `--cutoff-length 128`.
`--max-steps` may increase on resume; changes to the model, teacher, seed, batch,
or other scientific settings are rejected.

Smoke scores from synthetic questions are not paper benchmark results. Generated
inputs, outputs, logs, and build artifacts are excluded from the repository.
