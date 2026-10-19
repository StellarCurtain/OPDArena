# Training and checkpoint recovery

`--model` is the initial student, `--teacher` is a frozen teacher, `--data` is the
prepared training parquet, and `--output` is a new run directory. For multi-GPU
runs, download model weights first and pass their Hugging Face directories.

## Methods

| Method | Teacher | Default learning rate | Default responses per prompt |
| --- | --- | --- | --- |
| `sft` | Full solutions in the dataset | 1e-5 | Not applicable |
| `rlvr` | Mathematical verification rewards; GRPO | 1e-6 | 8 |
| `opd` | Explicit frozen `--teacher` | 1e-6 | 4 |
| `self-opd` | Frozen SFT/RLVR endpoint in `--teacher` | 1e-6 | 4 |
| `opsd` | Frozen Base model with a teacher-only reference solution | 1e-6 | 4 |

OPD uses student-selected Top-16 candidates with student-probability weighting.
The tokenizer compatibility check rejects incompatible token-ID assignments and
records excluded student-only tokens. Self-OPD starts a fresh student from Base;
the teacher is not updated during distillation. OPSD keeps the solution out of
the student's prompt and adds it only to the teacher's input.

```bash
python -m opdarena.train --method rlvr --model ./models/student-base \
  --data ./data/openr1/train.parquet --output ./outputs/rlvr --gpus 4

python -m opdarena.train --method opsd --model ./models/student-base \
  --teacher-format raw-boundary --data ./data/openr1/train.parquet \
  --output ./outputs/opsd --gpus 4
```

The three OPSD teacher formats are `raw`, `chat`, and `raw-boundary`. The default
`raw-boundary` appends `Answer to the original problem:` after the teacher-only
reference context, matching the accuracy comparison. Use `raw` for the paper's
plain raw/raw distribution comparison. Student prompts remain raw in every arm.

## Experimental controls

A normalized teacher increment uses
`softmax(z_student_Base + z_teacher_endpoint - z_teacher_Base)`:

```bash
python -m opdarena.train --method opd --model ./models/student-base \
  --teacher ./models/teacher-sft --teacher-base ./models/teacher-base \
  --data ./data/openr1/train.parquet --output ./outputs/teacher-increment --gpus 4
```

The source models and student must have compatible token IDs. The construction
uses the student's output support.

To remove only the first response-token feedback, add `--mask-first-token` to a
`self-opd` command. It keeps the original response mask and loss denominator.
Per-step diagnostics are saved in `first-token-feedback/`.

## Batches and seeds

The default global prompt batch is 64. SFT uses microbatches of one with gradient
accumulation `batch_size / gpus`; VERL keeps the global prompt and PPO minibatch
sizes unchanged when the GPU count changes. The batch size must divide evenly
across GPUs. Rollout tensor parallelism must also divide the GPU count.

SFT defaults to seed 42; rollout methods default to sampling seed 0. Set `--seed`
explicitly for other repetitions. On-policy data order remains fixed.
For a Self-OPD repetition, train the intended SFT/RLVR teacher first and then pass
its actual frozen endpoint to a fresh student run.

`--max-steps`, `--max-response-tokens`, and `--batch-size` are useful for smoke
checks but change the experiment. Every run writes `launch.json` with its resolved
arguments and command. `--dry-run` prints that configuration without starting
workers, preparing data, or downloading models; tokenizer validation occurs at
actual launch.

## Resume and export

The default `--resume auto` selects the highest complete checkpoint with all
model, optimizer, RNG, and data-progress state for the requested GPU count.
Incomplete newer saves are skipped. Use the same GPU count when resuming. An
explicit checkpoint path is also accepted. Existing training is never silently
restarted from Base: use a new output directory for a new run.

SFT checkpoints use `checkpoint-N`; VERL checkpoints use `checkpoints/global_step_N`.
SFT final outputs are Hugging Face models. Convert a VERL actor before evaluation:

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir ./outputs/rlvr/checkpoints/global_step_780/actor \
  --target_dir ./models/rlvr-step780
```

Only resume checkpoints that you created or trust. Full-state training checkpoints
contain framework serialization, not just safe model-weight tensors.
