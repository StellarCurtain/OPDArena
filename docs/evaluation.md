# Evaluation and analysis

## Accuracy

`python -m opdarena.evaluate` runs a selected set of tasks sequentially, using
vLLM tensor parallelism across `--gpus`. `--tasks` accepts:

- Mathematics: AMC23, AIME24, AIME25, HMMT25, Math500.
- General capabilities: HellaSwag, ARC-Challenge, MMLU, IFEval, TriviaQA, BBH, NQ.

Prepare public inputs with `opdarena.data.benchmarks`. The likelihood and other
harness tasks download their public data on first use. Task templates and subsets
follow the evaluation-harness revision pinned in `requirements.txt`; custom
TriviaQA and BBH adapters are included.

The four competition-math tasks use eight responses per question. Math500 uses
four responses per question by default; select eight with `--math500-samples 8`.
The selected sample count is recorded with the results. IFEval uses a
512-token generation budget. Accuracy is averaged equally across the five math
or seven general tasks only when every member of that group is present.

Each output directory contains an immutable protocol, dataset fingerprints,
checkpoint fingerprints, raw responses/token IDs, and task summaries in SQLite;
`metrics.json` is the readable aggregate. Reusing a directory with a different
model or protocol raises an error. Interrupted matching runs reuse complete
batches. Changes to the model, selected tasks, or sample budget need a new directory.

```bash
python -m opdarena.evaluate --model ./models/opd-step780 \
  --data-dir ./data/benchmarks --tasks Math500 --math500-samples 4 \
  --output ./outputs/math500 --gpus 1
```

## Fixed-prefix distributions

Export raw prompts from a completed evaluation, then generate continuations from
the student Base and its RLVR endpoint. Use the **same student tokenizer** for both
banks and reuse the merged bank for every checkpoint being compared.

```bash
python -m opdarena.analysis.prompts --results ./outputs/eval-base/outputs.sqlite \
  --output ./data/prompts.jsonl
python -m opdarena.analysis.prefixes --model ./models/student-base \
  --tokenizer ./models/student-base --prompts ./data/prompts.jsonl --source Base \
  --output ./data/base-prefixes.json
python -m opdarena.analysis.prefixes --model ./models/rlvr-step780 \
  --tokenizer ./models/student-base --prompts ./data/prompts.jsonl --source RLVR \
  --output ./data/rlvr-prefixes.json
python -m opdarena.analysis.merge_prefixes ./data/base-prefixes.json ./data/rlvr-prefixes.json \
  --output ./data/fixed-prefixes.json
python -m opdarena.analysis.distributions --model ./models/opd-step780 \
  --base ./models/student-base --teacher ./models/teacher-base \
  --bank ./data/fixed-prefixes.json --output ./outputs/distributions --device cuda
```

Prefix generation defaults to 16 prompts per task, greedy 128-token continuations,
and up to 16 positions per continuation. JS and both KL directions use natural
logarithms on aligned student token IDs. The analysis loads one model at a time;
its GPU mode uses one visible device, while `--device cpu` is also available.
Each reference reports dataset-balanced averages and per-prefix measurements.
The prediction cache is keyed by weights, tokenizer, bank, and device.

Add `--teacher-base` to distribution analysis to measure the same normalized
teacher increment used by the corresponding training control.

## Parameter scaling and geometry

Use compatible, same-architecture Hugging Face safetensors checkpoints:

```bash
python -m opdarena.analysis.weights scale --base ./models/student-base \
  --endpoint ./models/rlvr-step780 --alpha 0.5 --output ./models/half-rlvr
python -m opdarena.analysis.weights measure --base ./models/student-base \
  --endpoint ./models/rlvr-step780 --student ./models/self-opd-rlvr \
  --output ./outputs/geometry.json
```

Scaling constructs `Base + alpha * (endpoint - Base)` and supports interpolation
and extrapolation. Evaluate the resulting model with the same accuracy command.
Measurement reports Euclidean distances, cosine, the projection coefficient onto
the Base-to-endpoint direction, and the off-axis distance divided by that direction's
norm. The tools read one parameter at a time; scaling writes one source-sized shard
at a time. They do not overwrite an existing model directory.
