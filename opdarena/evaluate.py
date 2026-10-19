"""Evaluate a Hugging Face checkpoint and save responses and scores."""

import argparse
import json
import os
from pathlib import Path

TASKS = (
    "AMC23",
    "AIME24",
    "AIME25",
    "HMMT25",
    "Math500",
    "HellaSwag",
    "ARC-Challenge",
    "MMLU",
    "IFEval",
    "TriviaQA",
    "BBH",
    "NQ",
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Merged Hugging Face checkpoint directory")
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--gpu-memory", type=float, default=0.8)
    p.add_argument("--max-sequences", type=int, default=4)
    p.add_argument(
        "--gpus", type=int, default=1, help="Number of GPUs for tensor-parallel inference"
    )
    p.add_argument("--math500-samples", type=int, choices=[4, 8], default=4)
    args = p.parse_args()
    from opdarena.utils.devices import select_gpus

    os.environ["CUDA_VISIBLE_DEVICES"] = select_gpus(args.gpus)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    from opdarena.evaluation.accuracy import evaluate
    from opdarena.utils.results import Results

    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "model": str(Path(args.model).resolve()),
        "math_samples": 8,
        "math500_samples": args.math500_samples,
        "IFEval_tokens": 512,
        "tasks": args.tasks,
        "data_dir": str(args.data_dir.resolve()),
        "bbh_scoring": "bbh-conclusion-v1",
        "inference": {"gpus": args.gpus, "max_sequences": args.max_sequences, "seed": 0},
    }
    model_config = Path(args.model) / "config.json"
    if not model_config.is_file():
        raise ValueError("Pass a local Hugging Face checkpoint; export a VERL checkpoint first")
    from opdarena.utils.provenance import file_sha256, model_signature

    protocol["model_signature"] = model_signature(args.model)
    data_files = {}
    for task in args.tasks:
        file = (
            args.data_dir
            / task
            / (
                "questions.json"
                if task == "NQ"
                else "test.jsonl"
                if task == "Math500"
                else "test.parquet"
            )
        )
        if task in TASKS[:5] or task == "NQ":
            if not file.is_file():
                raise FileNotFoundError(file)
            data_files[task] = file_sha256(file)
    protocol["data_sha256"] = data_files
    result = Results(args.output / "outputs.sqlite")
    result.put("protocol", protocol, immutable=True)
    try:
        values = evaluate(
            args.model,
            args.model,
            result,
            "checkpoint",
            {
                "data_dir": str(args.data_dir),
                "nq_questions": str(args.data_dir / "NQ/questions.json"),
                "gpu_memory_utilization": args.gpu_memory,
                "max_sequences": args.max_sequences,
                "tensor_parallel_size": args.gpus,
            },
            include_tasks=args.tasks,
            ifeval_generation_tokens=512,
            math500_samples=args.math500_samples,
        )
        for label, tasks in [("math5", TASKS[:5]), ("ood7", TASKS[5:])]:
            if all(t in values for t in tasks):
                values[label] = {"accuracy": sum(values[t]["accuracy"] for t in tasks) / len(tasks)}
        (args.output / "metrics.json").write_text(json.dumps(values, indent=2) + "\n")
        print(json.dumps(values, indent=2))
    finally:
        result.close()


if __name__ == "__main__":
    main()
