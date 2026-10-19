"""Generate a shared prefix bank before comparing checkpoints."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument(
        "--tokenizer", help="Student Base tokenizer; required for all compared models to share IDs"
    )
    p.add_argument(
        "--prompts", type=Path, required=True, help="JSONL records with task and raw prompt"
    )
    p.add_argument("--source", required=True, help="For the paper use Base or RLVR@780")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prompts-per-task", type=int, default=16)
    p.add_argument(
        "--gpu-memory", type=float, default=0.75, help="Fraction of GPU memory reserved by vLLM"
    )
    p.add_argument("--gpus", type=int, default=1)
    args = p.parse_args()
    from opdarena.utils.devices import select_gpus

    os.environ["CUDA_VISIBLE_DEVICES"] = select_gpus(args.gpus)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    if args.prompts_per_task < 1:
        p.error("--prompts-per-task must be positive")
    if not 0 < args.gpu_memory < 1:
        p.error("--gpu-memory must be between 0 and 1")
    if args.output.exists():
        raise FileExistsError(args.output)
    import numpy as np
    from vllm import LLM, SamplingParams

    model = LLM(
        model=args.model,
        tokenizer=args.tokenizer or args.model,
        tensor_parallel_size=args.gpus,
        dtype="bfloat16",
        max_model_len=2048,
        gpu_memory_utilization=args.gpu_memory,
        enforce_eager=True,
        seed=0,
        generation_config="vllm",
    )
    try:
        tok = model.get_tokenizer()
        groups = {}
        for line in args.prompts.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            ids = tok.encode(r["prompt"], add_special_tokens=False)
            group = groups.setdefault(r["task"], [])
            if 0 < len(ids) <= 1024 and len(group) < args.prompts_per_task:
                group.append((r["prompt"], ids))
        if not groups or any(len(v) != args.prompts_per_task for v in groups.values()):
            raise ValueError("Not enough short prompts in every task")
        selected = [(task, text, ids) for task, v in groups.items() for text, ids in v]
        valid = sorted(set(tok.get_vocab().values()))
        generated = model.generate(
            [{"prompt_token_ids": ids} for _, _, ids in selected],
            SamplingParams(temperature=0, max_tokens=128, allowed_token_ids=valid),
            use_tqdm=False,
        )
        rows = []
        for (task, text, ids), out in zip(selected, generated, strict=True):
            answer = out.outputs[0]
            tokens = ids + list(answer.token_ids)
            start, last = len(ids) - 1, len(tokens) - 2
            if last < start:
                raise ValueError("Empty bank continuation")
            positions = np.unique(
                np.linspace(start, last, min(16, last - start + 1), dtype=int)
            ).tolist()
            rows.append(
                {
                    "task": task,
                    "prompt": text,
                    "prefix_source": args.source,
                    "input_ids": tokens[:-1],
                    "positions": positions,
                    "continuation": answer.text,
                    "continuation_ids": list(answer.token_ids),
                }
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "tokenizer_vocab_sha256": hashlib.sha256(
                        json.dumps(tok.get_vocab(), sort_keys=True).encode()
                    ).hexdigest(),
                    "rows": rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
    finally:
        model.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
