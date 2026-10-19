"""Prepare OpenR1-format training data for SFT and on-policy methods."""

import argparse
import hashlib
import json
from pathlib import Path


def field(row, name):
    value = row
    for part in name.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSON/JSONL or existing OpenR1 parquet")
    source.add_argument("--dataset", help="Public Hugging Face dataset ID")
    p.add_argument("--revision", help="Hugging Face dataset revision")
    p.add_argument("--subset", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--question-field", default="problem")
    p.add_argument(
        "--solution-field",
        default="generations.0",
        help="Full worked solution, not the final answer",
    )
    p.add_argument("--answer-field", default="answer")
    p.add_argument("--limit", type=int, help="Optional first N records, preserving input order")
    p.add_argument(
        "--append-boxed-instruction",
        action="store_true",
        help="Append only when your source problems lack it",
    )
    args = p.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq

    if args.input:
        if args.input.suffix == ".parquet":
            records = pq.read_table(args.input).to_pylist()
        elif args.input.suffix == ".jsonl":
            records = [
                json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
            ]
        else:
            records = json.loads(args.input.read_text())
        source_info = {"input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest()}
    else:
        from datasets import load_dataset

        data = load_dataset(args.dataset, args.subset, split=args.split, revision=args.revision)
        records = data.to_list()
        source_info = {
            "dataset": args.dataset,
            "subset": args.subset,
            "split": args.split,
            "revision": args.revision,
            "fingerprint": data._fingerprint,
        }
    if not isinstance(records, list):
        raise ValueError("Input must contain a list of records")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        records = records[: args.limit]
    prepared = []
    for i, row in enumerate(records):
        if isinstance(row.get("prompt"), list) and isinstance(row.get("extra_info"), dict):
            if len(row["prompt"]) != 1 or row["prompt"][0]["role"] != "user":
                raise ValueError("Expected one raw user question")
            question = row["prompt"][0]["content"]
            solution = row["extra_info"]["answer"]
            answer = row["reward_model"]["ground_truth"]
            if (
                not isinstance(question, str)
                or not question.strip()
                or not isinstance(solution, str)
                or not solution.strip()
            ):
                raise ValueError(f"Row {i}: missing question or full solution")
            prepared.append(row)
            continue
        else:
            question = field(row, args.question_field)
            solution = field(row, args.solution_field)
            answer = field(row, args.answer_field)
            if args.append_boxed_instruction:
                question += (
                    r" Please reason step by step, and put your final answer within \boxed{}."
                )
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(solution, str)
            or not solution.strip()
        ):
            raise ValueError(f"Row {i}: missing question or full solution")
        prepared.append(
            {
                "data_source": "openr1",
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "" if answer is None else str(answer),
                },
                "extra_info": {"index": i, "answer": solution},
            }
        )
    if not prepared:
        raise ValueError("No training examples")
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "train.parquet"
    if target.exists():
        raise FileExistsError(target)
    pq.write_table(pa.Table.from_pylist(prepared), target)
    from opdarena.data.sft import prepare

    prepare(target, args.output / "sft", None, None)
    source_info.update(
        rows=len(prepared),
        fields={
            "question": args.question_field,
            "solution": args.solution_field,
            "answer": args.answer_field,
        },
        append_boxed_instruction=args.append_boxed_instruction,
        parquet_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    )
    (args.output / "manifest.json").write_text(json.dumps(source_info, indent=2) + "\n")
    print(f"Prepared {len(prepared)} examples: {target}")


if __name__ == "__main__":
    main()
