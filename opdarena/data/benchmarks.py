"""Download public benchmark inputs. No benchmark examples are distributed in the code package."""

import argparse
import hashlib
import json
import random
from pathlib import Path

UPSTREAM = "ac26e38d6f1572eb027597b48a9f4e01f6915ef8"
HASHES = {
    "AMC23": "0f933978194b73fc091bd667af7d9056d3d745c9d9e3577b5adcad77f8f1ae7c",
    "AIME24": "dc3ae15e0236828eb95033a68aab60e19da7ea1afdaad80e26924e5f82d3d96f",
    "AIME25": "14ee9a0e8d79cba76c62247bc5ebee1647e023eccf60e31994e279113777dad2",
}
SUFFIX = r" Please reason step by step, and put your final answer within \boxed{}."


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--tasks",
        nargs="+",
        choices=["AMC23", "AIME24", "AIME25", "HMMT25", "Math500", "NQ"],
        default=["AMC23", "AIME24", "AIME25", "HMMT25", "Math500", "NQ"],
    )
    args = p.parse_args()
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq
    import requests
    from datasets import load_dataset

    from opdarena.evaluation.nq_protocol import digest, prompt, validate_questions

    manifest = {}
    for task in args.tasks:
        out = args.output / task
        out.mkdir(parents=True, exist_ok=True)
        if task in HASHES:
            url = f"https://raw.githubusercontent.com/thunlp/OPD/{UPSTREAM}/datasets/test_data/{task}/test.parquet"
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            sha = hashlib.sha256(r.content).hexdigest()
            if sha != HASHES[task]:
                raise ValueError("Benchmark source changed: " + task)
            rows = pq.read_table(io.BytesIO(r.content)).to_pylist()
            if task == "AMC23":
                # The pinned upstream file combines83 questions; the paper uses its40 AMC2023 questions.
                rows = rows[43:83]
                for i, row in enumerate(rows):
                    row["id"] = str(i)
                    row["extra_info"] = {"index": f"AMC23-{i}", "split": "test"}
            assert len(rows) == (40 if task == "AMC23" else 30)
            pq.write_table(pa.Table.from_pylist(rows), out / "test.parquet")
            manifest[task] = {"url": url, "sha256": sha, "questions": len(rows)}
        elif task == "HMMT25":
            revision = "6fdc4277120810ff75aa22d2d5489b91f7a262a1"
            ds = load_dataset("MathArena/hmmt_feb_2025", revision=revision, split="train")
            rows = [
                {
                    "prompt": [{"role": "user", "content": str(r["problem"]).strip() + SUFFIX}],
                    "data_source": task,
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": str(r["answer"]).strip()},
                    "extra_info": {"index": f"HMMT25-{int(r['problem_idx'])}", "split": "test"},
                }
                for r in ds
            ]
            assert len(rows) == 30
            pq.write_table(pa.Table.from_pylist(rows), out / "test.parquet")
            manifest[task] = {
                "dataset": "MathArena/hmmt_feb_2025",
                "revision": revision,
                "questions": 30,
            }
        elif task == "Math500":
            revision = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
            ds = load_dataset("HuggingFaceH4/MATH-500", revision=revision, split="test")
            assert len(ds) == 500
            (out / "test.jsonl").write_text(
                "".join(json.dumps(dict(r), ensure_ascii=False) + "\n" for r in ds)
            )
            manifest[task] = {
                "dataset": "HuggingFaceH4/MATH-500",
                "revision": revision,
                "questions": 500,
            }
        else:
            revision = "5dd9790a83002ad084ddeb7c420dc716852c6f28"
            ds = load_dataset("google-research-datasets/nq_open", revision=revision)
            examples = random.Random(0).sample(list(ds["train"]), 5)
            rows = []
            for i, r in enumerate(ds["validation"]):
                row = {"index": i, "question": r["question"], "answers": r["answer"]}
                row["id"] = digest(row)
                row["prompt"] = prompt(row["question"], examples)
                rows.append(row)
            data = {"examples": examples, "questions": rows}
            validate_questions(data)
            (out / "questions.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2) + "\n"
            )
            manifest[task] = {
                "dataset": "google-research-datasets/nq_open",
                "revision": revision,
                "fingerprints": {k: v._fingerprint for k, v in ds.items()},
                "fewshot_seed": 0,
                "questions": 3610,
            }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared:", ", ".join(args.tasks))
    print("The other OOD tasks are downloaded by the pinned lm-evaluation-harness when evaluated.")


if __name__ == "__main__":
    main()
