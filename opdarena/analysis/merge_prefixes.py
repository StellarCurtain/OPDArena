"""Merge Base and RLVR-generated prefixes without changing their token IDs."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("banks", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    documents = [json.loads(p.read_text()) for p in a.banks]
    vocab = {d["tokenizer_vocab_sha256"] for d in documents}
    if len(vocab) != 1:
        raise ValueError("Prefix banks use different token IDs")
    rows = [r for d in documents for r in d["rows"]]
    keys = [(r["task"], r["prefix_source"], r["prompt"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate prefix-bank entries")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(
            {"tokenizer_vocab_sha256": vocab.pop(), "rows": rows}, ensure_ascii=False, indent=2
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
