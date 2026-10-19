#!/usr/bin/env python3
"""Fail-fast validation for parquet datasets consumed by RethinkingOPD RLVR/OPD."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_value(record: dict, dotted_path: str):
    value = record
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(dotted_path)
        value = value[key]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--profile", choices=("generic", "dapo", "openr1"), default="generic")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--max-empty-ground-truth", type=int, default=0)
    parser.add_argument("--privileged-context-path")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.path.is_file():
        raise FileNotFoundError(args.path)

    parquet = pq.ParquetFile(args.path)
    rows = parquet.metadata.num_rows
    names = set(parquet.schema_arrow.names)
    required = {"data_source", "prompt", "reward_model"}
    if missing := sorted(required - names):
        raise ValueError(f"missing required columns: {missing}")
    if args.expected_rows is not None and rows != args.expected_rows:
        raise ValueError(f"expected {args.expected_rows} rows, found {rows}")

    digest = sha256(args.path)
    if args.expected_sha256 is not None and digest != args.expected_sha256:
        raise ValueError(f"expected sha256 {args.expected_sha256}, found {digest}")

    columns = ["data_source", "prompt", "reward_model"]
    if args.privileged_context_path is not None:
        privileged_root = args.privileged_context_path.split(".", 1)[0]
        if privileged_root not in names:
            raise ValueError(f"missing privileged-context column: {privileged_root}")
        columns.append(privileged_root)
    table = pq.read_table(args.path, columns=columns)
    sources: set[str] = set()
    empty_ground_truth = []
    malformed_prompt = []
    styles: set[str] = set()
    invalid_privileged_context = []
    for index, row in enumerate(table.to_pylist()):
        source = row["data_source"]
        sources.add(str(source))

        prompt = row["prompt"]
        if (
            not isinstance(prompt, list)
            or len(prompt) != 1
            or prompt[0].get("role") != "user"
            or not isinstance(prompt[0].get("content"), str)
            or not prompt[0]["content"].strip()
        ):
            malformed_prompt.append(index)

        reward_model = row["reward_model"]
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        if ground_truth is None or not str(ground_truth).strip():
            empty_ground_truth.append(index)
        if isinstance(reward_model, dict) and reward_model.get("style") is not None:
            styles.add(str(reward_model["style"]))
        if args.privileged_context_path is not None:
            try:
                privileged_context = nested_value(row, args.privileged_context_path)
            except KeyError:
                invalid_privileged_context.append(index)
            else:
                if not isinstance(privileged_context, str) or not privileged_context.strip():
                    invalid_privileged_context.append(index)

    if malformed_prompt:
        raise ValueError(f"malformed single-user prompts at rows {malformed_prompt[:20]}")
    if len(empty_ground_truth) > args.max_empty_ground_truth:
        raise ValueError(
            f"found {len(empty_ground_truth)} empty reward_model.ground_truth values, "
            f"maximum is {args.max_empty_ground_truth}; first rows: {empty_ground_truth[:20]}"
        )
    if args.profile == "openr1" and sources != {"openr1"}:
        raise ValueError(f"OpenR1 profile requires data_source=openr1, found {sorted(sources)}")
    if args.profile == "dapo" and not all("dapo" in source.lower() for source in sources):
        raise ValueError(f"DAPO profile found unexpected data sources: {sorted(sources)}")
    if invalid_privileged_context:
        raise ValueError(
            f"found missing or empty privileged context at {args.privileged_context_path}: "
            f"rows {invalid_privileged_context[:20]}"
        )

    result = {
        "path": str(args.path),
        "profile": args.profile,
        "rows": rows,
        "sha256": digest,
        "data_sources": sorted(sources),
        "reward_styles": sorted(styles),
        "prompt_contract": "one non-empty user message; preserved verbatim",
        "ground_truth_contract": f"at most {args.max_empty_ground_truth} empty reward_model.ground_truth values",
        "empty_ground_truth_rows": len(empty_ground_truth),
        "privileged_context_path": args.privileged_context_path,
        "invalid_privileged_context_rows": len(invalid_privileged_context),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
