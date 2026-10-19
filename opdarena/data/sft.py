#!/usr/bin/env python3
"""Convert the OpenR1 parquet into LLaMA-Factory supervised examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

DATASET_NAME = "openr1_sft"
DATASET_FILE = "openr1_sft.jsonl"
DATASET_INFO_FILE = "dataset_info.json"
MANIFEST_FILE = "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonempty_text(value: Any, field: str, row_index: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"row {row_index}: {field} must be text")
    if not value.strip():
        raise ValueError(f"row {row_index}: {field} is empty")
    return value


def _single_user_prompt(value: Any, row_index: int) -> str:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"row {row_index}: prompt must contain exactly one message")

    message = value[0]
    if not isinstance(message, dict):
        raise ValueError(f"row {row_index}: prompt[0] must be a record")
    if message.get("role") != "user":
        raise ValueError(f"row {row_index}: prompt[0].role must be user")
    return _nonempty_text(message.get("content"), "prompt[0].content", row_index)


def openr1_sft_record(row: dict[str, Any], row_index: int) -> dict[str, Any]:
    prompt = _single_user_prompt(row.get("prompt"), row_index)
    extra_info = row.get("extra_info")
    if not isinstance(extra_info, dict):
        raise ValueError(f"row {row_index}: extra_info must be a record")

    solution = _nonempty_text(extra_info.get("answer"), "extra_info.answer", row_index)
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": solution},
        ],
        "source_index": row_index,
    }


def dataset_info() -> dict[str, Any]:
    return {
        DATASET_NAME: {
            "file_name": DATASET_FILE,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }


def prepare(
    input_path: Path, output_dir: Path, expected_rows: int | None, expected_sha256: str | None
) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    input_digest = sha256(input_path)
    if expected_sha256 is not None and input_digest != expected_sha256:
        raise ValueError(f"input SHA-256 is {input_digest}, expected {expected_sha256}")

    parquet = pq.ParquetFile(input_path)
    available_columns = set(parquet.schema_arrow.names)
    required_columns = {"prompt", "extra_info"}
    if not required_columns.issubset(available_columns):
        raise ValueError(f"missing parquet columns: {sorted(required_columns - available_columns)}")
    if expected_rows is not None and parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"input contains {parquet.metadata.num_rows} rows, expected {expected_rows}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / DATASET_FILE
    temporary_data_path = output_dir / f".{DATASET_FILE}.tmp-{os.getpid()}"

    row_count = 0
    try:
        with temporary_data_path.open("w", encoding="utf-8") as output:
            for batch in parquet.iter_batches(columns=["prompt", "extra_info"], batch_size=1024):
                for row in batch.to_pylist():
                    record = openr1_sft_record(row, row_count)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    row_count += 1
        os.replace(temporary_data_path, data_path)
    finally:
        temporary_data_path.unlink(missing_ok=True)

    info_path = output_dir / DATASET_INFO_FILE
    temporary_info_path = output_dir / f".{DATASET_INFO_FILE}.tmp-{os.getpid()}"
    temporary_info_path.write_text(json.dumps(dataset_info(), indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_info_path, info_path)

    manifest = {
        "schema_version": 2,
        "source": str(input_path),
        "source_sha256": input_digest,
        "rows": row_count,
        "dataset_name": DATASET_NAME,
        "data_file": DATASET_FILE,
        "prompt_field": "prompt[0].content",
        "solution_field": "extra_info.answer",
        "prompt_transform": "none",
    }
    manifest_path = output_dir / MANIFEST_FILE
    temporary_manifest_path = output_dir / f".{MANIFEST_FILE}.tmp-{os.getpid()}"
    temporary_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest_path, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()

    manifest = prepare(args.input, args.output_dir, args.expected_rows, args.expected_sha256)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
