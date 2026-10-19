"""Download the paper's 49,993-example OpenR1 training subset."""

import argparse
import hashlib
import json
from pathlib import Path

REVISION = "0c48b17c6a4e9301144fbf3f3d0bb361c613e2e1"
SOURCE_SHA256 = "76f1b3d2531e4505bdce8484ad3ec00886fb81d3e6818fa396c9e4962b235d71"
EXCLUDED = {514, 944, 2403, 9801, 10117, 30082, 30636}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (a.output / "train.parquet").exists():
        raise FileExistsError(a.output / "train.parquet")
    import pyarrow as pa
    import pyarrow.parquet as pq
    import requests

    from opdarena.data.sft import prepare

    a.output.mkdir(parents=True, exist_ok=True)
    raw = a.output / "source.parquet"
    url = f"https://huggingface.co/datasets/SphereLab/PEFTArena-data/resolve/{REVISION}/data/openr1-50k/train.parquet"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with raw.open("wb") as f:
            for chunk in r.iter_content(4 * 1024 * 1024):
                f.write(chunk)
    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    if sha != SOURCE_SHA256:
        raise ValueError("Public training source checksum differs from the registered source")
    table = pq.read_table(raw)
    if len(table) != 50000:
        raise ValueError("Expected50000 source rows")
    table = table.take(
        pa.array([i for i in range(len(table)) if i not in EXCLUDED])
    ).replace_schema_metadata(None)
    target = a.output / "train.parquet"
    pq.write_table(table, target)
    prepare(target, a.output / "sft", 49993, None)
    (a.output / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "SphereLab/PEFTArena-data",
                "revision": REVISION,
                "source_sha256": sha,
                "excluded_original_row_indices": sorted(EXCLUDED),
                "rows": 49993,
                "parquet_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "note": "Examples and order preserved; Arrow metadata is regenerated for datasets3.6 compatibility.",
            },
            indent=2,
        )
        + "\n"
    )
    raw.unlink()
    print("Prepared49993 original-order training examples:", target)


if __name__ == "__main__":
    main()
