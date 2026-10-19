"""Extract the evaluated raw prompts for a shared distribution-comparison bank."""

import argparse
import json
import re
import sqlite3
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    connection = sqlite3.connect("file:" + str(a.results.resolve()) + "?mode=ro", uri=True)
    entries = connection.execute(
        "SELECT key,value FROM records WHERE key LIKE ?", ("accuracy/checkpoint/%",)
    ).fetchall()
    connection.close()
    entries.sort(
        key=lambda x: (
            x[0].split("/")[2],
            int(m[1]) if (m := re.search(r"batch-(\d+)$", x[0])) else -1,
        )
    )
    seen = set()
    prompts = []
    for key, value in entries:
        data = json.loads(value)
        rows = (
            data
            if isinstance(data, list)
            else data.get("outputs", [])
            if isinstance(data, dict)
            else []
        )
        task = key.split("/")[2]
        for row in rows:
            text = row.get("prompt", row.get("context"))
            if text and (task, text) not in seen:
                seen.add((task, text))
                prompts.append({"task": task, "prompt": text})
    if not prompts:
        raise ValueError("No saved raw prompts")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in prompts))
    print("Exported", len(prompts), "prompts")


if __name__ == "__main__":
    main()
