"""Math500 prompt protocol and atomic file output."""

import os
import shutil
import tempfile
from pathlib import Path

PROTOCOL = {
    "prompt_suffix": r" Please reason step by step, and put your final answer within \boxed{}.",
    "dataset": "HuggingFaceH4/MATH-500",
    "split": "test",
    "dataset_revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
}


def publish(source, destination):
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
