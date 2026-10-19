"""Fingerprint checkpoints and benchmark inputs to identify evaluation results."""

import hashlib
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_signature(path):
    path = Path(path).resolve()
    if not (path / "config.json").is_file():
        raise ValueError("Expected a local Hugging Face model: " + str(path))
    weights = sorted(path.glob("*.safetensors")) or sorted(path.glob("pytorch_model*.bin"))
    if not weights:
        raise ValueError("No Hugging Face model weights: " + str(path))
    files = [path / "config.json", *weights]
    files += [
        p
        for p in [
            path / "tokenizer.json",
            path / "tokenizer_config.json",
            path / "generation_config.json",
        ]
        if p.is_file()
    ]
    return {"path": str(path), "files": {p.name: file_sha256(p) for p in files}}
