"""Atomically save generated responses as JSONL."""

import tempfile
from pathlib import Path

from opdarena.evaluation.math500 import publish


def rollout_json_fields(scores, extra, responses, response_mask):
    """Reduce logging rewards per sequence and preserve unpadded response IDs."""
    import torch

    size = responses.shape[0]
    sequence_scores = scores.detach().reshape(size, -1).sum(-1).cpu().tolist()
    values = {}
    for key, value in extra.items():
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if key == "true_reward_score":
                value = value.reshape(size, -1).sum(-1)
            values[key] = value.cpu().tolist()
        else:
            values[key] = value.tolist() if hasattr(value, "tolist") else value
    values["response_ids"] = [
        ids[mask.bool()].detach().cpu().tolist()
        for ids, mask in zip(responses, response_mask, strict=True)
    ]
    lengths = response_mask.sum(-1).detach().cpu().tolist()
    return sequence_scores, values, lengths


def save_jsonl(destination, lines):
    with tempfile.TemporaryDirectory(prefix="opd-validation-") as directory:
        local = Path(directory) / "generations.jsonl"
        local.write_text("\n".join(lines) + "\n", encoding="utf-8")
        publish(local, destination)
