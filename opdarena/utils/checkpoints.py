"""Select complete checkpoints, including every distributed rank."""

import json
import re
import zipfile
from pathlib import Path


def mark_training_started(output):
    path = Path(output) / "state/training-started.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"phase": "training"}) + "\n")


def torch_archive(path):
    """Check the container structure without unpickling checkpoint contents."""
    try:
        with zipfile.ZipFile(path) as archive:
            return any(name.endswith("/data.pkl") for name in archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False


def complete(folder, method, world_size=1):
    folder = Path(folder)
    if method == "sft":
        required = ["trainer_state.json", "scheduler.pt"]
        if (folder / "optimizer.bin").is_file():
            required += ["optimizer.bin", "pytorch_model_fsdp.bin"]
        else:
            required.append("optimizer.pt")
        required += (
            ["rng_state.pth"]
            if world_size == 1
            else [f"rng_state_{rank}.pth" for rank in range(world_size)]
        )
        if not all((folder / name).is_file() for name in required):
            return False
        for index in folder.glob("*.index.json"):
            try:
                shards = set(json.loads(index.read_text())["weight_map"].values())
            except (ValueError, KeyError):
                return False
            if not shards or not all((folder / shard).is_file() for shard in shards):
                return False
        return bool(
            list(folder.glob("model*.safetensors")) or list(folder.glob("pytorch_model*.bin"))
        )
    if not torch_archive(folder / "data.pt"):
        return False
    return all(
        torch_archive(folder / "actor" / f"{kind}_world_size_{world_size}_rank_{rank}.pt")
        for rank in range(world_size)
        for kind in ("model", "optim", "extra_state")
    )


def select(output, method, request, world_size=1):
    output = Path(output)
    if request not in ("auto", "none"):
        folder = Path(request).resolve()
        if not complete(folder, method, world_size):
            raise ValueError(f"Incomplete checkpoint for {world_size} rank(s): {folder}")
        return str(folder)
    root = output if method == "sft" else output / "checkpoints"
    pattern = r"checkpoint-(\d+)" if method == "sft" else r"global_step_(\d+)"
    found = sorted(
        (
            (int(match[1]), path)
            for path in root.glob("*")
            if path.is_dir() and (match := re.fullmatch(pattern, path.name))
        ),
        reverse=True,
    )
    if request == "auto":
        for _, folder in found:
            if complete(folder, method, world_size):
                return str(folder.resolve())
    if (
        found
        or (output / "metrics.jsonl").exists()
        or (output / "state/training-started.json").exists()
    ):
        raise ValueError(
            "No complete checkpoint matches this run and GPU count. Use a new output directory to restart."
        )
    return None


def validate_resume_arguments(output, arguments):
    """Prevent a resumed run from silently changing its training configuration."""
    previous = Path(output) / "launch.json"
    if not previous.is_file():
        return
    old = json.loads(previous.read_text())["arguments"]
    fields = (
        "method",
        "model",
        "teacher",
        "teacher_base",
        "teacher_format",
        "data",
        "batch_size",
        "lr",
        "seed",
        "responses",
        "attention",
        "max_prompt_tokens",
        "max_response_tokens",
        "cutoff_length",
        "warmup_ratio",
        "mask_first_token",
        "gpus",
        "tensor_parallel",
        "sft_sharding",
    )
    changed = [field for field in fields if old.get(field) != arguments.get(field)]
    if changed:
        raise ValueError(
            "Resuming would change training settings: "
            + ", ".join(changed)
            + ". Use a new output directory."
        )
