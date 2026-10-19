"""Interpolate/extrapolate HF weights and measure a same-architecture parameter trajectory."""

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path


class Weights:
    """Read safetensors one parameter at a time, including sharded checkpoints."""

    def __init__(self, path):
        from safetensors import safe_open

        self.path = Path(path)
        index = self.path / "model.safetensors.index.json"
        if index.is_file():
            self.files = json.loads(index.read_text())["weight_map"]
        else:
            file = self.path / "model.safetensors"
            if not file.is_file():
                raise ValueError("Export the model to Hugging Face safetensors format first")
            with safe_open(file, framework="pt", device="cpu") as handle:
                self.files = {key: file.name for key in handle.keys()}

    def get(self, key):
        from safetensors import safe_open

        with safe_open(self.path / self.files[key], framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)


def aligned(*readers):
    names = set(readers[0].files)
    if any(set(reader.files) != names for reader in readers[1:]):
        raise ValueError("Models must have identical parameter names and architecture")
    for name in sorted(names):
        tensors = [reader.get(name) for reader in readers]
        if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
            raise ValueError("Parameter shapes differ: " + name)
        yield name, tensors


def trajectory(base, endpoint, student):
    """Report Euclidean distances and projection onto the Base-to-endpoint direction."""
    teacher_sq = student_sq = difference_sq = dot = 0.0
    count = 0
    for _, (base_tensor, teacher_tensor, student_tensor) in aligned(base, endpoint, student):
        if not base_tensor.is_floating_point():
            continue
        teacher_delta = teacher_tensor.double() - base_tensor.double()
        student_delta = student_tensor.double() - base_tensor.double()
        teacher_sq += float(teacher_delta.square().sum())
        student_sq += float(student_delta.square().sum())
        difference_sq += float((student_delta - teacher_delta).square().sum())
        dot += float((teacher_delta * student_delta).sum())
        count += base_tensor.numel()
    if not teacher_sq:
        raise ValueError("The endpoint equals Base; its direction is undefined")
    alpha = dot / teacher_sq
    return {
        "parameters": count,
        "base_to_endpoint": math.sqrt(teacher_sq),
        "base_to_student": math.sqrt(student_sq),
        "student_to_endpoint": math.sqrt(difference_sq),
        "projection_alpha": alpha,
        "normalized_off_axis": math.sqrt(max(0, student_sq - dot * dot / teacher_sq) / teacher_sq),
        "cosine": dot / math.sqrt(teacher_sq * student_sq) if student_sq else None,
    }


def scale(base, endpoint, alpha, output):
    """Save Base + alpha * (endpoint - Base), retaining the Base tokenizer."""
    import torch
    from safetensors.torch import save_file

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Each output shard is bounded by one source shard, rather than the full model.
    if set(base.files) != set(endpoint.files):
        raise ValueError("Model parameter names differ")
    groups = {}
    for name, file in base.files.items():
        groups.setdefault(file, []).append(name)
    with tempfile.TemporaryDirectory(prefix="opdarena-weights-", dir=output.parent) as temporary:
        target = Path(temporary)
        weight_map = {}
        total_size = 0
        for index, names in enumerate(groups.values(), 1):
            shard = {}
            for name in names:
                a, b = base.get(name), endpoint.get(name)
                if a.shape != b.shape:
                    raise ValueError("Parameter shapes differ: " + name)
                if a.is_floating_point():
                    value = (a.float() + alpha * (b.float() - a.float())).to(a.dtype)
                elif torch.equal(a, b):
                    value = a
                else:
                    raise ValueError("Nonfloating model buffers differ: " + name)
                shard[name] = value.contiguous()
                total_size += value.numel() * value.element_size()
            filename = (
                "model.safetensors"
                if len(groups) == 1
                else f"model-{index:05d}-of-{len(groups):05d}.safetensors"
            )
            save_file(shard, target / filename, metadata={"format": "pt"})
            weight_map.update({name: filename for name in names})
        for source in base.path.iterdir():
            if (
                source.is_file()
                and source.suffix in (".json", ".txt", ".model", ".jinja", ".tiktoken")
                and not source.name.endswith(".index.json")
            ):
                shutil.copy2(source, target / source.name)
        if len(groups) > 1:
            (target / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2
                )
                + "\n"
            )
        (target / "weight_transform.json").write_text(
            json.dumps(
                {
                    "formula": "Base + alpha * (endpoint - Base)",
                    "alpha": alpha,
                    "base": str(base.path),
                    "endpoint": str(endpoint.path),
                },
                indent=2,
            )
            + "\n"
        )
        target.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    for name in ("scale", "measure"):
        command = sub.add_parser(name)
        command.add_argument("--base", type=Path, required=True)
        command.add_argument("--endpoint", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "scale":
            command.add_argument("--alpha", type=float, required=True)
        else:
            command.add_argument("--student", type=Path, required=True)
    args = parser.parse_args()
    base, endpoint = Weights(args.base), Weights(args.endpoint)
    if args.operation == "scale":
        scale(base, endpoint, args.alpha, args.output)
    else:
        result = trajectory(base, endpoint, Weights(args.student))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
