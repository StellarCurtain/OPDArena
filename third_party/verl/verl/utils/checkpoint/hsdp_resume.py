# Copyright 2026 rethinking-opd contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resize HSDP replica groups while retaining the checkpoint's FSDP shards."""

import json
from pathlib import Path

from torch.distributed.tensor import DTensor, Replicate


def source_rank(model, directory, rank, world_size):
    saved = json.loads((Path(directory) / "fsdp_config.json").read_text())["world_size"]
    if saved == world_size:
        return saved, rank, False
    mesh = getattr(model, "_device_mesh", None)
    if mesh is None or mesh.ndim != 2 or tuple(mesh.mesh_dim_names) != ("ddp", "fsdp"):
        raise ValueError("Changing checkpoint world size requires an HSDP mesh")
    if saved % mesh.size(1) or world_size % mesh.size(1):
        raise ValueError("HSDP recovery must retain the original FSDP shard count")
    # Retain surviving replicas' optimizer/RNG shards when shrinking; cycle
    # through saved replicas when expanding. The loaded DTensors below verify
    # the actual shard count, including old checkpoints without mesh metadata.
    return saved, rank % saved, True


def replicate_model_state(state, mesh):
    converted = {}
    for name, value in state.items():
        if not isinstance(value, DTensor) or value.device_mesh.ndim not in (1, 2):
            raise ValueError(f"Expected an FSDP or HSDP DTensor checkpoint: {name}")
        old_mesh = value.device_mesh
        if old_mesh.ndim == 2 and (
            tuple(old_mesh.mesh_dim_names or ()) != ("ddp", "fsdp") or not isinstance(value.placements[0], Replicate)
        ):
            raise ValueError("Checkpoint does not use replicated HSDP groups")
        if old_mesh.size(old_mesh.ndim - 1) != mesh.size(1):
            raise ValueError("Checkpoint and target FSDP shard sizes differ")
        converted[name] = DTensor.from_local(
            value.to_local(),
            device_mesh=mesh,
            placements=(Replicate(), value.placements[-1]),
            run_check=False,
            shape=value.shape,
            stride=value.stride(),
        )
    return converted
