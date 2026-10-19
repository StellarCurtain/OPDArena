# Copyright 2026 OPD contributors
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
"""Keep AdamW state on CPU except during optimizer.step; do not move parameters/gradients."""

import time

import torch


def validate_step_offload(config, strategy):
    if not config.get("optimizer_offload_on_step", False):
        return
    if strategy != "fsdp" or not config.get("optimizer_offload", False):
        raise ValueError("optimizer_offload_on_step requires FSDP1 and optimizer_offload=True")
    if config.get("param_offload", False) or config.get("offload_policy", False):
        raise ValueError("optimizer_offload_on_step keeps actor parameters/gradients on GPU")


def validate_optimizer(optimizer):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("optimizer_offload_on_step currently supports torch.optim.AdamW only")
    if any(
        group.get("capturable", False) or group.get("fused", False) or group.get("differentiable", False)
        for group in optimizer.param_groups
    ):
        raise ValueError("Step offload requires non-capturable, non-fused, non-differentiable AdamW")


def optimizer_state_bytes(optimizer, device_type=None):
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor) and (device_type is None or value.device.type == device_type)
    )


def require_cpu_optimizer_state(optimizer):
    if any(
        value.device.type != "cpu"
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    ):
        raise RuntimeError("Optimizer state must remain on CPU throughout forward/backward")


def move_optimizer_state(optimizer, load):
    """Blocking copies make CPU state safe to checkpoint immediately after a step.

    The scalar step counter stays on CPU, matching non-capturable AdamW's normal
    initialization. Moment tensors retain their dtype/layout; parameters and
    gradients never move. Return transferred bytes and synchronized wall time.
    """
    start, moved = time.perf_counter(), 0
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            for key, value in optimizer.state.get(parameter, {}).items():
                if not isinstance(value, torch.Tensor):
                    continue
                destination = parameter.device if load and key != "step" else torch.device("cpu")
                if value.device != destination:
                    moved += value.numel() * value.element_size()
                    optimizer.state[parameter][key] = value.to(destination, non_blocking=False)
    return moved, time.perf_counter() - start


def offloaded_optimizer_step(optimizer):
    validate_optimizer(optimizer)
    require_cpu_optimizer_state(optimizer)
    h2d_bytes, h2d_seconds = 0, 0.0
    try:
        h2d_bytes, h2d_seconds = move_optimizer_state(optimizer, load=True)
        optimizer.step()
    finally:
        d2h_bytes, d2h_seconds = move_optimizer_state(optimizer, load=False)
    require_cpu_optimizer_state(optimizer)
    return {
        "actor/optimizer_h2d_seconds": h2d_seconds,
        "actor/optimizer_d2h_seconds": d2h_seconds,
        "actor/optimizer_h2d_bytes": float(h2d_bytes),
        "actor/optimizer_d2h_bytes": float(d2h_bytes),
        "actor/optimizer_cpu_state_bytes_after_step": float(optimizer_state_bytes(optimizer, "cpu")),
    }
