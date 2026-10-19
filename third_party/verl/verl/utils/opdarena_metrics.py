# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Display-only metric aliases used by the OPD-Arena W&B schema."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import torch


METRIC_SCHEMA_VERSION = 2

_PREFIX_RENAMES = (
    ("perf/", "infra/perf/"),
    ("timing_s/", "infra/timing_s/"),
    ("timing_per_token_ms/", "infra/timing_per_token_ms/"),
    ("prompt_length/", "data/prompt_length/"),
    ("response/", "data/response/"),
    ("response_length/", "data/response_length/"),
    ("response_length_non_aborted/", "data/response_length_non_aborted/"),
    ("global_seqlen/", "data/global_seqlen/"),
    ("training/", "train/state/"),
)


def compute_entropy_metrics(entropys: torch.Tensor, response_mask: torch.Tensor) -> dict[str, float | int]:
    """Aggregate response-token entropy without changing the objective."""

    if entropys.shape != response_mask.shape:
        raise ValueError(
            f"Entropy and response mask shapes differ: {tuple(entropys.shape)} != {tuple(response_mask.shape)}"
        )
    valid_entropys = torch.masked_select(entropys, response_mask.bool()).float()
    if valid_entropys.numel() == 0:
        mean = std = float("nan")
    else:
        mean = valid_entropys.mean().detach().item()
        std = valid_entropys.std(unbiased=False).detach().item()
    return {
        "actor/entropy": mean,
        "actor/entropy_mean": mean,
        "actor/entropy_std": std,
        "actor/entropy_valid_tokens": valid_entropys.numel(),
    }


def normalize_rlvr_metrics(metrics: MutableMapping[str, Any]) -> None:
    """Apply OPD-Arena metric schema v2 and add optimization aliases."""

    normalized: dict[str, Any] = {}
    for key, value in metrics.items():
        target = next(
            (
                replacement + key[len(prefix) :]
                for prefix, replacement in _PREFIX_RENAMES
                if key.startswith(prefix)
            ),
            key,
        )
        if target in normalized:
            raise ValueError(f"Metric schema collision: {key!r} maps to existing key {target!r}")
        normalized[target] = value

    metrics.clear()
    metrics.update(normalized)
    aliases = {
        "optimization/loss": "actor/pg_loss",
        "optimization/entropy": "actor/entropy",
        "optimization/grad_norm": "actor/grad_norm",
    }
    for alias, source in aliases.items():
        if source in metrics:
            metrics[alias] = metrics[source]
