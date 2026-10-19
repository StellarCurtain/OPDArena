# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Optional registered-vocabulary bounds for aligned-token OPD."""


def registered_vocab_size(vocab, model_vocab_size):
    """Require contiguous registered IDs; never infer validity from padded rows."""
    ids = set(vocab.values())
    if not ids or any(type(token_id) is not int for token_id in ids):
        raise ValueError("Registered vocabulary must contain integer token IDs")
    size = len(ids)
    if min(ids) != 0 or max(ids) != size - 1 or size > model_vocab_size:
        raise ValueError("OPD vocabulary bounds require contiguous registered IDs within the model vocabulary")
    return size


def limit_logits_to_vocab(logits, valid_vocab_size):
    """Slice before normalization to keep entropy and backward free of 0 * -inf."""
    if valid_vocab_size is None:
        return logits
    if type(valid_vocab_size) is not int or not 0 < valid_vocab_size <= logits.shape[-1]:
        raise ValueError(f"Invalid OPD vocabulary bound {valid_vocab_size} for {logits.shape[-1]} logits")
    return logits[..., :valid_vocab_size]


def validate_token_ids(token_ids, valid_vocab_size, context):
    """Fail before an embedding/gather kernel can trigger a CUDA device assertion."""
    if valid_vocab_size is None or token_ids is None or token_ids.numel() == 0:
        return None
    low, high = int(token_ids.min().item()), int(token_ids.max().item())
    if low < 0 or high >= valid_vocab_size:
        raise ValueError(f"{context}: token ID range [{low}, {high}] exceeds registered range [0, {valid_vocab_size})")
    return low, high
