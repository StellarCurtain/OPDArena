"""Optional per-response training seeds, independent of validation sampling."""

import hashlib
from copy import copy


def training_sampling_params(params, seed, step, replica, count):
    if seed is None:
        return params
    result = []
    for index in range(count):
        value = copy(params)
        payload = f"{seed}:{step}:{replica}:{index}".encode()
        value.seed = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
        result.append(value)
    return result


def validation_sampling_params(params, seeds):
    """Preserve per-question seeds when validation batches are padded or sharded."""
    result = []
    for seed in seeds:
        value = copy(params)
        value.seed = int(seed)
        if not 0 <= value.seed < 2**31:
            raise ValueError('Invalid per-question validation seed')
        result.append(value)
    return result
