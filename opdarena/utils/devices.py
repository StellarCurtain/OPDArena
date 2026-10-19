"""Select training and inference devices from CUDA_VISIBLE_DEVICES."""

import os


def select_gpus(count):
    if count < 1:
        raise ValueError("GPU count must be positive")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return ",".join(map(str, range(count)))
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if len(devices) < count or "-1" in devices or len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES does not expose enough distinct GPUs")
    return ",".join(devices[:count])
