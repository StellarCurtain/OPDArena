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

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        # symmetric memory allreduce not work properly in spmd mode
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    # With an existing multi-node Ray cluster, its workers do not inherit
    # variables the training shell exported after `ray start`.
    forwarded = (
        "VERL_FILE_LOGGER_PATH", "VERL_FILE_LOGGER_MODE", "VERL_FILE_LOGGER_ROOT",
        "WANDB_RUN_ID", "WANDB_NAME", "WANDB_RESUME", "WANDB_DIR", "WANDB_MODE", "WANDB_ENTITY",
        "HF_HOME", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "HF_DATASETS_TRUST_REMOTE_CODE",
        "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE", "CODE_REVISION", "PYTHONPATH",
        "CUDA_LAUNCH_BLOCKING", "HYDRA_FULL_ERROR", "NCCL_TIMEOUT",
        "TORCH_DISTRIBUTED_DEBUG", "TORCH_NCCL_BLOCKING_WAIT",
    )
    for key in (*runtime_env["env_vars"], *forwarded):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"][key] = os.environ[key]
    return runtime_env


def task_runner_scheduling_options():
    return {}
