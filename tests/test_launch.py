import json
import os
import subprocess
import sys

import pytest


def preview(tmp_path, method, *extra):
    command = [
        sys.executable,
        "-m",
        "opdarena.train",
        "--method",
        method,
        "--model",
        "./models/base",
        "--data",
        "./data/train.parquet",
        "--output",
        str(tmp_path / "run"),
        "--dry-run",
        *extra,
    ]
    env = dict(os.environ)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return subprocess.run(command, env=env, text=True, capture_output=True)


def test_two_gpu_global_batch_and_local_ray(tmp_path):
    result = preview(tmp_path, "opd", "--teacher", "./models/teacher", "--gpus", "2")
    assert result.returncode == 0, result.stderr
    launch = json.loads(result.stdout)
    command = launch["command"]
    assert "trainer.n_gpus_per_node=2" in command
    assert "trainer.nnodes=1" in command
    assert "actor_rollout_ref.actor.fsdp_config.fsdp_size=2" in command
    assert '+ray_kwargs.ray_init.address="local"' in command
    assert launch["visible_gpus"] == "0,1"
    assert not (tmp_path / "run").exists()


def test_sft_batch_is_global(tmp_path):
    result = preview(tmp_path, "sft", "--gpus", "4", "--batch-size", "64")
    assert result.returncode == 0, result.stderr
    launch = json.loads(result.stdout)
    assert "--nproc-per-node=4" in launch["command"]
    assert launch["arguments"]["sft_config"]["gradient_accumulation_steps"] == 16
    assert launch["arguments"]["sft_config"]["fsdp"] == "full_shard auto_wrap"


@pytest.mark.parametrize(
    "extra",
    [
        ("--gpus", "3"),
        ("--gpus", "2", "--tensor-parallel", "3"),
        ("--gpus", "0"),
        ("--responses", "0"),
        ("--lr", "-1"),
    ],
)
def test_reject_invalid_launch(tmp_path, extra):
    assert preview(tmp_path, "rlvr", *extra).returncode != 0


def test_opsd_default_matches_answer_boundary(tmp_path):
    result = preview(tmp_path, "opsd")
    assert result.returncode == 0, result.stderr
    command = json.loads(result.stdout)["command"]
    assert 'data.custom_cls.name="AnswerBoundaryDataset"' in command
    assert any(
        x.startswith("+reward_model.model.apply_chat_template_kwargs.chat_template=")
        for x in command
    )


def test_removed_variant_is_not_public(tmp_path):
    assert preview(tmp_path, "delta-self-opd").returncode != 0
    assert preview(tmp_path, "opd", "--teacher", "./teacher", "--feedback", "mean").returncode != 0
