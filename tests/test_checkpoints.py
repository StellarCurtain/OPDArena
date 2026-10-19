import json
import zipfile

import pytest

from opdarena.utils.checkpoints import complete, select


def archive(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("archive/data.pkl", b"fixture")


def checkpoint(folder, size=2):
    archive(folder / "data.pt")
    for rank in range(size):
        for kind in ("model", "optim", "extra_state"):
            archive(folder / "actor" / f"{kind}_world_size_{size}_rank_{rank}.pt")


def test_partial_newer_checkpoint_is_skipped(tmp_path):
    older = tmp_path / "checkpoints/global_step_20"
    newer = tmp_path / "checkpoints/global_step_40"
    checkpoint(older)
    checkpoint(newer)
    (newer / "actor/optim_world_size_2_rank_1.pt").unlink()
    assert complete(older, "opd", 2)
    assert not complete(newer, "opd", 2)
    assert select(tmp_path, "opd", "auto", 2) == str(older)
    with pytest.raises(ValueError):
        select(tmp_path, "opd", str(newer), 2)
    with pytest.raises(ValueError):
        select(tmp_path, "opd", "auto", 1)


def test_restart_never_overwrites_existing_training(tmp_path):
    checkpoint(tmp_path / "checkpoints/global_step_20")
    with pytest.raises(ValueError):
        select(tmp_path, "opd", "none", 2)


def test_sft_requires_each_rng_and_weight_shard(tmp_path):
    for name in [
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state_0.pth",
        "model-1.safetensors",
    ]:
        (tmp_path / name).touch()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"one": "model-1.safetensors", "two": "model-2.safetensors"}})
    )
    assert not complete(tmp_path, "sft", 2)
    (tmp_path / "rng_state_1.pth").touch()
    assert not complete(tmp_path, "sft", 2)
    (tmp_path / "model-2.safetensors").touch()
    assert complete(tmp_path, "sft", 2)


def test_fsdp_sft_optimizer_binary_requires_fsdp_model(tmp_path):
    for name in [
        "trainer_state.json",
        "optimizer.bin",
        "scheduler.pt",
        "rng_state_0.pth",
        "rng_state_1.pth",
        "model.safetensors",
    ]:
        (tmp_path / name).touch()
    assert not complete(tmp_path, "sft", 2)
    (tmp_path / "pytorch_model_fsdp.bin").touch()
    assert complete(tmp_path, "sft", 2)


def test_resume_refuses_teacher_or_seed_change(tmp_path):
    from opdarena.utils.checkpoints import validate_resume_arguments

    original = {"method": "self-opd", "teacher": "./teacher", "seed": 0, "max_steps": 2}
    (tmp_path / "launch.json").write_text(json.dumps({"arguments": original}))
    validate_resume_arguments(tmp_path, {**original, "max_steps": 3})
    for changes in [{"teacher": "./different"}, {"seed": 1}]:
        with pytest.raises(ValueError):
            validate_resume_arguments(tmp_path, {**original, **changes})
