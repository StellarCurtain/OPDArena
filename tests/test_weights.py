import json

import pytest
import torch
from safetensors.torch import save_file

from opdarena.analysis.weights import Weights, scale, trajectory


def model(path, values):
    path.mkdir()
    save_file({"weight": torch.tensor(values)}, path / "model.safetensors")
    (path / "config.json").write_text(json.dumps({"model_type": "fixture"}))
    return Weights(path)


def test_scaling_endpoints_and_extrapolation(tmp_path):
    base = model(tmp_path / "base", [1.0, 2.0])
    endpoint = model(tmp_path / "end", [3.0, 6.0])
    (base.path / "chat_template.jinja").write_text("{{ messages }}")
    for alpha, expected in [(0, [1.0, 2.0]), (1, [3.0, 6.0]), (1.5, [4.0, 8.0])]:
        output = tmp_path / str(alpha)
        scale(base, endpoint, alpha, output)
        assert torch.equal(Weights(output).get("weight"), torch.tensor(expected))
        assert (output / "chat_template.jinja").read_text() == "{{ messages }}"
        with pytest.raises(FileExistsError):
            scale(base, endpoint, alpha, output)


def test_geometry_projection_and_off_axis(tmp_path):
    base = model(tmp_path / "base", [0.0, 0.0])
    endpoint = model(tmp_path / "end", [2.0, 0.0])
    student = model(tmp_path / "student", [1.0, 1.0])
    got = trajectory(base, endpoint, student)
    assert got["projection_alpha"] == pytest.approx(0.5)
    assert got["normalized_off_axis"] == pytest.approx(0.5)
    assert got["student_to_endpoint"] == pytest.approx(2**0.5)
