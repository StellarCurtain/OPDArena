from types import SimpleNamespace

import pytest
import torch

from opdarena.analysis.distances import distances
from opdarena.methods.first_token import apply
from opdarena.utils.results import Results


def test_directional_kl_and_js():
    p = torch.tensor([[0.8, 0.2]], dtype=torch.float64)
    q = torch.tensor([[0.3, 0.7]], dtype=torch.float64)
    got = distances(p.log(), q.log())
    assert got["kl_first_second"] == pytest.approx(float((p * (p / q).log()).sum()))
    assert got["kl_second_first"] == pytest.approx(float((q * (q / p).log()).sum()))
    assert got["js"] == pytest.approx(distances(q.log(), p.log())["js"])
    assert distances(p.log(), p.log())["js"] == pytest.approx(0, abs=1e-12)


def test_first_token_preserves_response_mask_and_later_feedback(tmp_path):
    mask = torch.tensor([[0, 1, 1], [1, 1, 0]])
    advantage = torch.ones((2, 3, 16))
    data = SimpleNamespace(batch={"response_mask": mask.clone(), "advantages": advantage})
    result = apply(data, "mask_first", 1, tmp_path)
    assert torch.equal(data.batch["response_mask"], mask)
    assert not advantage[0, 1].any() and not advantage[1, 0].any()
    assert advantage[0, 2].eq(1).all() and advantage[1, 1].eq(1).all()
    assert result["feedback_first/removed_positions"] == 2


def test_results_reject_protocol_mixing(tmp_path):
    results = Results(tmp_path / "results.sqlite")
    results.put("protocol", {"seed": 0}, immutable=True)
    with pytest.raises(ValueError):
        results.put("protocol", {"seed": 1}, immutable=True)
    assert results.get("protocol") == {"seed": 0}
    results.close()
