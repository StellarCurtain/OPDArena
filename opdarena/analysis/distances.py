"""JS, directional KL and top-one agreement on aligned log probabilities."""

import numpy as np
import torch


def distances(logp, logq):
    if logp.shape != logq.shape or logp.ndim != 2:
        raise ValueError("Predictions must have the same positions and vocabulary")
    if not torch.isfinite(logp).all() or not torch.isfinite(logq).all():
        raise ValueError("Nonfinite log probabilities")
    p, q = logp.exp(), logq.exp()
    middle = torch.logaddexp(logp, logq) - np.log(2)
    return {
        "js": float(
            (
                0.5
                * (
                    (p * (logp - middle)).sum(-1, dtype=torch.float64)
                    + (q * (logq - middle)).sum(-1, dtype=torch.float64)
                )
            ).mean()
        ),
        "kl_first_second": float((p * (logp - logq)).sum(-1, dtype=torch.float64).mean()),
        "kl_second_first": float((q * (logq - logp)).sum(-1, dtype=torch.float64).mean()),
        "top1_agreement": float((logp.argmax(-1) == logq.argmax(-1)).double().mean()),
    }
