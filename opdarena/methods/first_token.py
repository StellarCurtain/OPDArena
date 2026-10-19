"""First-response-token ablation with an unchanged loss denominator."""

from pathlib import Path

from .common import save


def apply(data, mode, step, directory):
    if mode not in ("original", "mask_first"):
        raise ValueError("Unknown first-token control")
    advantages, mask = data.batch["advantages"], data.batch["response_mask"]
    if advantages.ndim != 3 or advantages.shape[-1] != 16 or advantages.shape[:2] != mask.shape:
        raise ValueError("First-token control requires ordinary Top16 OPD")
    valid = mask.bool()
    first = valid & (valid.long().cumsum(-1) == 1)
    before = float(advantages[first].abs().mean()) if first.any() else 0.0
    if mode == "mask_first":
        advantages[first] = 0.0
    # Do not change response_mask: the original token/example normalization stays fixed.
    stats = {
        "mode": mode,
        "step": step,
        "response_tokens": int(mask.sum()),
        "first_positions": int(first.sum()),
        "removed_positions": int(first.sum()) if mode == "mask_first" else 0,
        "first_absolute_feedback_before": before,
        "first_absolute_feedback_after": float(advantages[first].abs().mean())
        if first.any()
        else 0.0,
    }
    save(stats, Path(directory) / f"step-{step}.json")
    return {"feedback_first/" + k: float(v) for k, v in stats.items() if k not in ("mode", "step")}
