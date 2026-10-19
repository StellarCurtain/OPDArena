"""TriviaQA metrics that retain the complete sampled response."""

from __future__ import annotations

import re
import string
from typing import Any

LEGACY_STOPS = ("\n", ".", ",")


def truncate_at_legacy_stop(text: str) -> str:
    positions = [text.find(stop) for stop in LEGACY_STOPS]
    positions = [position for position in positions if position >= 0]
    if positions:
        text = text[: min(positions)]
    return text.strip()


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(char for char in text if char not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(
    references: list[str],
    predictions: list[str],
    regexes_to_ignore: list[str] | None = None,
    ignore_case: bool = False,
    ignore_punctuation: bool = False,
    ignore_numbers: bool = False,
) -> dict[str, float]:
    from lm_eval.api.metrics import exact_match_hf_evaluate

    return exact_match_hf_evaluate(
        references=references,
        predictions=[truncate_at_legacy_stop(item) for item in predictions],
        regexes_to_ignore=regexes_to_ignore,
        ignore_case=ignore_case,
        ignore_punctuation=ignore_punctuation,
        ignore_numbers=ignore_numbers,
    )


def flexible_contains_match(references: list[str], predictions: list[Any]) -> dict[str, float]:
    reference = normalize_answer(references[0])
    prediction: Any = predictions[0]
    while isinstance(prediction, (list, tuple)):
        if not prediction:
            prediction = ""
            break
        prediction = prediction[0]
    prediction = normalize_answer(str(prediction))
    return {"flexible_contains_match": float(bool(reference) and reference in prediction)}
