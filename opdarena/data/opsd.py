#!/usr/bin/env python3
"""Dataset adapter that gives an OPD teacher privileged reference context."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

from verl.utils.dataset.rl_dataset import RLHFDataset

REFERENCE_SOLUTION_HEADER = "Here is a privileged reference solution to the original problem:"
REFERENCE_SOLUTION_BEGIN = "=== Reference Solution Begin ==="
REFERENCE_SOLUTION_END = "=== Reference Solution End ==="
TEACHER_TRANSITION = (
    "Use the reference solution as private guidance. Independently solve the original problem, "
    "and produce the answer in the format requested in the original prompt."
)


def nested_value(record: Mapping[str, Any], dotted_path: str) -> Any:
    """Read a dotted field path such as ``extra_info.answer``."""

    value: Any = record
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(dotted_path)
        value = value[key]
    return value


def build_privileged_teacher_prompt(
    student_prompt: Sequence[Mapping[str, Any]], context: str
) -> list[dict]:
    """Append private context to the final user message without mutating the student prompt."""

    if not isinstance(context, str) or not context.strip():
        raise ValueError("privileged teacher context must be non-empty text")

    teacher_prompt = copy.deepcopy(list(student_prompt))
    if not teacher_prompt:
        raise ValueError("student prompt must contain at least one message")

    final_message = teacher_prompt[-1]
    if not isinstance(final_message, dict) or final_message.get("role") != "user":
        raise ValueError("student prompt must end with a user message")
    content = final_message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("final student user message must contain non-empty text")

    final_message["content"] = (
        f"{content}\n\n"
        f"{REFERENCE_SOLUTION_HEADER}\n"
        f"{REFERENCE_SOLUTION_BEGIN}\n"
        f"{context.strip()}\n"
        f"{REFERENCE_SOLUTION_END}\n\n"
        f"{TEACHER_TRANSITION}"
    )
    return teacher_prompt


class PrivilegedTeacherRLHFDataset(RLHFDataset):
    """RLHF dataset with a separate teacher prompt for OPSD training.

    The regular ``raw_prompt`` remains the only input used by the student rollout.
    For the configured training source, ``teacher_prompt`` contains the same prompt
    plus privileged context. Other sources, including validation datasets, copy the
    regular prompt so the shared validation workflow remains unchanged.
    """

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and len(args) >= 3:
            config = args[2]
        if config is None:
            raise TypeError("config is required")

        opsd_config = config.get("opsd", {})
        self.teacher_prompt_key = opsd_config.get("teacher_prompt_key", "teacher_prompt")
        self.privileged_context_path = opsd_config.get(
            "privileged_context_path", "extra_info.answer"
        )
        self.required_data_source = opsd_config.get("required_data_source", "openr1")
        super().__init__(*args, **kwargs)

    def build_teacher_prompt(
        self, student_prompt: Sequence[Mapping[str, Any]], context: str
    ) -> list[dict]:
        """Extension point for experiments that change the teacher's private input."""

        return build_privileged_teacher_prompt(student_prompt, context)

    def __getitem__(self, item):
        example = super().__getitem__(item)
        if "raw_prompt" not in example:
            raise ValueError("OPSD requires data.return_raw_chat=True")

        student_prompt = example["raw_prompt"]
        if str(example.get("data_source")) != self.required_data_source:
            example[self.teacher_prompt_key] = copy.deepcopy(student_prompt)
            return example

        try:
            context = nested_value(example, self.privileged_context_path)
        except KeyError as error:
            raise ValueError(
                f"row {item} from {self.required_data_source!r} is missing privileged context "
                f"at {self.privileged_context_path!r}"
            ) from error
        try:
            example[self.teacher_prompt_key] = self.build_teacher_prompt(student_prompt, context)
        except ValueError as error:
            raise ValueError(f"row {item}: {error}") from error
        return example


class AnswerBoundaryDataset(PrivilegedTeacherRLHFDataset):
    """Raw teacher input with the paper's explicit answer boundary."""

    def build_teacher_prompt(self, student_prompt, context):
        messages = build_privileged_teacher_prompt(student_prompt, context)
        messages[-1]["content"] += "\n\nAnswer to the original problem:\n"
        return messages
