#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--teacher-enable-thinking", choices=("True", "False"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-registered-vocab-subset", action="store_true")
    return parser.parse_args()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_registered_vocab_contract(student_vocab, teacher_vocab, student_rows, teacher_rows):
    if any(teacher_vocab.get(token) != token_id for token, token_id in student_vocab.items()):
        raise ValueError(
            "Vocabulary masking requires every registered student token to share the teacher token ID"
        )
    sizes = []
    for vocab, rows in ((student_vocab, student_rows), (teacher_vocab, teacher_rows)):
        ids = set(vocab.values())
        if not ids or any(type(token_id) is not int for token_id in ids):
            raise ValueError("Vocabulary masking requires integer registered token IDs")
        size = len(ids)
        if min(ids) != 0 or max(ids) != size - 1 or size > rows:
            raise ValueError(
                "Vocabulary masking requires contiguous registered IDs within each model vocabulary"
            )
        sizes.append(size)
    return {"student_valid_vocab_size": sizes[0], "teacher_valid_vocab_size": sizes[1]}


def main() -> None:
    args = parse_args()
    teacher_enable_thinking = args.teacher_enable_thinking == "True"
    student_config = AutoConfig.from_pretrained(args.student_model, trust_remote_code=False)
    teacher_config = AutoConfig.from_pretrained(args.teacher_model, trust_remote_code=False)

    student_tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=False)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=False)
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    registered_contract = None
    if args.require_registered_vocab_subset:
        registered_contract = validate_registered_vocab_contract(
            student_vocab, teacher_vocab, student_config.vocab_size, teacher_config.vocab_size
        )
    conflicting_tokens = {
        token: (token_id, teacher_vocab.get(token))
        for token, token_id in student_vocab.items()
        if teacher_vocab.get(token) is not None and teacher_vocab[token] != token_id
    }
    if conflicting_tokens:
        examples = list(conflicting_tokens.items())[:5]
        raise ValueError(f"Student and teacher token IDs conflict: {examples}")
    missing_tokens = {
        token: token_id for token, token_id in student_vocab.items() if token not in teacher_vocab
    }
    unsupported_tokens = {
        token: token_id
        for token, token_id in missing_tokens.items()
        if token_id < 0 or token_id >= teacher_config.vocab_size
    }
    if unsupported_tokens:
        examples = list(unsupported_tokens.items())[:5]
        raise ValueError(
            "Student token IDs missing from the teacher tokenizer exceed the teacher model vocabulary: "
            f"{examples}"
        )

    messages = [{"role": "user", "content": "What is 1+1?"}]
    student_prompt = student_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    teacher_prompt = teacher_tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=teacher_enable_thinking,
    )
    forced_nonthinking_suffix = "<think>\n\n</think>\n\n"
    if teacher_config.model_type == "qwen3":
        if teacher_enable_thinking and teacher_prompt.endswith(forced_nonthinking_suffix):
            raise ValueError(
                "Thinking teacher prompt unexpectedly contains the forced non-thinking suffix"
            )
        if not teacher_enable_thinking and not teacher_prompt.endswith(forced_nonthinking_suffix):
            raise ValueError(
                "Non-thinking teacher prompt is missing the forced non-thinking suffix"
            )
    elif teacher_enable_thinking:
        raise ValueError(
            f"Thinking mode is only supported for Qwen3 teachers, got {teacher_config.model_type!r}"
        )

    result = {
        "registered_vocab_mask_contract": registered_contract,
        "student_model": str(Path(args.student_model)),
        "teacher_model": str(Path(args.teacher_model)),
        "student_model_type": student_config.model_type,
        "teacher_model_type": teacher_config.model_type,
        "student_model_vocab_size": student_config.vocab_size,
        "teacher_model_vocab_size": teacher_config.vocab_size,
        "student_registered_tokens": len(student_vocab),
        "teacher_registered_tokens": len(teacher_vocab),
        "shared_token_ids_match": True,
        "student_vocab_is_teacher_subset_with_same_ids": not missing_tokens,
        "student_tokens_missing_from_teacher_tokenizer": [
            {"token": token, "token_id": token_id}
            for token, token_id in sorted(missing_tokens.items(), key=lambda item: item[1])
        ],
        "missing_student_token_ids_fit_teacher_model_vocab": True,
        "student_enable_thinking": False,
        "teacher_enable_thinking": teacher_enable_thinking,
        "student_chat_template_sha256": sha256_text(student_tokenizer.chat_template),
        "teacher_chat_template_sha256": sha256_text(teacher_tokenizer.chat_template),
        "student_rendered_prompt": student_prompt,
        "teacher_rendered_prompt": teacher_prompt,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
