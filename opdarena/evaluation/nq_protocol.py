"""NQ-Open prompts and answer scoring."""

import hashlib
import json
import re
import string
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


PROTOCOL = {
    "version": "nq-open-v1",
    "dataset": "google-research-datasets/nq_open",
    "split": "validation",
    "questions": 3610,
    "fewshot": 5,
    "fewshot_seed": 0,
    "input_format": "raw",
    "responses_per_question": 1,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": -1,
    "seed": 0,
    "max_response_tokens": 256,
    "max_model_len": 4096,
    "stop": ["\n", ".", ","],
    "prompt": "lm-evaluation-harness nq_open; five fixed training examples",
    "score": "normalized exact match against any official answer alias",
    "normalization": "lowercase, remove ASCII punctuation and a/an/the, collapse whitespace",
    "sampling_seed": "SHA256(0:question_id) modulo 2**31; independent of checkpoint",
}


def normalize(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def exact_match(response, answers):
    if not answers or not all(isinstance(a, str) and a.strip() for a in answers):
        raise ValueError("Missing official answer aliases")
    return int(any(normalize(response) == normalize(answer) for answer in answers))


def prompt(question, examples):
    prefix = "Answer these questions:\n\n"
    for example in examples:
        prefix += f"Q: {example['question']}?\nA: {example['answer'][0]}\n"
    return prefix + f"Q: {question}?\nA:"


def validate_questions(data):
    rows = data["questions"]
    if len(rows) != 3610 or len({q["id"] for q in rows}) != 3610 or len(data["examples"]) != 5:
        raise ValueError("Expected all 3610 questions and five training examples")
    for q in rows:
        if not q["answers"] or q["prompt"] != prompt(q["question"], data["examples"]):
            raise ValueError("Question answers or fixed prompt changed")
        if q["id"] != digest(
            {"index": q["index"], "question": q["question"], "answers": q["answers"]}
        ):
            raise ValueError("Question identity changed")


def load_questions(path):
    if not path:
        raise ValueError("Supply the local NQ/questions.json created by opdarena.data.benchmarks")
    data = json.loads(Path(path).read_text())
    validate_questions(data)
    return data


def question_seed(question_id):
    return int(hashlib.sha256(("0:" + question_id).encode()).hexdigest()[:8], 16) % 2**31
