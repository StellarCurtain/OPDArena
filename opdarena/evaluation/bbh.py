"""Extract BBH conclusions without consulting the reference answer."""

import re

VERSION = "bbh-conclusion-v1"
LEGACY_PATTERN = re.compile(r"(?<=the answer is )(.*)(?=.)")
CONCLUSION_PATTERN = re.compile(
    r"(?:the\s+(?:final\s+|correct\s+)?answer\s*(?:is|:)\s*|final\s+answer\s*:\s*)",
    re.IGNORECASE,
)
VOCABULARIES = {
    "boolean_expressions": {"true": "True", "false": "False"},
    "causal_judgement": {"yes": "Yes", "no": "No"},
    "web_of_lies": {"yes": "Yes", "no": "No"},
    "navigate": {"yes": "Yes", "no": "No"},
    "sports_understanding": {"yes": "yes", "no": "no"},
    "formal_fallacies": {"valid": "valid", "invalid": "invalid"},
}


def legacy_answer(text):
    matches = LEGACY_PATTERN.findall(text)
    return matches[0].strip() if matches else "[invalid]"


def _boxed(text):
    results = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth, end = 1, match.end()
        while end < len(text) and depth:
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            end += 1
        if not depth:
            results.append((match.start(), text[match.end() : end - 1]))
    return results


def extract_answer(text, task_name, question):
    task = task_name.removeprefix("bbh_cot_fewshot_")
    fallback = legacy_answer(text)
    cues = [(match.start(), text[match.end() :]) for match in CONCLUSION_PATTERN.finditer(text)]
    cues += _boxed(text)
    if not cues:
        return fallback
    _, value = max(cues, key=lambda item: item[0])
    value = value.strip().strip("`*").strip().strip("\"'").strip().lstrip(":").strip()
    value = re.sub(r"^\\(?:text|mathrm|mathbf)\{([^{}]+)\}", r"\1", value)
    value = value.lstrip("$").lstrip("`*").strip()
    if not value:
        return fallback
    options = set(re.findall(r"^\(([A-Z])\)", question, re.MULTILINE))
    if options:
        match = re.match(r"(?:[Oo]ption\s+)?[\(\[]?([A-Z])[\)\]]?(?=$|[\s.,:;!])", value)
        return "(" + match[1] + ")" if match and match[1] in options else fallback
    vocabulary = VOCABULARIES.get(task)
    if vocabulary:
        match = re.match(r"([A-Za-z]+)\b", value)
        return (
            vocabulary[match[1].lower()] if match and match[1].lower() in vocabulary else fallback
        )
    if task in ("object_counting", "multistep_arithmetic_two"):
        match = re.match(r"(-?\d+)(?=$|[\s.,$}])", value)
        return str(int(match[1])) if match else fallback
    if task == "dyck_languages":
        match = re.match(r"([\])}>\s]+)", value)
        return " ".join(match[1].split()) if match else fallback
    if task == "word_sorting":
        value = value.splitlines()[0].rstrip(".").strip("\"'`*")
        if re.fullmatch(r"[a-z]+(?:\s+[a-z]+)*", value):
            return " ".join(value.split())
    return fallback


def install_filter(task, task_name):
    """Update both the live harness filter and its exported task configuration."""
    from lm_eval.api.filter import Filter
    from lm_eval.api.registry import filter_registry, register_filter
    from lm_eval.filters import build_filter_ensemble

    if "bbh_conclusion" not in filter_registry:

        @register_filter("bbh_conclusion")
        class ConclusionFilter(Filter):
            def __init__(self, task_name, **kwargs):
                self.task_name = task_name

            def apply(self, resps, docs):
                return [
                    [extract_answer(text, self.task_name, doc["input"]) for text in answers]
                    for answers, doc in zip(resps, docs, strict=True)
                ]

    components = [("bbh_conclusion", {"task_name": task_name}), ("take_first", None)]
    task._filters = [build_filter_ensemble("get-answer", components)]
    task.set_config(
        key="filter_list",
        value=[
            {
                "name": "get-answer",
                "filter": [{"function": name, **(kwargs or {})} for name, kwargs in components],
            }
        ],
    )
    task.set_config(key="metadata", value={"answer_extraction": VERSION}, update=True)
