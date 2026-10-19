"""Training OOD suites; historical runs retain their recorded protocol."""

PAPER = "paper-v2"
LEGACY = "legacy-v1"
PAPER_DATASETS = ("HellaSwag", "ARC-Challenge", "MMLU", "IFEval", "TriviaQA", "BBH", "NQ")
LEGACY_DATASETS = ("HellaSwag", "ARC-Challenge", "MMLU", "MMLU-Pro", "IFEval", "TriviaQA", "BBH")
COUNTS = {
    "HellaSwag": 256,
    "ARC-Challenge": 256,
    "MMLU": 570,
    "MMLU-Pro": 140,
    "IFEval": 541,
    "TriviaQA": 500,
    "BBH": 270,
    "NQ": 3610,
}
GENERATION = frozenset(("MMLU-Pro", "IFEval", "TriviaQA", "BBH", "NQ"))
MANIFEST_PROTOCOLS = {
    PAPER: "rethinking-opd-training-validation-v2",
    LEGACY: "rethinking-opd-training-validation-v1",
}


def datasets(protocol):
    if protocol not in MANIFEST_PROTOCOLS:
        raise ValueError("Unknown OOD protocol: " + str(protocol))
    return PAPER_DATASETS if protocol == PAPER else LEGACY_DATASETS


def response_budget(protocol, task):
    datasets(protocol)
    return {
        "IFEval": 512 if protocol == PAPER else 2048,
        "MMLU-Pro": 2048,
        "TriviaQA": 256,
        "BBH": 1024,
        "NQ": 256,
    }.get(task)
