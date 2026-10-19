"""A normalized teacher containing only a source model's training increment."""

import copy
from types import SimpleNamespace


def initialize(worker):
    source_base = worker.config.model.get("increment_source_base_path")
    anchor = worker.config.model.get("increment_student_base_path")
    if not source_base and not anchor:
        worker.increment_modules = []
        return
    if not source_base or not anchor or worker.use_fused_kernels:
        raise ValueError("A transferred teacher needs both frozen Base models and explicit logits")
    from transformers import AutoConfig, AutoTokenizer

    paths = [worker.config.model.path, source_base, anchor]
    vocabularies = [
        AutoTokenizer.from_pretrained(p, local_files_only=True).get_vocab() for p in paths
    ]
    sizes = [AutoConfig.from_pretrained(p, local_files_only=True).vocab_size for p in paths]
    if any(v != vocabularies[0] for v in vocabularies[1:]):
        raise ValueError("Transferred teachers require identical registered vocabulary IDs")
    if (
        sizes[0] != sizes[1]
        or min(sizes) < max(vocabularies[0].values()) + 1
        or sizes[0] < sizes[2]
    ):
        raise ValueError("Source models must cover the complete student logit support")
    # Qwen2.5 shares registered token IDs across sizes but pads its output heads
    # differently. Normalize the constructed teacher on the student's support.
    # The two source-model normalization constants cancel in their ratio.
    worker.increment_output_vocab_size = sizes[2]
    original = {
        name: getattr(worker, name)
        for name in ("_do_switch_chat_template", "input_tokenizer", "tokenizer")
        if hasattr(worker, name)
    }
    modules = []
    for path in (source_base, anchor):
        config = copy.deepcopy(worker.config)
        config.model.path = path
        modules.append(worker._build_model(config).eval())
    for name, value in original.items():
        setattr(worker, name, value)
    worker.increment_modules = modules


def logits_of(output):
    return output[0] if isinstance(output, tuple) else output.logits


def forward(worker, **inputs):
    output = worker.reward_module(**inputs)
    modules = getattr(worker, "increment_modules", [])
    if not modules:
        return output
    # softmax(z_student_Base + z_teacher_final - z_teacher_Base)
    # equals normalize(p_student_Base * q_teacher_final / q_teacher_Base).
    size = getattr(worker, "increment_output_vocab_size", logits_of(output).shape[-1])
    logits = logits_of(output)[..., :size].float()
    del output
    baseline = logits_of(modules[0](**inputs))[..., :size]
    if baseline.shape != logits.shape:
        raise ValueError("Teacher endpoint and source Base have different output shapes")
    logits.sub_(baseline.float())
    del baseline
    anchor = logits_of(modules[1](**inputs))
    if anchor.shape != logits.shape:
        raise ValueError("Student Base and source teacher have different output shapes")
    logits.add_(anchor.float())
    return SimpleNamespace(logits=logits)
