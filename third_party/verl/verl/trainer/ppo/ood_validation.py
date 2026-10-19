# Modified for OPDArena; see THIRD_PARTY.md for integration changes.
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small, fixed OOD validation suite evaluated by the live training actor."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from opdarena.evaluation import protocol as ood_protocol
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

LM_EVAL_COMMIT = "fe13f316285edad122818e39aa059152132267fd"
OOD_PROTOCOL = ood_protocol.MANIFEST_PROTOCOLS[ood_protocol.PAPER]
OOD_DATASET_ORDER = ood_protocol.PAPER_DATASETS
SUPPORTED_DATASETS = (*ood_protocol.LEGACY_DATASETS, 'NQ')
GENERATION_DATASETS = ood_protocol.GENERATION


@dataclass(frozen=True)
class OODTaskSpec:
    lm_eval_name: str
    samples_per_leaf: int | None
    expected_leaf_tasks: int
    primary_metric: str
    max_generation_tokens: int | None = None


OOD_TASK_SPECS = {
    "HellaSwag": OODTaskSpec("hellaswag", 256, 1, "acc,none"),
    "ARC-Challenge": OODTaskSpec("arc_challenge", 256, 1, "acc,none"),
    "MMLU": OODTaskSpec("mmlu", 10, 57, "acc,none"),
    "MMLU-Pro": OODTaskSpec("mmlu_pro", 10, 14, "exact_match,custom-extract", 2048),
    "IFEval": OODTaskSpec("ifeval", None, 1, "prompt_level_strict_acc,none", 512),
    "TriviaQA": OODTaskSpec("triviaqa", 500, 1, "exact_match,remove_whitespace", 256),
    "BBH": OODTaskSpec("bbh", 10, 27, "exact_match,get-answer", 1024),
    "NQ": OODTaskSpec("nq_open", None, 1, "exact_match,none", 256),
}


def _stable_seed(seed: int, dataset: str, task_name: str) -> int:
    payload = f"{seed}:{dataset}:{task_name}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def fixed_subset_indices(length: int, count: int | None, seed: int, dataset: str, task_name: str) -> list[int]:
    """Select one deterministic subset without depending on task enumeration order."""

    if length <= 0:
        raise ValueError(f"OOD task {task_name} is empty")
    if count is None:
        return list(range(length))
    if length < count:
        raise ValueError(f"OOD task {task_name} has {length} rows, fewer than requested {count}")
    rng = np.random.default_rng(_stable_seed(seed, dataset, task_name))
    return sorted(int(index) for index in rng.choice(length, size=count, replace=False))


def infer_validation_model_type(model_path: str) -> str:
    """Infer Base versus Instruct for the supported Qwen2.5 and Qwen3 paths."""

    name = Path(model_path.rstrip("/")).name.lower()
    if "instruct" in name:
        return "instruct"
    if name.endswith("-base"):
        return "base"
    if name.startswith("qwen3-"):
        return "instruct"
    if name.startswith("qwen2.5-") or name.startswith("qwen2_5-"):
        return "base"
    raise ValueError(
        f"Cannot infer validation model type from {model_path!r}; set trainer.ood_validation.model_type explicitly"
    )


def _object_array(items: list[Any]) -> np.ndarray:
    array = np.empty(len(items), dtype=object)
    array[:] = items
    return array


def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.cumsum(attention_mask, dim=-1) - 1, min=0)


def _prompt_batch(token_lists: list[list[int]], pad_token_id: int) -> DataProto:
    if not token_lists:
        raise ValueError("cannot build an empty OOD prompt batch")
    width = max(len(tokens) for tokens in token_lists)
    input_ids = torch.full((len(token_lists), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, tokens in enumerate(token_lists):
        if tokens:
            input_ids[row, -len(tokens) :] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[row, -len(tokens) :] = 1
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": _position_ids(attention_mask),
        },
        non_tensors={"raw_prompt_ids": _object_array([list(tokens) for tokens in token_lists])},
    )


def _likelihood_batch(
    requests: list[tuple[tuple[str, str], list[int], list[int]]], pad_token_id: int, max_model_length: int
) -> tuple[DataProto, torch.Tensor]:
    contexts: list[list[int]] = []
    continuations: list[list[int]] = []
    for _, context_tokens, continuation_tokens in requests:
        if not continuation_tokens:
            raise ValueError("OOD likelihood continuation tokenized to an empty sequence")
        if len(continuation_tokens) >= max_model_length:
            raise ValueError("OOD likelihood continuation exceeds the model context")
        available_context = max_model_length - len(continuation_tokens)
        contexts.append(context_tokens[-available_context:])
        continuations.append(continuation_tokens)

    prompt_width = max(len(tokens) for tokens in contexts)
    response_width = max(len(tokens) for tokens in continuations)
    input_ids = torch.full((len(requests), prompt_width + response_width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    responses = torch.full((len(requests), response_width), pad_token_id, dtype=torch.long)
    response_mask = torch.zeros_like(responses)
    for row, (context_tokens, continuation_tokens) in enumerate(zip(contexts, continuations, strict=True)):
        if context_tokens:
            input_ids[row, prompt_width - len(context_tokens) : prompt_width] = torch.tensor(
                context_tokens, dtype=torch.long
            )
            attention_mask[row, prompt_width - len(context_tokens) : prompt_width] = 1
        continuation = torch.tensor(continuation_tokens, dtype=torch.long)
        responses[row, : len(continuation_tokens)] = continuation
        response_mask[row, : len(continuation_tokens)] = 1
        input_ids[row, prompt_width : prompt_width + len(continuation_tokens)] = continuation
        attention_mask[row, prompt_width : prompt_width + len(continuation_tokens)] = 1

    return (
        DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": _position_ids(attention_mask),
                "responses": responses,
            }
        ),
        response_mask,
    )


def _generation_sampling_params(generation_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate lm-eval generation kwargs into vLLM validation overrides."""

    return {
        "do_sample": bool(generation_kwargs.get("do_sample", True)),
        "temperature": float(generation_kwargs.get("temperature", 0.7)),
        "top_p": float(generation_kwargs.get("top_p", 0.95)),
        "max_tokens": int(generation_kwargs.get("max_gen_toks", 256)),
        "stop": list(generation_kwargs.get("until", [])),
        "seed": 0,
        "skip_special_tokens": True,
        # vLLM rejects string stop conditions when detokenization is disabled.
        # The rollout still returns token_ids, which are what this adapter uses.
        "detokenize": True,
    }


def normalize_ood_datasets(value: Any = None) -> tuple[str, ...]:
    """Return requested datasets in the suite's fixed display order."""

    if value is None:
        return OOD_DATASET_ORDER
    if isinstance(value, str):
        requested = [item.strip() for item in value.split(",") if item.strip()]
    else:
        requested = [str(item).strip() for item in value if str(item).strip()]
    if not requested:
        raise ValueError("trainer.ood_validation.datasets must select at least one dataset")
    if len(requested) != len(set(requested)):
        raise ValueError(f"trainer.ood_validation.datasets contains duplicates: {requested}")
    unknown = set(requested) - set(SUPPORTED_DATASETS)
    if unknown:
        raise ValueError(f"unknown OOD datasets: {sorted(unknown)}")
    return tuple(dataset for dataset in SUPPORTED_DATASETS if dataset in requested)


def _config_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{name} must be True or False, got {value!r}")


def _apply_chat_template(tokenizer, chat_history, add_generation_prompt: bool, enable_thinking: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    try:
        kwargs["continue_final_message"] = not add_generation_prompt
        return tokenizer.apply_chat_template(chat_history, **kwargs)
    except TypeError:
        kwargs.pop("continue_final_message", None)
        return tokenizer.apply_chat_template(chat_history, **kwargs)


def _make_actor_adapter(trainer, max_model_length: int, enable_thinking: bool):
    # lm-eval is an optional dependency unless OOD validation is selected.
    from lm_eval.api.model import TemplateLM

    class ActorLmEvalAdapter(TemplateLM):
        def __init__(self):
            super().__init__()
            self.tokenizer = trainer.tokenizer
            self.actor_rollout_wg = trainer.actor_rollout_wg
            self.max_model_length = max_model_length
            self.generation_lengths: list[int] = []

        @property
        def eot_token_id(self) -> int:
            return self.tokenizer.eos_token_id

        @property
        def prefix_token_id(self) -> int:
            bos_token_id = self.tokenizer.bos_token_id
            return self.tokenizer.eos_token_id if bos_token_id is None else bos_token_id

        @property
        def max_length(self) -> int:
            return self.max_model_length

        @property
        def tokenizer_name(self) -> str:
            return self.tokenizer.name_or_path.replace("/", "__")

        def tok_encode(self, string: str, add_special_tokens: bool | None = None, **kwargs) -> list[int]:
            tokenizer_kwargs = dict(kwargs)
            if add_special_tokens is not None:
                tokenizer_kwargs["add_special_tokens"] = add_special_tokens
            return self.tokenizer.encode(string, **tokenizer_kwargs)

        def apply_chat_template(self, chat_history, add_generation_prompt: bool = True) -> str:
            return _apply_chat_template(
                self.tokenizer, chat_history, add_generation_prompt, enable_thinking
            )

        def _padded(self, batch: DataProto) -> tuple[DataProto, int]:
            return pad_dataproto_to_divisor(batch, self.actor_rollout_wg.world_size)

        def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False) -> list[tuple[float, bool]]:
            del disable_tqdm
            batch, response_mask = _likelihood_batch(requests, self.tokenizer.pad_token_id, self.max_model_length)
            padded, pad_size = self._padded(batch)
            output = self.actor_rollout_wg.compute_log_prob(padded)
            output = unpad_dataproto(output, pad_size)
            log_probs = output.batch["old_log_probs"].float().cpu()
            totals = (log_probs * response_mask).sum(dim=-1).tolist()
            # The seven-task suite uses the scalar likelihood but none of its
            # metrics consumes lm-eval's optional token-wise greedy flag.
            return [(float(total), False) for total in totals]

        def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
            del requests, disable_tqdm
            raise NotImplementedError("rolling likelihood is not used by the fixed OOD suite")

        def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
            del disable_tqdm
            if not requests:
                return []
            encoded: list[list[int]] = []
            normalized_kwargs: dict[str, Any] | None = None
            for request in requests:
                context, generation_kwargs = request.args
                current = _generation_sampling_params(generation_kwargs)
                if normalized_kwargs is None:
                    normalized_kwargs = current
                elif current != normalized_kwargs:
                    raise ValueError("one OOD task produced inconsistent generation settings")
                tokens = self.tok_encode(context)
                max_prompt_tokens = self.max_model_length - current["max_tokens"]
                if max_prompt_tokens <= 0:
                    raise ValueError("OOD generation budget leaves no room for its prompt")
                encoded.append(tokens[-max_prompt_tokens:])

            assert normalized_kwargs is not None
            batch = _prompt_batch(encoded, self.tokenizer.pad_token_id)
            batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": normalized_kwargs["do_sample"],
                "validate": True,
                "global_steps": trainer.global_steps,
                "validation_sampling_params": normalized_kwargs,
            }
            padded, pad_size = self._padded(batch)
            if trainer.async_rollout_mode:
                output = trainer.async_rollout_manager.generate_sequences(padded)
            else:
                output = self.actor_rollout_wg.generate_sequences(padded)
            output = unpad_dataproto(output, pad_size)
            response_ids = output.batch["responses"]
            response_mask = output.batch["attention_mask"][:, -response_ids.shape[-1] :].bool()
            texts: list[str] = []
            for ids, mask in zip(response_ids, response_mask, strict=True):
                valid_ids = ids[mask].cpu().tolist()
                self.generation_lengths.append(len(valid_ids))
                texts.append(self.tokenizer.decode(valid_ids, skip_special_tokens=True))
            return texts

    return ActorLmEvalAdapter()


def bbh_stop_strings(until):
    """Allow paragraph breaks; stop only at EOS or an actual next question."""
    until = [until] if isinstance(until, str) else list(until or [])
    return ["\nQ:" if value == "Q" else value for value in until if value != "\n\n"]


def _prepare_loaded_tasks(loaded: dict[str, Any], generation_tokens: int | None, fewshot_seed: int) -> None:
    for name, task in loaded["tasks"].items():
        if task.get_config("output_type") == "generate_until":
            task.set_config(
                key="generation_kwargs",
                value={
                    "do_sample": True,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "max_gen_toks": generation_tokens,
                },
                update=True,
            )
            if name.startswith("bbh_"):
                from opdarena.evaluation.bbh import install_filter

                kwargs = task.get_config("generation_kwargs") or {}
                task.set_config(
                    key="generation_kwargs", value={"until": bbh_stop_strings(kwargs.get("until"))}, update=True
                )
                install_filter(task, name)
            if name == 'ifeval' and generation_tokens == 512:
                process_results = task.process_results

                def stable_results(doc, responses, process=process_results):
                    # Match the existing per-document CPU IFEval-512 scorer.
                    doc_hash = hashlib.sha256(json.dumps(doc, indent=2, ensure_ascii=False).encode()).hexdigest()
                    state = random.getstate()
                    try:
                        random.seed(int.from_bytes(hashlib.sha256(doc_hash.encode()).digest()[:8]))
                        return process(doc, responses)
                    finally:
                        random.setstate(state)

                task.process_results = stable_results
        if task.get_config("num_fewshot") is None:
            task.set_config(key="num_fewshot", value=0)
        task.set_fewshot_seed(seed=fewshot_seed)


def _build_subset_manifest(
    loaded_by_dataset: dict[str, dict[str, Any]],
    seed: int,
    dataset_order: tuple[str, ...] = OOD_DATASET_ORDER,
    protocol: str = ood_protocol.PAPER,
) -> dict[str, Any]:
    manifest_datasets: dict[str, Any] = {}
    total_samples = 0
    for dataset in dataset_order:
        spec = OOD_TASK_SPECS[dataset]
        if dataset == 'NQ':
            from opdarena.evaluation.nq_protocol import PROTOCOL, digest

            questions = loaded_by_dataset[dataset]['questions']
            manifest_datasets[dataset] = {
                'lm_eval_name': 'nq_open', 'samples_per_leaf': None,
                'leaf_tasks': {'nq_open': {'original_size': 3610, 'indices': list(range(3610))}},
                'questions_sha256': digest(questions), 'protocol': PROTOCOL,
            }
            total_samples += len(questions['questions'])
            continue
        tasks = loaded_by_dataset[dataset]["tasks"]
        if len(tasks) != spec.expected_leaf_tasks:
            raise ValueError(
                f"{dataset} expanded to {len(tasks)} leaf tasks, expected {spec.expected_leaf_tasks}; "
                "the pinned OOD task definition has drifted"
            )
        leaves: dict[str, Any] = {}
        for task_name, task in sorted(tasks.items()):
            original_size = len(task.eval_docs)
            indices = fixed_subset_indices(original_size, spec.samples_per_leaf, seed, dataset, task_name)
            leaves[task_name] = {"original_size": original_size, "indices": indices}
            total_samples += len(indices)
        manifest_datasets[dataset] = {
            "lm_eval_name": spec.lm_eval_name,
            "samples_per_leaf": spec.samples_per_leaf,
            "leaf_tasks": leaves,
        }
    expected = sum(ood_protocol.COUNTS[name] for name in dataset_order)
    if total_samples != expected:
        raise ValueError(f"fixed OOD suite contains {total_samples} samples, expected {expected}")
    result = {
        "schema_version": 1,
        "protocol": ood_protocol.MANIFEST_PROTOCOLS[protocol],
        "lm_eval_commit": LM_EVAL_COMMIT,
        "subset_seed": seed,
        "total_samples": total_samples,
        "datasets": manifest_datasets,
    }
    if protocol == ood_protocol.PAPER:
        result['response_budgets'] = {name: ood_protocol.response_budget(protocol, name)
                                      for name in dataset_order if name in GENERATION_DATASETS}
        result['bbh_scoring'] = 'bbh-conclusion-v1'
    return result


def _write_or_verify_manifest(path: Path, manifest: dict[str, Any]) -> None:
    serialized = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(f"existing OOD subset manifest differs from the fixed protocol: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)


def _metric_source(results: dict[str, Any], spec: OODTaskSpec) -> dict[str, Any]:
    if spec.lm_eval_name in results.get("groups", {}):
        return results["groups"][spec.lm_eval_name]
    return results["results"][spec.lm_eval_name]


def _numeric_aux_metrics(dataset: str, results: dict[str, Any], primary_metric: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for source_section in ("results", "groups"):
        for source_name, source_metrics in results.get(source_section, {}).items():
            for metric_name, value in source_metrics.items():
                if source_name == OOD_TASK_SPECS[dataset].lm_eval_name and metric_name == primary_metric:
                    continue
                if isinstance(value, bool | int | float) and np.isfinite(value):
                    key = f"val-aux/ood/{dataset}/{source_name}/{metric_name.replace('/', '_')}"
                    metrics[key] = float(value)
    return metrics


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set | tuple):
        return list(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _dump_samples(
    path: Path,
    results_by_dataset: dict[str, dict[str, Any]],
    datasets: tuple[str, ...] = OOD_DATASET_ORDER,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for dataset in datasets:
            for task_name, samples in results_by_dataset[dataset].get("samples", {}).items():
                for sample in samples:
                    scoring = {}
                    if dataset == "BBH":
                        from opdarena.evaluation.bbh import VERSION

                        scoring["answer_extraction"] = VERSION
                    stream.write(
                        json.dumps(
                            {"dataset": dataset, "task": task_name, **sample, **scoring},
                            ensure_ascii=False,
                            default=_json_default,
                        )
                        + "\n"
                    )
    temporary.replace(path)


def run_ood_validation(trainer) -> dict[str, float]:
    """Evaluate the requested part of the fixed OOD suite."""

    from lm_eval.evaluator import evaluate
    from lm_eval.tasks import TaskManager

    config = trainer.config.trainer.get("ood_validation", {})
    protocol = str(config.get('protocol', ood_protocol.PAPER))
    model_type = str(config.get("model_type", "auto"))
    if model_type == "auto":
        model_type = infer_validation_model_type(str(trainer.config.actor_rollout_ref.model.path))
    if model_type not in {"base", "instruct"}:
        raise ValueError(f"trainer.ood_validation.model_type must be base, instruct, or auto; got {model_type!r}")

    seed = int(config.get("subset_seed", 42))
    max_model_length = int(config.get("max_model_length", 32768))
    datasets = normalize_ood_datasets(config.get("datasets", ood_protocol.datasets(protocol)))
    enable_thinking = _config_bool(
        config.get("enable_thinking", False), "trainer.ood_validation.enable_thinking"
    )
    include_path = config.get("include_path")
    task_manager = TaskManager(include_path=str(include_path) if include_path else None)
    adapter = _make_actor_adapter(
        trainer, max_model_length=max_model_length, enable_thinking=enable_thinking
    )

    started_at = time.perf_counter()
    loaded_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        spec = OOD_TASK_SPECS[dataset]
        if dataset == 'NQ':
            from opdarena.evaluation.nq_protocol import load_questions

            loaded_by_dataset[dataset] = {'questions': load_questions(config.get('nq_questions'))}
            continue
        loaded = task_manager.load([spec.lm_eval_name])
        _prepare_loaded_tasks(loaded, ood_protocol.response_budget(protocol, dataset), fewshot_seed=0)
        loaded_by_dataset[dataset] = loaded

    manifest = _build_subset_manifest(
        loaded_by_dataset, seed=seed, dataset_order=datasets, protocol=protocol
    )
    validation_data_dir = trainer.config.trainer.get("validation_data_dir")
    if validation_data_dir:
        _write_or_verify_manifest(Path(validation_data_dir) / "ood_subset_manifest.json", manifest)

    metrics: dict[str, float] = {}
    primary_values: list[float] = []
    generation_mean_lengths: list[float] = []
    results_by_dataset: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        dataset_started_at = time.perf_counter()
        spec = OOD_TASK_SPECS[dataset]
        loaded = loaded_by_dataset[dataset]
        samples = {
            task_name: manifest["datasets"][dataset]["leaf_tasks"][task_name]["indices"]
            for task_name in loaded.get("tasks", {})
        }
        adapter.generation_lengths.clear()
        use_chat_wrapper = model_type == "instruct" and dataset == "IFEval"
        if dataset == 'NQ':
            from opdarena.evaluation.nq import evaluate as evaluate_nq

            results, lengths = evaluate_nq(trainer, loaded['questions']['questions'])
            adapter.generation_lengths.extend(lengths)
        else:
            results = evaluate(
                lm=adapter, task_dict=loaded, samples=samples, bootstrap_iters=0,
                log_samples=True, apply_chat_template=use_chat_wrapper,
                fewshot_as_multiturn=use_chat_wrapper,
            )
        if results is None:
            raise RuntimeError(f"lm-eval returned no result for {dataset}")
        results_by_dataset[dataset] = results
        primary_value = float(_metric_source(results, spec)[spec.primary_metric])
        primary_values.append(primary_value)
        metrics[f"val-ood/{dataset}/acc/mean"] = primary_value
        if dataset in GENERATION_DATASETS:
            if not adapter.generation_lengths:
                raise ValueError(f"generation dataset {dataset} produced no output lengths")
            mean_length = float(np.mean(adapter.generation_lengths))
            generation_mean_lengths.append(mean_length)
            metrics[f"val-ood/{dataset}/output_length_tokens/mean"] = mean_length
        metrics.update(_numeric_aux_metrics(dataset, results, spec.primary_metric))
        metrics[f"val-aux/ood/{dataset}/elapsed_seconds"] = time.perf_counter() - dataset_started_at

    if validation_data_dir:
        _dump_samples(
            Path(validation_data_dir) / f"{trainer.global_steps}.ood.jsonl",
            results_by_dataset,
            datasets=datasets,
        )

    metrics["val-ood/overall/acc/mean"] = float(np.mean(primary_values))
    metrics["val-ood/overall/elapsed_seconds"] = time.perf_counter() - started_at
    metrics["val-ood/overall/output_length_tokens/mean"] = (
        float(np.mean(generation_mean_lengths)) if generation_mean_lengths else 0.0
    )
    if 'IFEval' in datasets:
        metrics['val-aux/ood/IFEval/max_response_tokens'] = ood_protocol.response_budget(protocol, 'IFEval')
    return metrics
