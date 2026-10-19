"""Single-GPU capability evaluation for explicitly named experimental weights."""

import hashlib
import statistics
from pathlib import Path

MATH_TASKS = ("AMC23", "AIME24", "AIME25", "HMMT25", "Math500")


def math_batch_starts(questions, shard=None):
    index, count = (0, 1) if shard is None else shard
    if not 0 <= index < count:
        raise ValueError("Invalid math batch shard")
    return range(index * 8, questions, count * 8)


def math_rows(task, data_dir):
    import json

    root = Path(data_dir)
    if task == "Math500":
        from opdarena.evaluation.math500 import PROTOCOL

        rows = [
            json.loads(line)
            for line in (root / "Math500/test.jsonl").read_text().splitlines()
            if line
        ]
        return [
            {"prompt": r["problem"].strip() + PROTOCOL["prompt_suffix"], "answer": str(r["answer"])}
            for r in rows
        ]
    if task not in MATH_TASKS:
        raise ValueError("Unknown math task")
    import pyarrow.parquet as pq

    rows = pq.read_table(root / task / "test.parquet").to_pylist()
    return [
        {
            "prompt": "\n\n".join(m["content"] for m in r["prompt"]),
            "answer": r["reward_model"]["ground_truth"],
        }
        for r in rows
    ]


def summarize_math(result, target, task, rows, samples=8):
    outputs = []
    for index in range(0, len(rows), 8):
        batch = result.get("accuracy/" + target + "/" + task + "/batch-" + str(index))
        expected = rows[index : index + 8]
        if batch is None or len(batch) != len(expected) * samples:
            raise ValueError("Missing or incomplete math batch: " + task + "/" + str(index))
        for offset, row in enumerate(expected):
            if any(
                r["prompt"] != row["prompt"] or str(r["answer"]) != str(row["answer"])
                for r in batch[offset * samples : (offset + 1) * samples]
            ):
                raise ValueError("Saved math batch has different questions")
        outputs.extend(batch)
    return {
        "accuracy": statistics.mean(r["accuracy"] for r in outputs),
        "output_length_tokens": statistics.mean(len(r["tokens"]) for r in outputs),
        "questions": len(rows),
        "responses": len(outputs),
        "budget": 16384 if task == "Math500" else 31744,
        "responses_per_question": samples,
    }


def evaluate(
    model,
    tokenizer_path,
    result,
    target,
    plan,
    include_tasks=None,
    ifeval_generation_tokens=2048,
    math500_samples=8,
    math_batch_shard=None,
):
    if ifeval_generation_tokens not in (512, 2048):
        raise ValueError("Unsupported IFEval generation budget")
    if math500_samples not in (4, 8):
        raise ValueError("Unsupported Math500 sample count")
    if math_batch_shard is not None:
        math_batch_starts(0, math_batch_shard)
    import json

    from lm_eval.api.model import TemplateLM
    from lm_eval.evaluator import evaluate as harness_evaluate
    from verl.trainer.ppo.ood_validation import OOD_TASK_SPECS, _metric_source
    from verl.utils.reward_score.ttrl_math import reward_func
    from vllm import LLM, SamplingParams

    from opdarena.evaluation.likelihood import encode_pair
    from opdarena.evaluation.nq_protocol import PROTOCOL as NQ
    from opdarena.evaluation.nq_protocol import exact_match, load_questions
    from opdarena.evaluation.tasks import load_tasks

    llm = LLM(
        model=str(model),
        tokenizer=tokenizer_path,
        tensor_parallel_size=plan.get("tensor_parallel_size", 1),
        dtype="bfloat16",
        max_model_len=32768,
        gpu_memory_utilization=plan.get("gpu_memory_utilization", 0.80),
        max_num_seqs=plan.get("max_sequences", 4),
        max_num_batched_tokens=4096,
        enable_chunked_prefill=True,
        enable_prefix_caching=False,
        enforce_eager=True,
        seed=0,
        generation_config="vllm",
    )
    try:
        tok = llm.get_tokenizer()
        metrics = {}
        stops = sorted(
            {
                tok.convert_tokens_to_ids(t)
                for t in ("<|endoftext|>", "<|im_end|>")
                if t in tok.get_vocab()
            }
        )
        for task in MATH_TASKS:
            if include_tasks is not None and task not in include_tasks:
                continue
            name = "accuracy/" + target + "/" + task
            saved = result.get(name)
            if saved is None:
                rows = math_rows(task, plan["data_dir"])
                budget = 16384 if task == "Math500" else 31744
                samples = math500_samples if task == "Math500" else 8
                for index in math_batch_starts(len(rows), math_batch_shard):
                    key = name + "/batch-" + str(index)
                    batch = result.get(key)
                    if batch is None:
                        prompts = [
                            tok.encode(r["prompt"], add_special_tokens=False)
                            for r in rows[index : index + 8]
                        ]
                        if any(len(ids) > 1024 for ids in prompts):
                            raise ValueError("Math prompt exceeds formal budget")
                        generated = llm.generate(
                            [{"prompt_token_ids": ids} for ids in prompts],
                            SamplingParams(
                                n=samples,
                                temperature=0.7,
                                top_p=0.95,
                                top_k=-1,
                                max_tokens=budget,
                                stop_token_ids=stops,
                            ),
                            use_tqdm=False,
                        )
                        batch = []
                        for row, out in zip(rows[index : index + 8], generated, strict=True):
                            if len(out.outputs) != samples:
                                raise ValueError("Missing math samples")
                            for answer in out.outputs:
                                grade = reward_func(
                                    "MATH500" if task == "Math500" else task,
                                    answer.text,
                                    str(row["answer"]),
                                )
                                batch.append(
                                    {
                                        "prompt": row["prompt"],
                                        "answer": row["answer"],
                                        "response": answer.text,
                                        "tokens": list(answer.token_ids),
                                        "accuracy": float(grade["acc"]),
                                    }
                                )
                        result.put(key, batch, immutable=True)
                        result.publish()
                        print(
                            json.dumps(
                                {
                                    "event": "accuracy_progress",
                                    "key": target,
                                    "dataset": task,
                                    "questions": min(index + 8, len(rows)),
                                    "total_questions": len(rows),
                                }
                            ),
                            flush=True,
                        )
                    if len(batch) != len(rows[index : index + 8]) * samples:
                        raise ValueError("Saved math batch has a different sample count")
                if math_batch_shard is not None:
                    continue
                saved = summarize_math(result, target, task, rows, samples)
                result.put(name, saved, immutable=True)
                result.publish()
            metrics[task] = saved
            print(
                json.dumps(
                    {
                        "event": "accuracy_task_complete",
                        "key": target,
                        "dataset": task,
                        "accuracy": saved["accuracy"],
                        "questions": saved["questions"],
                    }
                ),
                flush=True,
            )

        class Adapter(TemplateLM):
            def __init__(self):
                super().__init__()
                self.outputs = []

            @property
            def eot_token_id(self):
                return tok.eos_token_id

            @property
            def prefix_token_id(self):
                return tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id

            @property
            def max_length(self):
                return 32768

            @property
            def tokenizer_name(self):
                return tok.name_or_path.replace("/", "__")

            def tok_encode(self, text, add_special_tokens=False, **kwargs):
                return tok.encode(text, add_special_tokens=add_special_tokens, **kwargs)

            def apply_chat_template(self, *args, **kwargs):
                raise ValueError("Raw evaluation")

            def _loglikelihood_tokens(self, *args, **kwargs):
                raise NotImplementedError

            def loglikelihood_rolling(self, *args, **kwargs):
                raise NotImplementedError

            def loglikelihood(self, requests, **kwargs):
                values = []
                for start in range(0, len(requests), 32):
                    batch = requests[start : start + 32]
                    encoded = [
                        encode_pair(tok, *request.args, max_length=32767) for request in batch
                    ]
                    outputs = llm.generate(
                        [{"prompt_token_ids": ids} for ids, _, _ in encoded],
                        SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1),
                        use_tqdm=False,
                    )
                    for request, (_, positions, targets), output in zip(
                        batch, encoded, outputs, strict=True
                    ):
                        selected = [
                            output.prompt_logprobs[p + 1][t]
                            for p, t in zip(positions, targets, strict=True)
                        ]
                        values.append(
                            (sum(x.logprob for x in selected), all(x.rank == 1 for x in selected))
                        )
                        context, continuation = request.args
                        self.outputs.append(
                            {
                                "context": context,
                                "continuation": continuation,
                                "loglikelihood": values[-1][0],
                            }
                        )
                return values

            def generate_until(self, requests, **kwargs):
                texts = []
                for start in range(0, len(requests), 32):
                    batch = requests[start : start + 32]
                    inputs = []
                    params = []
                    for request in batch:
                        context, options = request.args
                        until = options.get("until", [])
                        until = [until] if isinstance(until, str) else until
                        inputs.append(
                            {
                                "prompt_token_ids": tok.encode(context, add_special_tokens=False)[
                                    -(32768 - budget) :
                                ]
                            }
                        )
                        params.append(
                            SamplingParams(
                                temperature=0.7,
                                top_p=0.95,
                                top_k=-1,
                                max_tokens=budget,
                                stop=until,
                                seed=0,
                            )
                        )
                    generated = llm.generate(inputs, params, use_tqdm=False)
                    for request, out in zip(batch, generated, strict=True):
                        answer = out.outputs[0]
                        ids = list(answer.token_ids)
                        scored = (
                            tok.decode(ids[:512], skip_special_tokens=True)
                            if task == "IFEval"
                            else answer.text
                        )
                        texts.append(scored)
                        self.outputs.append(
                            {
                                "prompt": request.args[0],
                                "response": answer.text,
                                "scored_response": scored,
                                "tokens": ids,
                                "scored_tokens": min(len(ids), 512)
                                if task == "IFEval"
                                else len(ids),
                            }
                        )
                return texts

        for task in ("HellaSwag", "ARC-Challenge", "MMLU", "IFEval", "TriviaQA", "BBH"):
            if include_tasks is not None and task not in include_tasks:
                continue
            budget = {"IFEval": ifeval_generation_tokens, "TriviaQA": 256, "BBH": 1024}.get(
                task, 2048
            )
            name = "accuracy/" + target + "/" + task
            saved = result.get(name)
            if saved is None:
                loaded, samples, manifest = load_tasks(task, budget)
                adapter = Adapter()
                output = harness_evaluate(
                    lm=adapter,
                    task_dict=loaded,
                    samples=samples,
                    bootstrap_iters=0,
                    log_samples=True,
                    apply_chat_template=False,
                    fewshot_as_multiturn=False,
                )
                saved = {
                    "accuracy": float(
                        _metric_source(output, OOD_TASK_SPECS[task])[
                            OOD_TASK_SPECS[task].primary_metric
                        ]
                    ),
                    "questions": sum(map(len, samples.values())),
                }
                if task in ("IFEval", "TriviaQA", "BBH"):
                    saved["output_length_tokens"] = statistics.mean(
                        r["scored_tokens"] for r in adapter.outputs
                    )
                if task == "IFEval" and ifeval_generation_tokens == 512:
                    saved["max_response_tokens"] = 512
                result.put(
                    name + "/outputs",
                    {
                        "manifest": manifest,
                        "samples": output["samples"],
                        "outputs": adapter.outputs,
                    },
                    immutable=True,
                )
                result.put(name, saved, immutable=True)
                result.publish()
            metrics[task] = saved
            print(
                json.dumps(
                    {
                        "event": "accuracy_task_complete",
                        "key": target,
                        "dataset": task,
                        "accuracy": saved["accuracy"],
                        "questions": saved["questions"],
                    }
                ),
                flush=True,
            )
        if include_tasks is not None and "NQ" not in include_tasks:
            return metrics
        name = "accuracy/" + target + "/NQ"
        saved = result.get(name)
        if saved is None:
            questions = load_questions(plan["nq_questions"])["questions"]
            outputs = []
            for start in range(0, len(questions), 64):
                key = name + "/batch-" + str(start)
                batch = result.get(key)
                if batch is None:
                    selection = questions[start : start + 64]
                    generated = llm.generate(
                        [
                            {"prompt_token_ids": tok.encode(q["prompt"], add_special_tokens=False)}
                            for q in selection
                        ],
                        [
                            SamplingParams(
                                temperature=0.7,
                                top_p=0.95,
                                top_k=-1,
                                max_tokens=256,
                                stop=NQ["stop"],
                                seed=int(
                                    hashlib.sha256(("0:" + q["id"]).encode()).hexdigest()[:8], 16
                                )
                                % 2**31,
                            )
                            for q in selection
                        ],
                        use_tqdm=False,
                    )
                    batch = [
                        {
                            "id": q["id"],
                            "prompt": q["prompt"],
                            "answers": q["answers"],
                            "response": o.outputs[0].text,
                            "tokens": list(o.outputs[0].token_ids),
                            "accuracy": exact_match(o.outputs[0].text, q["answers"]),
                        }
                        for q, o in zip(selection, generated, strict=True)
                    ]
                    result.put(key, batch, immutable=True)
                    result.publish()
                    print(
                        json.dumps(
                            {
                                "event": "accuracy_progress",
                                "key": target,
                                "dataset": "NQ",
                                "questions": min(start + 64, len(questions)),
                                "total_questions": len(questions),
                            }
                        ),
                        flush=True,
                    )
                outputs.extend(batch)
            saved = {
                "accuracy": statistics.mean(r["accuracy"] for r in outputs),
                "questions": len(outputs),
                "output_length_tokens": statistics.mean(len(r["tokens"]) for r in outputs),
            }
            result.put(name, saved, immutable=True)
            result.publish()
        metrics["NQ"] = saved
        return metrics
    finally:
        # Release the pinned vLLM engine before interpreter teardown.
        llm.llm_engine.engine_core.shutdown()
