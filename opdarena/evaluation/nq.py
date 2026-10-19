"""NQ-Open evaluation using the live actor and the frozen backfill protocol."""

import hashlib

import numpy as np

from opdarena.evaluation.nq_protocol import PROTOCOL, exact_match, question_seed


def evaluate(trainer, questions):
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
    from verl.trainer.ppo.ood_validation import _prompt_batch

    tokenizer = trainer.tokenizer
    world_size = trainer.actor_rollout_wg.world_size
    rows, lengths = [], []
    for start in range(0, len(questions), 8 * world_size):
        batch = questions[start : start + 8 * world_size]
        tokens = [tokenizer.encode(q["prompt"], add_special_tokens=False) for q in batch]
        if any(len(ids) + 256 > 4096 for ids in tokens):
            raise ValueError("NQ prompt exceeds the frozen context budget")
        prompts = _prompt_batch(tokens, tokenizer.pad_token_id)
        prompts.non_tensor_batch["validation_seed"] = np.array(
            [question_seed(q["id"]) for q in batch]
        )
        prompts.meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
            "validate": True,
            "global_steps": trainer.global_steps,
            "preserve_generation_outputs": True,
            "validation_sampling_params": {
                "n": 1,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": -1,
                "max_tokens": 256,
                "repetition_penalty": 1.0,
                "stop": PROTOCOL["stop"],
                "detokenize": True,
                "skip_special_tokens": True,
            },
        }
        padded, count = pad_dataproto_to_divisor(prompts, world_size)
        out = unpad_dataproto(trainer.actor_rollout_wg.generate_sequences(padded), count)
        for index, q in enumerate(batch):
            # vLLM's text excludes stop strings; decoding token_ids alone can retain them.
            text = out.non_tensor_batch["validation_response_text"][index]
            length = int(out.non_tensor_batch["validation_response_length"][index])
            ids = out.batch["responses"][index, :length].cpu().tolist()
            lengths.append(length)
            rows.append(
                {
                    "doc_id": q["index"],
                    "doc_hash": q["id"],
                    "prompt_hash": hashlib.sha256(q["prompt"].encode()).hexdigest(),
                    "doc": {"question": q["question"], "answer": q["answers"]},
                    "question_id": q["id"],
                    "prompt": q["prompt"],
                    "answers": q["answers"],
                    "resps": [[text]],
                    "response_token_ids": ids,
                    "output_length_tokens": length,
                    "exact_match": exact_match(text, q["answers"]),
                    "sampling_seed": question_seed(q["id"]),
                    "protocol": PROTOCOL["version"],
                }
            )
    accuracy = sum(r["exact_match"] for r in rows) / len(rows)
    return {
        "results": {"nq_open": {"exact_match,none": accuracy}},
        "samples": {"nq_open": rows},
    }, lengths
