"""JS and forward/reverse KL at fixed prefixes, loading one model at a time."""

import argparse
import gc
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from opdarena.utils.provenance import model_signature as signature


def predictions(path, tokenizer, rows, device, cache, bank_hash):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    other = AutoTokenizer.from_pretrained(path)
    vocab = tokenizer.get_vocab()
    ov = other.get_vocab()
    if any(ov.get(t) != i for t, i in vocab.items()):
        raise ValueError("Reference tokenizer does not preserve the student token IDs")
    valid = sorted(set(vocab.values()))
    stamp = hashlib.sha256(
        json.dumps([signature(path), bank_hash, valid, device], sort_keys=True).encode()
    ).hexdigest()
    dest = cache / (stamp + ".pt")
    if dest.exists():
        return torch.load(dest, map_location="cpu", weights_only=True)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = (
        AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa")
        .to(device)
        .eval()
    )
    result = []
    with torch.inference_mode():
        for row in rows:
            ids = torch.tensor([row["input_ids"]], device=device)
            if (
                ids.ndim != 2
                or ids.shape[-1] == 0
                or any(not 0 <= p < ids.shape[-1] for p in row["positions"])
            ):
                raise ValueError("Invalid prefix positions")
            logits = (
                model(input_ids=ids, use_cache=False).logits[0, row["positions"]].float()[:, valid]
            )
            result.append(logits.log_softmax(-1).cpu())
    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    torch.save(result, dest)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Current student in HF format")
    p.add_argument("--base", required=True, help="The released student Base")
    p.add_argument("--teacher", help="Optional actual frozen teacher endpoint")
    p.add_argument(
        "--teacher-base", help="Construct teacher=normalize(student Base * teacher / teacher Base)"
    )
    p.add_argument("--bank", type=Path, required=True, help="Shared fixed-prefix JSON bank")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = p.parse_args()
    if args.teacher_base and not args.teacher:
        p.error("--teacher-base also requires --teacher")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    if args.device == "cuda" and len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")) != 1:
        p.error("Expose one GPU")
    import torch
    from transformers import AutoTokenizer

    from opdarena.analysis.distances import distances
    from opdarena.evaluate import TASKS

    document = json.loads(args.bank.read_text())
    rows = document["rows"] if isinstance(document, dict) else document
    if not rows:
        raise ValueError("Empty prefix bank")
    if isinstance(document, dict) and document.get("tokenizer_vocab_sha256"):
        check = hashlib.sha256(
            json.dumps(
                AutoTokenizer.from_pretrained(args.base).get_vocab(), sort_keys=True
            ).encode()
        ).hexdigest()
        if check != document["tokenizer_vocab_sha256"]:
            raise ValueError("Bank uses different student token IDs")
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "probabilities"
    cache.mkdir(exist_ok=True)
    bank_hash = hashlib.sha256(args.bank.read_bytes()).hexdigest()
    tok = AutoTokenizer.from_pretrained(args.base)
    current = predictions(args.model, tok, rows, args.device, cache, bank_hash)
    base = predictions(args.base, tok, rows, args.device, cache, bank_hash)
    refs = {"Base": base}
    if args.teacher:
        target = predictions(args.teacher, tok, rows, args.device, cache, bank_hash)
        if args.teacher_base:
            source = predictions(args.teacher_base, tok, rows, args.device, cache, bank_hash)
            target = [b + t - s for b, t, s in zip(base, target, source, strict=True)]
            target = [x - torch.logsumexp(x, dim=-1, keepdim=True) for x in target]
        refs["Teacher"] = target
    measured = []
    summary = {}
    for name, ref in refs.items():
        groups = defaultdict(list)
        for row, a, b in zip(rows, ref, current, strict=True):
            d = distances(a, b)
            v = {
                "js": d["js"],
                "kl_reference_current": d["kl_first_second"],
                "kl_current_reference": d["kl_second_first"],
                "top1_agreement": d["top1_agreement"],
            }
            record = {
                "reference": name,
                "task": row["task"],
                "prefix_source": row.get("prefix_source", "Base"),
                "positions": row["positions"],
                "metrics": v,
            }
            measured.append(record)
            groups[row["task"], record["prefix_source"]].append(v)
        # Equal prefix weights within each source, equal source weights, then equal dataset weights.
        bytask = defaultdict(list)
        for (task, source), vals in groups.items():
            bytask[task].append({k: sum(v[k] for v in vals) / len(vals) for k in vals[0]})
        tasks = {
            t: {k: sum(v[k] for v in vals) / len(vals) for k in vals[0]}
            for t, vals in bytask.items()
        }
        summary[name] = {"tasks": tasks}
        for label, names in [("math5", TASKS[:5]), ("ood7", TASKS[5:])]:
            if all(t in tasks for t in names):
                summary[name][label] = {
                    k: sum(tasks[t][k] for t in names) / len(names)
                    for k in next(iter(tasks.values()))
                }
    artifact = {
        "bank_sha256": bank_hash,
        "models": {"current": signature(args.model), "Base": signature(args.base)},
        "references": summary,
        "prefix_results": measured,
    }
    if args.teacher:
        artifact["models"]["Teacher"] = signature(args.teacher)
    if args.teacher_base:
        artifact["models"]["teacher_Base"] = signature(args.teacher_base)
        artifact["teacher_construction"] = "normalize(student_Base * teacher / teacher_Base)"
    (args.output / "distances.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
