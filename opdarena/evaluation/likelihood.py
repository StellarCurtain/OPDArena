"""Token boundaries for likelihood evaluation."""


def encode_pair(tokenizer, context, continuation, max_length=8192):
    spaces = len(context) - len(context.rstrip())
    if spaces:
        continuation = context[-spaces:] + continuation
        context = context[:-spaces]
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    whole = tokenizer.encode(context + continuation, add_special_tokens=False)
    continuation_ids = whole[len(context_ids) :]
    if not continuation_ids or len(continuation_ids) >= max_length:
        raise ValueError("Invalid continuation length")
    context_ids = context_ids[-(max_length - len(continuation_ids)) :]
    if not context_ids:
        context_ids = [tokenizer.eos_token_id]
    ids = context_ids + continuation_ids
    positions = list(range(len(context_ids) - 1, len(ids) - 1))
    return ids, positions, continuation_ids
