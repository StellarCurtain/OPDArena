"""Create synthetic inputs for a short GPU integration check."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    root = args.output
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    words = [
        "<pad>",
        "<eos>",
        "<unk>",
        "What",
        "is",
        "1",
        "+",
        "2",
        "?",
        "The",
        "answer",
        "A",
        "B",
        "C",
        "D",
        r"\boxed{2}",
    ]
    vocab = {word: i for i, word in enumerate(words)}
    vocab.update({f"token{i}": i for i in range(len(vocab), 128)})
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>"
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message.content }} {% endfor %}"
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32768,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    for seed, name in [(11, "base"), (12, "teacher")]:
        torch.manual_seed(seed)
        model = Qwen2ForCausalLM(config)
        model.save_pretrained(root / name)
        tokenizer.save_pretrained(root / name)
    rows = [
        {
            "data_source": "openr1",
            "prompt": [{"role": "user", "content": "What is 1 + 1 ?"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": "2"},
            "extra_info": {"index": i, "answer": r"The answer is \boxed{2}"},
        }
        for i in range(16)
    ]
    pq.write_table(pa.Table.from_pylist(rows), root / "train.parquet")
    print(root)


if __name__ == "__main__":
    main()
