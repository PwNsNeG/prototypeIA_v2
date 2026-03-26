#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
from tqdm import tqdm


def iter_texts(jsonl_path: Path, limit_docs: int | None = None):
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text", "")
            if not text or not isinstance(text, str):
                continue

            yield text
            count += 1

            if limit_docs is not None and count >= limit_docs:
                return


def count_docs_and_chars(jsonl_path: Path):
    docs = 0
    chars = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text", "")
            if not text or not isinstance(text, str):
                continue
            docs += 1
            chars += len(text)
    return docs, chars


def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer from filtered JSONL corpus.")
    parser.add_argument("--input", default="data/prepared/corpus_v1/train.jsonl", help="Input JSONL corpus")
    parser.add_argument("--out-dir", default="data/tokenizer/bpe16k", help="Output directory")
    parser.add_argument("--vocab-size", type=int, default=16000, help="Tokenizer vocab size")
    parser.add_argument("--min-frequency", type=int, default=2, help="Minimum token frequency")
    parser.add_argument("--limit-docs", type=int, default=None, help="Optional limit for quick tests")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    docs, chars = count_docs_and_chars(input_path)
    print(f"Input corpus : {input_path}")
    print(f"Documents    : {docs:,}")
    print(f"Characters   : {chars:,}")
    if args.limit_docs is not None:
        print(f"Limit docs    : {args.limit_docs:,}")
    print(f"Output dir    : {out_dir}")
    print(f"Vocab size    : {args.vocab_size:,}")
    print(f"Min frequency : {args.min_frequency}")
    print()

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFC(),
    ])

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = [
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]",
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    print("Training tokenizer...")
    tokenizer.train_from_iterator(
        iter_texts(input_path, limit_docs=args.limit_docs),
        trainer=trainer,
    )

    tokenizer_json_path = out_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json_path))

    vocab_path = out_dir / "vocab.json"
    merges_path = out_dir / "merges.txt"
    tokenizer.model.save(str(out_dir), prefix="")

    # Rename generated files if needed for consistency
    generated_vocab = out_dir / "vocab.json"
    generated_merges = out_dir / "merges.txt"

    config = {
        "type": "BPE",
        "vocab_size": tokenizer.get_vocab_size(),
        "min_frequency": args.min_frequency,
        "special_tokens": special_tokens,
        "input": str(input_path),
        "documents": docs,
        "characters": chars,
    }

    config_path = out_dir / "tokenizer_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Quick sanity test
    sample = "Monsieur, je ne sais pas si le théâtre représente bien la vérité."
    encoded = tokenizer.encode(sample)
    decoded = tokenizer.decode(encoded.ids)

    print("\nSanity check:")
    print(f"Sample text   : {sample}")
    print(f"Token count   : {len(encoded.ids)}")
    print(f"First IDs     : {encoded.ids[:20]}")
    print(f"Decoded       : {decoded}")

    print("\nSaved:")
    print(f"  {tokenizer_json_path}")
    print(f"  {generated_vocab}")
    print(f"  {generated_merges}")
    print(f"  {config_path}")
    print("\nTokenizer ready.")
    print(f"Final vocab size: {tokenizer.get_vocab_size():,}")


if __name__ == "__main__":
    main()
