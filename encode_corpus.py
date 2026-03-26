#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


def iter_jsonl_texts(jsonl_path: Path) -> Iterator[str]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text", "")
            if text and isinstance(text, str):
                yield text


def count_docs(jsonl_path: Path) -> int:
    count = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def choose_dtype(vocab_size: int) -> tuple[np.dtype, str]:
    if vocab_size <= np.iinfo(np.uint16).max:
        return np.uint16, "uint16"
    if vocab_size <= np.iinfo(np.uint32).max:
        return np.uint32, "uint32"
    raise ValueError(f"Vocab size too large for uint32: {vocab_size}")


def encode_split(
    jsonl_path: Path,
    out_bin_path: Path,
    tokenizer: Tokenizer,
    np_dtype,
    append_eos: bool,
    eos_id: int | None,
) -> dict:
    total_docs = count_docs(jsonl_path)
    docs_written = 0
    total_tokens = 0
    total_chars = 0
    empty_docs = 0

    with out_bin_path.open("wb") as fout:
        for text in tqdm(iter_jsonl_texts(jsonl_path), total=total_docs, desc=f"Encoding {jsonl_path.name}", unit="doc"):
            docs_written += 1
            total_chars += len(text)

            ids = tokenizer.encode(text).ids
            if not ids:
                empty_docs += 1
                continue

            if append_eos and eos_id is not None:
                ids.append(eos_id)

            arr = np.asarray(ids, dtype=np_dtype)
            arr.tofile(fout)
            total_tokens += int(arr.size)

    return {
        "jsonl_path": str(jsonl_path),
        "bin_path": str(out_bin_path),
        "docs_seen": total_docs,
        "docs_written": docs_written,
        "empty_docs": empty_docs,
        "characters": total_chars,
        "tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="Encode filtered JSONL corpus into train.bin / val.bin")
    parser.add_argument("--train-jsonl", default="data/prepared/corpus_v1/train.jsonl")
    parser.add_argument("--val-jsonl", default="data/prepared/corpus_v1/val.jsonl")
    parser.add_argument("--tokenizer", default="data/tokenizer/bpe16k/tokenizer.json")
    parser.add_argument("--out-dir", default="data/tokenized/corpus_v1_bpe16k")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--append-eos", action="store_true", default=True)
    parser.add_argument("--no-append-eos", dest="append_eos", action="store_false")
    args = parser.parse_args()

    train_jsonl = Path(args.train_jsonl).expanduser().resolve()
    val_jsonl = Path(args.val_jsonl).expanduser().resolve()
    tokenizer_path = Path(args.tokenizer).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_jsonl.exists():
        raise FileNotFoundError(f"Train JSONL not found: {train_jsonl}")
    if not val_jsonl.exists():
        raise FileNotFoundError(f"Val JSONL not found: {val_jsonl}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    vocab = tokenizer.get_vocab()
    vocab_size = tokenizer.get_vocab_size()
    np_dtype, dtype_name = choose_dtype(vocab_size)

    eos_id = vocab.get("[EOS]")
    if args.append_eos and eos_id is None:
        print("[!] [EOS] token not found in tokenizer vocab, disabling EOS appending.")
        args.append_eos = False

    train_bin = out_dir / "train.bin"
    val_bin = out_dir / "val.bin"

    print(f"Tokenizer        : {tokenizer_path}")
    print(f"Vocab size       : {vocab_size:,}")
    print(f"Dtype            : {dtype_name}")
    print(f"Block size       : {args.block_size}")
    print(f"Append EOS       : {args.append_eos}")
    if args.append_eos:
        print(f"EOS token id     : {eos_id}")
    print(f"Output dir       : {out_dir}")
    print()

    train_stats = encode_split(
        jsonl_path=train_jsonl,
        out_bin_path=train_bin,
        tokenizer=tokenizer,
        np_dtype=np_dtype,
        append_eos=args.append_eos,
        eos_id=eos_id,
    )

    val_stats = encode_split(
        jsonl_path=val_jsonl,
        out_bin_path=val_bin,
        tokenizer=tokenizer,
        np_dtype=np_dtype,
        append_eos=args.append_eos,
        eos_id=eos_id,
    )

    meta = {
        "vocab_size": vocab_size,
        "dtype": dtype_name,
        "block_size": args.block_size,
        "append_eos": args.append_eos,
        "eos_id": eos_id if args.append_eos else None,
        "tokenizer_path": str(tokenizer_path),
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "train_tokens": train_stats["tokens"],
        "val_tokens": val_stats["tokens"],
        "train_docs": train_stats["docs_written"],
        "val_docs": val_stats["docs_written"],
    }

    meta_path = out_dir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nTrain stats:")
    print(json.dumps(train_stats, ensure_ascii=False, indent=2))
    print("\nVal stats:")
    print(json.dumps(val_stats, ensure_ascii=False, indent=2))

    print("\nSaved:")
    print(f"  {train_bin} ({train_bin.stat().st_size / (1024 * 1024):.1f} MB)")
    print(f"  {val_bin} ({val_bin.stat().st_size / (1024 * 1024):.1f} MB)")
    print(f"  {meta_path}")
    print("\nEncoding complete.")


if __name__ == "__main__":
    main()
