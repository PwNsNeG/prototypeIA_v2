#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer

from model import TinyGPT, GPTConfig


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_np_dtype(dtype_name: str):
    mapping = {
        "uint16": np.uint16,
        "uint32": np.uint32,
        "int32": np.int32,
        "int64": np.int64,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype in meta.json: {dtype_name}")
    return mapping[dtype_name]


def build_model_from_checkpoint(ckpt: dict[str, Any], device: str):
    if "model_config" not in ckpt:
        raise KeyError(
            "Checkpoint does not contain 'model_config'. "
            "Your training script must save it for v2."
        )

    config = GPTConfig(**ckpt["model_config"])
    model = TinyGPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def make_batch(data: np.memmap, block_size: int, batch_size: int, device: str):
    if len(data) <= block_size + 1:
        raise ValueError(
            f"Validation data too small for block_size={block_size}. "
            f"len(data)={len(data)}"
        )

    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i + block_size].astype(np.int64).copy())
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64).copy())
        for i in ix
    ])

    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate_loss(model, val_data, block_size: int, batch_size: int, eval_iters: int, device: str):
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = make_batch(val_data, block_size, batch_size, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    return losses.mean().item()


@torch.no_grad()
def evaluate_accuracy(model, val_data, block_size: int, batch_size: int, acc_iters: int, device: str):
    correct = 0
    total = 0

    for _ in range(acc_iters):
        x, y = make_batch(val_data, block_size, batch_size, device)
        logits, _ = model(x, y)
        preds = logits.argmax(dim=-1)
        correct += (preds == y).sum().item()
        total += y.numel()

    return (100.0 * correct / total) if total else 0.0


@torch.no_grad()
def sample_text(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    max_new: int,
    temperature: float,
    top_k: int | None,
):
    prompt_ids = tokenizer.encode(prompt).ids
    if not prompt_ids:
        raise ValueError(f"Prompt produced no tokens: {prompt!r}")

    ctx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    out = model.generate(
        ctx,
        max_new_tokens=max_new,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(out[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Evaluate a tokenizer-based TinyGPT model.")
    parser.add_argument("--ckpt", default="checkpoints/best.pt", help="Path to checkpoint")
    parser.add_argument("--tokenizer", default="data/tokenizer/bpe16k/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--data-dir", default="data/tokenized/corpus_v1_bpe16k", help="Directory containing val.bin and meta.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--acc-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new", type=int, default=120)
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    tokenizer_path = Path(args.tokenizer)
    data_dir = Path(args.data_dir)
    meta_path = data_dir / "meta.json"
    val_bin_path = data_dir / "val.bin"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    if not val_bin_path.exists():
        raise FileNotFoundError(f"val.bin not found: {val_bin_path}")

    print(f"Device          : {args.device}")
    print(f"Checkpoint      : {ckpt_path}")
    print(f"Tokenizer       : {tokenizer_path}")
    print(f"Tokenized data  : {data_dir}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    meta = load_json(meta_path)

    block_size = int(meta["block_size"])
    vocab_size = int(meta["vocab_size"])
    dtype_name = meta["dtype"]
    np_dtype = get_np_dtype(dtype_name)

    val_data = np.memmap(val_bin_path, dtype=np_dtype, mode="r")
    ckpt = torch.load(ckpt_path, map_location=args.device)

    model, config = build_model_from_checkpoint(ckpt, args.device)

    print(f"\nCheckpoint step : {ckpt.get('step', 'n/a')}")
    if "val_loss" in ckpt:
        print(f"Saved val loss  : {ckpt['val_loss']:.4f}")
        print(f"Saved ppl       : {math.exp(ckpt['val_loss']):.2f}")

    print(f"Block size      : {block_size}")
    print(f"Vocab size      : {vocab_size}")
    print(f"val.bin tokens  : {len(val_data):,}")
    print(f"Model config    : {config}")

    print(f"\nEvaluating over {args.eval_iters} batches...")
    mean_loss = evaluate_loss(
        model=model,
        val_data=val_data,
        block_size=block_size,
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
        device=args.device,
    )
    perplexity = math.exp(mean_loss)

    print(f"Val loss        : {mean_loss:.4f}")
    print(f"Perplexity      : {perplexity:.2f}")

    print(f"\nPerplexity interpretation:")
    print(f"  Uniform random: {vocab_size}")
    print(f"  Your model    : {perplexity:.2f}")
    print(f"  Improvement   : {vocab_size / perplexity:.1f}x better than uniform random")

    acc = evaluate_accuracy(
        model=model,
        val_data=val_data,
        block_size=block_size,
        batch_size=args.batch_size,
        acc_iters=args.acc_iters,
        device=args.device,
    )
    print(f"\nPer-token accuracy:")
    print(f"  Top-1 accuracy: {acc:.1f}%")

    prompts = [
        "Monsieur,",
        "Je ne sais pas",
        "Le théâtre représente",
    ]

    print(f"\n--- Samples at different temperatures ---\n")
    for temp in [0.7, 0.9, 1.0]:
        print(f"Temperature {temp}:")
        for prompt in prompts:
            out = sample_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=args.device,
                max_new=args.max_new,
                temperature=temp,
                top_k=40,
            )
            print(f"  Prompt: {repr(prompt)}")
            print(f"  Output: {repr(out)}")
        print()


if __name__ == "__main__":
    main()
