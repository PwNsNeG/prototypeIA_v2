#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import Tokenizer

from model import TinyGPT, GPTConfig


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def make_batch(data: np.memmap, block_size: int, batch_size: int, device: str):
    if len(data) <= block_size + 1:
        raise ValueError(
            f"Dataset too small for block_size={block_size}. len(data)={len(data)}"
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
def estimate_loss(model, train_data, val_data, block_size: int, batch_size: int, eval_iters: int, device: str):
    model.eval()
    out = {}

    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = make_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()

    model.train()
    return out


@torch.no_grad()
def sample_text(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    max_new: int = 120,
    temperature: float = 0.8,
    top_k: int | None = 40,
):
    model.eval()
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
    text = tokenizer.decode(out[0].tolist())
    model.train()
    return text


def get_lr(step: int, learning_rate: float, warmup_steps: int, max_iters: int, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(1, warmup_steps)

    if step >= max_iters:
        return learning_rate * min_lr_ratio

    decay_ratio = (step - warmup_steps) / max(1, max_iters - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    min_lr = learning_rate * min_lr_ratio
    return min_lr + coeff * (learning_rate - min_lr)


def set_lr(optimizer, lr: float):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    step: int,
    best_val_loss: float,
    model_config: GPTConfig,
    train_config: dict[str, Any],
):
    ckpt = {
        "step": step,
        "val_loss": best_val_loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "model_config": asdict(model_config),
        "train_config": train_config,
    }
    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser(description="Train TinyGPT v2 on tokenized corpus.")
    parser.add_argument("--data-dir", default="data/tokenized/corpus_v1_bpe16k")
    parser.add_argument("--tokenizer", default="data/tokenizer/bpe16k/tokenizer.json")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--resume", action="store_true")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-iters", type=int, default=50000)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-iters", type=int, default=50)

    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--sample-prompt", default="Monsieur,")
    parser.add_argument("--sample-max-new", type=int, default=120)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(cpu_count)
    torch.set_num_interop_threads(cpu_count)

    data_dir = Path(args.data_dir)
    tokenizer_path = Path(args.tokenizer)
    checkpoints_dir = Path(args.checkpoints)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    meta_path = data_dir / "meta.json"
    train_bin_path = data_dir / "train.bin"
    val_bin_path = data_dir / "val.bin"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found: {meta_path}")
    if not train_bin_path.exists():
        raise FileNotFoundError(f"train.bin not found: {train_bin_path}")
    if not val_bin_path.exists():
        raise FileNotFoundError(f"val.bin not found: {val_bin_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    meta = load_json(meta_path)

    vocab_size = int(meta["vocab_size"])
    block_size = int(meta["block_size"])
    np_dtype = get_np_dtype(meta["dtype"])

    train_data = np.memmap(train_bin_path, dtype=np_dtype, mode="r")
    val_data = np.memmap(val_bin_path, dtype=np_dtype, mode="r")

    model_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
    )

    model = TinyGPT(model_config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_config = {
        "batch_size": args.batch_size,
        "max_iters": args.max_iters,
        "eval_interval": args.eval_interval,
        "eval_iters": args.eval_iters,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "device": args.device,
        "seed": args.seed,
        "tokenizer": str(tokenizer_path),
        "data_dir": str(data_dir),
    }

    latest_ckpt = checkpoints_dir / "latest.pt"
    best_ckpt = checkpoints_dir / "best.pt"
    config_dump = checkpoints_dir / "run_config.json"

    start_step = 0
    best_val_loss = float("inf")

    if args.resume:
        if not latest_ckpt.exists():
            raise FileNotFoundError(f"--resume set, but checkpoint not found: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt["step"]) + 1
        best_val_loss = float(ckpt["val_loss"])
        print(f"Resumed from {latest_ckpt} at step {start_step} (best val loss {best_val_loss:.4f})")

    save_json(
        config_dump,
        {
            "model_config": asdict(model_config),
            "train_config": train_config,
            "meta": meta,
        },
    )

    total_params = sum(p.numel() for p in model.parameters())

    print(f"CPU cores        : {cpu_count}")
    print(f"Torch threads    : {torch.get_num_threads()}")
    print(f"Device           : {args.device}")
    print(f"Tokenizer        : {tokenizer_path}")
    print(f"Data dir         : {data_dir}")
    print(f"Train tokens     : {len(train_data):,}")
    print(f"Val tokens       : {len(val_data):,}")
    print(f"Model config     : {asdict(model_config)}")
    print(f"Parameters       : {total_params:,}")
    print(f"Training         : {args.max_iters} steps | eval every {args.eval_interval} steps")
    print("-" * 72)

    t0 = time.time()

    for step in range(start_step, args.max_iters + 1):
        lr = get_lr(
            step=step,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            max_iters=args.max_iters,
        )
        set_lr(optimizer, lr)

        if step % args.eval_interval == 0:
            losses = estimate_loss(
                model=model,
                train_data=train_data,
                val_data=val_data,
                block_size=block_size,
                batch_size=args.batch_size,
                eval_iters=args.eval_iters,
                device=args.device,
            )
            elapsed = time.time() - t0

            print(
                f"step {step:6d} | "
                f"lr {lr:.6f} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f} | "
                f"ppl {math.exp(losses['val']):.2f} | "
                f"{elapsed:.1f}s"
            )

            save_checkpoint(
                path=latest_ckpt,
                model=model,
                optimizer=optimizer,
                step=step,
                best_val_loss=min(best_val_loss, losses["val"]),
                model_config=model_config,
                train_config=train_config,
            )

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                save_checkpoint(
                    path=best_ckpt,
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    best_val_loss=best_val_loss,
                    model_config=model_config,
                    train_config=train_config,
                )
                print(f"         -> checkpoint saved ({best_ckpt}, val loss {best_val_loss:.4f})")

            try:
                preview = sample_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=args.sample_prompt,
                    device=args.device,
                    max_new=args.sample_max_new,
                    temperature=0.8,
                    top_k=40,
                )
                print(f"         -> {repr(preview)}")
            except Exception as e:
                print(f"         -> sample failed: {e}")

            print()

        if step == args.max_iters:
            break

        x, y = make_batch(train_data, block_size, args.batch_size, args.device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    total_time = time.time() - t0
    print("-" * 72)
    print(f"Training complete in {total_time / 60:.1f} minutes")
    print(f"Best val loss    : {best_val_loss:.4f}")
    print(f"Best checkpoint  : {best_ckpt}")
    print(f"Latest checkpoint: {latest_ckpt}")

    try:
        final_sample = sample_text(
            model=model,
            tokenizer=tokenizer,
            prompt="Je ne sais pas",
            device=args.device,
            max_new=200,
            temperature=0.9,
            top_k=40,
        )
        print("\nFinal sample:")
        print(final_sample)
    except Exception as e:
        print(f"\nFinal sample failed: {e}")


if __name__ == "__main__":
    main()
