#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model import TinyGPT, GPTConfig


def choose_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_runtime(
    checkpoint_path: Path,
    tokenizer_path: Path | None,
    device: str,
):
    ckpt = torch.load(checkpoint_path, map_location=device)

    model_config_dict = ckpt["model_config"]
    train_config = ckpt.get("train_config", {})

    if tokenizer_path is None:
        tokenizer_from_ckpt = train_config.get("tokenizer")
        if not tokenizer_from_ckpt:
            raise ValueError(
                "Tokenizer path not provided and not found in checkpoint train_config. "
                "Pass --tokenizer explicitly."
            )
        tokenizer_path = Path(tokenizer_from_ckpt)

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    model_config = GPTConfig(**model_config_dict)
    model = TinyGPT(model_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    return ckpt, model, tokenizer, model_config_dict, train_config, tokenizer_path


@torch.no_grad()
def generate_text(
    model,
    tokenizer: Tokenizer,
    prompt: str,
    device: str,
    max_new: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
) -> str:
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
    return tokenizer.decode(out[0].tolist(), skip_special_tokens=True)


PRESETS = {
    "1": ("Polite opener", "Monsieur,", 120, 0.8, 40),
    "2": ("Narrative", "Il était une fois", 160, 0.8, 40),
    "3": ("Dialogue", "— Je ne sais pas,", 180, 0.8, 40),
    "4": ("Formal French", "Je vous prie de", 150, 0.7, 30),
    "5": ("Conservative", "Le ", 120, 0.5, 20),
    "6": ("Creative", "Ah ! ", 200, 1.0, 80),
}


def print_help(temperature: float, max_new: int, top_k: int | None) -> None:
    print("\nCommands:")
    print("  <enter text>     use as prompt")
    print("  preset           show preset list")
    print("  1-6              run a preset")
    print(f"  temp <value>     set temperature (current: {temperature:.2f})")
    print(f"  len <number>     set output length (current: {max_new})")
    print(f"  topk <number>    set top-k (current: {top_k})")
    print("  info             show runtime info")
    print("  help             show this help")
    print("  quit             exit\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive TinyGPT v2 generator")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--prompt", default=None, help="Run once and exit")
    parser.add_argument("--max-new", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    tokenizer_path = Path(args.tokenizer) if args.tokenizer else None
    device = choose_device(args.device)

    ckpt, model, tokenizer, model_cfg, train_cfg, resolved_tokenizer = load_runtime(
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        device=device,
    )

    val_loss = float(ckpt.get("val_loss", float("nan")))
    ppl = math.exp(val_loss) if math.isfinite(val_loss) else float("nan")
    total_params = sum(p.numel() for p in model.parameters())

    def show_info():
        print("=" * 72)
        print("TinyGPT v2 — interactive generator")
        print(f"Checkpoint      : {checkpoint_path}")
        print(f"Tokenizer       : {resolved_tokenizer}")
        print(f"Device          : {device}")
        print(f"Step            : {ckpt.get('step', 'n/a')}")
        if math.isfinite(val_loss):
            print(f"Val loss        : {val_loss:.4f}")
            print(f"Perplexity      : {ppl:.2f}")
        else:
            print("Val loss        : n/a")
        print(f"Parameters      : {total_params:,}")
        print(f"Vocab size      : {model_cfg.get('vocab_size')}")
        print(f"Block size      : {model_cfg.get('block_size')}")
        print(f"n_embd          : {model_cfg.get('n_embd')}")
        print(f"n_head          : {model_cfg.get('n_head')}")
        print(f"n_layer         : {model_cfg.get('n_layer')}")
        print("=" * 72)

    if args.prompt is not None:
        text = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            device=device,
            max_new=args.max_new,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(text)
        return

    temperature = args.temperature
    max_new = args.max_new
    top_k = args.top_k

    show_info()
    print("\nType a French prompt to continue, or 'preset' to see examples.")
    print("Type 'help' for all commands.\n")

    while True:
        try:
            user = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir !")
            break

        if not user:
            continue

        if user == "quit":
            print("Au revoir !")
            break

        if user == "help":
            print_help(temperature, max_new, top_k)
            continue

        if user == "info":
            show_info()
            continue

        if user == "preset":
            print("\nPresets:")
            for k, (name, prompt, ln, temp, tk) in PRESETS.items():
                print(
                    f"  {k}. {name:16s} prompt={prompt!r} temp={temp} len={ln} top_k={tk}"
                )
            print()
            continue

        if user in PRESETS:
            name, prompt, ln, temp, tk = PRESETS[user]
            print(f"\n[{name}]\n")
            try:
                print(
                    generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        device=device,
                        max_new=ln,
                        temperature=temp,
                        top_k=tk,
                    )
                )
            except Exception as e:
                print(f"Generation failed: {e}")
            print()
            continue

        if user.startswith("temp "):
            try:
                temperature = float(user.split(maxsplit=1)[1])
                print(f"Temperature set to {temperature}")
            except ValueError:
                print("Usage: temp 0.8")
            continue

        if user.startswith("len "):
            try:
                max_new = int(user.split(maxsplit=1)[1])
                print(f"Length set to {max_new}")
            except ValueError:
                print("Usage: len 200")
            continue

        if user.startswith("topk "):
            try:
                top_k = int(user.split(maxsplit=1)[1])
                print(f"Top-k set to {top_k}")
            except ValueError:
                print("Usage: topk 40")
            continue

        try:
            print()
            print(
                generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=user,
                    device=device,
                    max_new=max_new,
                    temperature=temperature,
                    top_k=top_k,
                )
            )
            print()
        except Exception as e:
            print(f"Generation failed: {e}")


if __name__ == "__main__":
    main()
