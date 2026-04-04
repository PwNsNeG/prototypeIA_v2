#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

try:
    from safetensors.torch import save_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False


def save_json(path: Path, obj) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Export TinyGPT v2 checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or latest.pt")
    parser.add_argument("--out-dir", required=True, help="Export directory")
    parser.add_argument("--tokenizer", default=None, help="Optional tokenizer.json override")
    parser.add_argument("--copy-model-py", action="store_true", help="Copy local model.py into export dir")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_state = ckpt["model"]
    model_config = ckpt["model_config"]
    train_config = ckpt.get("train_config", {})
    step = ckpt.get("step")
    val_loss = ckpt.get("val_loss")

    tokenizer_path = args.tokenizer or train_config.get("tokenizer")
    if not tokenizer_path:
        raise ValueError("Tokenizer path not found in checkpoint. Pass --tokenizer explicitly.")
    tokenizer_path = Path(tokenizer_path)
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    data_dir = Path(train_config["data_dir"]) if "data_dir" in train_config else None
    dataset_meta = None
    if data_dir:
        meta_path = data_dir / "meta.json"
        if meta_path.exists():
            dataset_meta = load_json(meta_path)

    if HAS_SAFETENSORS:
        weights_file = "model.safetensors"
        save_file(model_state, str(out_dir / weights_file))
    else:
        weights_file = "model.pt"
        torch.save(model_state, out_dir / weights_file)

    save_json(out_dir / "config.json", model_config)
    save_json(out_dir / "train_config.json", train_config)
    save_json(
        out_dir / "meta.json",
        {
            "source_checkpoint": str(ckpt_path),
            "step": step,
            "val_loss": val_loss,
            "weights_file": weights_file,
            "model_class": "TinyGPT",
            "config_class": "GPTConfig",
            "tokenizer_file": "tokenizer.json",
            "dataset_meta": dataset_meta,
        },
    )

    shutil.copy2(tokenizer_path, out_dir / "tokenizer.json")

    if args.copy_model_py:
        src_model_py = Path("model.py")
        if not src_model_py.exists():
            raise FileNotFoundError("model.py not found in current directory")
        shutil.copy2(src_model_py, out_dir / "model.py")

    print(f"Export complete: {out_dir}")
    print(f"Weights       : {out_dir / weights_file}")
    print(f"Config        : {out_dir / 'config.json'}")
    print(f"Train config  : {out_dir / 'train_config.json'}")
    print(f"Tokenizer     : {out_dir / 'tokenizer.json'}")
    if args.copy_model_py:
        print(f"Model source  : {out_dir / 'model.py'}")


if __name__ == "__main__":
    main()
