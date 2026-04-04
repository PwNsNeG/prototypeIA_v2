#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.1


class Head(nn.Module):
    def __init__(self, config: GPTConfig, head_size: int):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(config.block_size, config.block_size))
        )

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        scale = k.shape[-1] ** -0.5
        wei = q @ k.transpose(-2, -1) * scale
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"n_embd ({config.n_embd}) must be divisible by n_head ({config.n_head})"
            )

        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds block_size {self.config.block_size}"
            )

        device = idx.device
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature <= 0:
                raise ValueError("temperature must be > 0")

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = GPTConfig(
        vocab_size=16000,
        block_size=256,
        n_embd=384,
        n_head=6,
        n_layer=6,
        dropout=0.1,
    )

    model = TinyGPT(config).to(device)

    total_params = model.num_parameters()
    trainable = model.trainable_parameters()

    print(f"Device           : {device}")
    print(f"Config           : {asdict(config)}")
    print(f"Total params     : {total_params:,}")
    print(f"Trainable params : {trainable:,}")

    x = torch.randint(0, config.vocab_size, (4, config.block_size), device=device)
    y = torch.randint(0, config.vocab_size, (4, config.block_size), device=device)
    logits, loss = model(x, y)

    print("\nForward pass test:")
    print(f"  Input shape  : {tuple(x.shape)}")
    print(f"  Logits shape : {tuple(logits.shape)}")
    print(f"  Loss         : {loss.item():.4f} (random init ≈ {math.log(config.vocab_size):.2f})")

    print("\nModel ready.")
