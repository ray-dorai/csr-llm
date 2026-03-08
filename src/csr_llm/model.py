"""Tiny GPT-2 style model definition with mutation and recombination operators."""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyGPT(nn.Module):
    """Minimal GPT-2 style decoder-only transformer."""

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 8,
        d_ff: int = 1024,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        self._n_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def n_params(self) -> int:
        return self._n_params

    def forward(
        self, input_ids: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        tok_emb = self.token_emb(input_ids)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)

        x = self.drop(tok_emb + pos_emb)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        self.eval()
        for _ in range(max_new_tokens):
            # Crop to max_seq_len
            idx_cond = input_ids[:, -self.max_seq_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                # Greedy
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_id], dim=1)

            if eos_token_id is not None and (next_id == eos_token_id).all():
                break

        self.train()
        return input_ids


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention with causal mask
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = (att @ v).transpose(1, 2).contiguous().reshape(B, T, C)
        return self.proj_drop(self.proj(y))


# --- Evolutionary operators ---


def save_model(model: TinyGPT, path: str | Path) -> None:
    """Save model weights."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: str | Path, cfg: dict) -> TinyGPT:
    """Load model weights from checkpoint."""
    m = cfg["model"]
    model = TinyGPT(
        vocab_size=m["vocab_size"],
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        d_ff=m["d_ff"],
        max_seq_len=m["max_seq_len"],
        dropout=m["dropout"],
    )
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model


def create_model(cfg: dict) -> TinyGPT:
    """Create a fresh model from config."""
    m = cfg["model"]
    return TinyGPT(
        vocab_size=m["vocab_size"],
        d_model=m["d_model"],
        n_heads=m["n_heads"],
        n_layers=m["n_layers"],
        d_ff=m["d_ff"],
        max_seq_len=m["max_seq_len"],
        dropout=m["dropout"],
    )


def mutate_model(model: TinyGPT, sigma: float = 0.01, seed: Optional[int] = None) -> TinyGPT:
    """Create a mutant by adding Gaussian noise to all parameters."""
    mutant = copy.deepcopy(model)
    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        for param in mutant.parameters():
            noise = torch.randn_like(param) * sigma
            param.add_(noise)

    return mutant


def recombine_models(
    parent_a: TinyGPT, parent_b: TinyGPT, seed: Optional[int] = None
) -> TinyGPT:
    """Create offspring by randomly swapping layers between two parents."""
    child = copy.deepcopy(parent_a)
    if seed is not None:
        torch.manual_seed(seed)

    with torch.no_grad():
        for (name_a, param_a), (name_b, param_b) in zip(
            child.named_parameters(), parent_b.named_parameters()
        ):
            # For each parameter tensor, 50/50 chance of taking from parent B
            if torch.rand(1).item() < 0.5:
                param_a.copy_(param_b)

    return child
