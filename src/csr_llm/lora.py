"""LoRA (Low-Rank Adaptation) for TinyGPT.

Adds trainable low-rank adapters to QKV projections while freezing
all base model weights. This decouples task fine-tuning from the
weights used for generation, eliminating EOS-creep.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from csr_llm.model import TinyGPT


class LoRALinear(nn.Module):
    """nn.Linear with a frozen base weight and a trainable low-rank adapter.

    output = x @ W.T + (x @ A.T @ B.T) * scaling
    where W is frozen, A and B are trained.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        in_f = linear.in_features
        out_f = linear.out_features
        self.scaling = alpha / rank

        # Frozen base
        self.weight = nn.Parameter(linear.weight.data.clone(), requires_grad=False)
        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        # Trainable low-rank matrices (B init to zero → adapter is zero at start)
        self.lora_A = nn.Parameter(torch.empty(rank, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        delta = (x @ self.lora_A.T) @ self.lora_B.T
        return base + delta * self.scaling


def apply_lora(model: "TinyGPT", rank: int = 8, alpha: float = 16.0) -> "TinyGPT":
    """Freeze all base weights and inject LoRA adapters into QKV projections.

    Call this on a freshly loaded base model to get an offspring model
    where only the LoRA parameters are trainable.
    """
    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Replace qkv linear in every attention layer with LoRALinear
    for layer in model.layers:
        layer.attn.qkv = LoRALinear(layer.attn.qkv, rank=rank, alpha=alpha)

    return model


def lora_param_count(model: "TinyGPT") -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_lora(model: "TinyGPT", path) -> None:
    """Save only the LoRA adapter weights (not the frozen base)."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lora_state = {
        name: param.data
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save(lora_state, path)


def load_lora(model: "TinyGPT", path) -> "TinyGPT":
    """Load LoRA adapter weights into a model that already has apply_lora applied."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model_params = dict(model.named_parameters())
    for name, data in state.items():
        if name in model_params:
            model_params[name].data.copy_(data)
    return model
