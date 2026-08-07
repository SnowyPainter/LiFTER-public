from __future__ import annotations

import math

import torch
from torch import nn

from .attention_block import AttentionBlock


class AttentionedAttention(nn.Module):
    """Single-head attention whose aggregation can be switched to AttentionBlock."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        use_attention: bool = False,
        dropout: float = 0.0,
        gate_init: float = 0.95,
        gate_leak: float = 0.05,
    ) -> None:
        super().__init__()
        self.use_attention = bool(use_attention)
        self.query_proj = nn.Linear(input_dim, hidden_dim)
        self.key_proj = nn.Linear(input_dim, hidden_dim)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = AttentionBlock(
            query_dim=hidden_dim,
            key_dim=hidden_dim,
            value_dim=hidden_dim,
            gate_init=gate_init,
            gate_leak=gate_leak,
            dropout=dropout,
        )

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if query.ndim == 2:
            query_3d = query.unsqueeze(1)
            squeeze = True
        else:
            query_3d = query
            squeeze = False
        q = self.query_proj(query_3d)
        k = self.key_proj(memory)
        v = self.value_proj(memory)
        logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(float(k.shape[-1]))
        if mask is not None:
            pair_mask = mask.unsqueeze(1) if mask.ndim == 2 else mask
            logits = logits.masked_fill(~pair_mask, torch.finfo(logits.dtype).min)
            empty_rows = ~pair_mask.any(dim=-1, keepdim=True)
            logits = torch.where(empty_rows, torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=-1)
        if mask is not None:
            weights = weights * pair_mask.to(weights.dtype)
        if self.use_attention:
            out, weights, gate, attention_logits = self.attention(
                q,
                k,
                v,
                attention_weights=weights,
                mask=mask,
            )
        else:
            if mask is not None:
                weights = weights * (mask.unsqueeze(1) if mask.ndim == 2 else mask).to(weights.dtype)
            out = torch.matmul(self.dropout(weights), v)
            gate = torch.ones_like(weights)
            attention_logits = torch.zeros_like(weights)
        if squeeze:
            out = out.squeeze(1)
            weights = weights.squeeze(1)
            gate = gate.squeeze(1)
            attention_logits = attention_logits.squeeze(1)
        return out, {"attention": weights, "attention_gate": gate, "attention_logits": attention_logits}


class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        frequencies = torch.exp(torch.linspace(0.0, 6.0, steps=max(1, dim // 2)))
        self.register_buffer("frequencies", frequencies, persistent=False)
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        scaled = torch.log1p(t.float().clamp_min(0.0)).unsqueeze(-1) / self.frequencies
        encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
        return encoded[..., : self.dim] if encoded.shape[-1] >= self.dim else torch.nn.functional.pad(
            encoded,
            (0, self.dim - encoded.shape[-1]),
        )
