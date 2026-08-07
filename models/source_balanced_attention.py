from __future__ import annotations

import torch
from torch import nn


class SourceBalancedSoftmax(nn.Module):
    """Softmax whose total prior mass is allocated per evidence source.

    For key ``i`` from source group ``g(i)``, the adjusted logit is

        s_i - strength * log |g(i)|.

    At ``strength=1``, equal-scoring records from one source contribute their
    mean exponential score instead of a sum that grows with serialization
    multiplicity. ``source_ids`` identify common provenance; negative IDs are
    treated as invalid/padding.
    """

    def __init__(self, strength: float = 1.0, *, learnable: bool = False) -> None:
        super().__init__()
        value = torch.tensor(float(strength))
        if learnable:
            self.strength = nn.Parameter(value)
        else:
            self.register_buffer("strength", value, persistent=False)

    def forward(
        self,
        logits: torch.Tensor,
        source_ids: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.ndim != 3:
            raise ValueError(f"logits must be [batch, queries, keys], got {tuple(logits.shape)}")
        if source_ids.shape != (logits.shape[0], logits.shape[2]):
            raise ValueError(
                "source_ids must be [batch, keys], got "
                f"{tuple(source_ids.shape)} for logits {tuple(logits.shape)}"
            )
        valid = source_ids.ge(0)
        if mask is None:
            pair_mask = valid.unsqueeze(1).expand(-1, logits.shape[1], -1)
        else:
            pair_mask = mask.unsqueeze(1) if mask.ndim == 2 else mask
            pair_mask = pair_mask.to(torch.bool) & valid.unsqueeze(1)
        same_source = source_ids.unsqueeze(2).eq(source_ids.unsqueeze(1))
        same_source = same_source & valid.unsqueeze(2) & valid.unsqueeze(1)
        group_count = torch.einsum(
            "bik,bqk->bqi",
            same_source.to(logits.dtype),
            pair_mask.to(logits.dtype),
        ).clamp_min(1.0)
        strength = self.strength.to(device=logits.device, dtype=logits.dtype).clamp(0.0, 1.0)
        adjusted = logits - strength * torch.log(group_count)
        adjusted = adjusted.masked_fill(~pair_mask, torch.finfo(logits.dtype).min)
        empty = ~pair_mask.any(dim=-1, keepdim=True)
        adjusted = torch.where(empty, torch.zeros_like(adjusted), adjusted)
        weights = torch.softmax(adjusted, dim=-1) * pair_mask.to(logits.dtype)
        return weights, group_count


class SourceBalancedAttentionAdapter(nn.Module):
    """Evaluation-time adapter for the repository's base attention module.

    The wrapped module is called unchanged while source balancing is disabled.
    The separate path is intended for base models only and never modifies the
    wrapped module or its parameters.
    """

    def __init__(self, base_attention: nn.Module) -> None:
        super().__init__()
        if bool(getattr(base_attention, "use_attention", False)):
            raise ValueError("SourceBalancedAttentionAdapter only accepts base attention modules")
        self.base_attention = base_attention
        self.softmax = SourceBalancedSoftmax()
        self.enabled = False
        self.source_ids: torch.Tensor | None = None
        self.last_attention: torch.Tensor | None = None
        self.last_group_count: torch.Tensor | None = None

    def set_source_context(self, source_ids: torch.Tensor | None, *, enabled: bool) -> None:
        self.source_ids = source_ids
        self.enabled = bool(enabled)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if not self.enabled:
            output, info = self.base_attention(query, memory, mask=mask)
            self.last_attention = info.get("attention", None)
            if self.last_attention is not None:
                self.last_attention = self.last_attention.detach()
            self.last_group_count = None
            return output, info
        if self.source_ids is None:
            raise RuntimeError("source context must be set before balanced attention")
        if query.ndim == 2:
            query_3d = query.unsqueeze(1)
            squeeze = True
        else:
            query_3d = query
            squeeze = False
        q = self.base_attention.query_proj(query_3d)
        k = self.base_attention.key_proj(memory)
        v = self.base_attention.value_proj(memory)
        logits = torch.matmul(q, k.transpose(1, 2)) / float(k.shape[-1]) ** 0.5
        pair_mask = None
        if mask is not None:
            pair_mask = mask.unsqueeze(1) if mask.ndim == 2 else mask
            logits = logits.masked_fill(~pair_mask, torch.finfo(logits.dtype).min)
            empty = ~pair_mask.any(dim=-1, keepdim=True)
            logits = torch.where(empty, torch.zeros_like(logits), logits)
        weights, group_count = self.softmax(logits, self.source_ids, mask=mask)
        self.last_attention = weights.detach()
        self.last_group_count = group_count.detach()
        out = torch.matmul(self.base_attention.dropout(weights), v)
        gate = torch.ones_like(weights)
        zero_logits = torch.zeros_like(weights)
        if squeeze:
            out = out.squeeze(1)
            weights = weights.squeeze(1)
            gate = gate.squeeze(1)
            zero_logits = zero_logits.squeeze(1)
            group_count = group_count.squeeze(1)
        return out, {
            "attention": weights,
            "attention_gate": gate,
            "attention_logits": zero_logits,
            "source_group_count": group_count.detach(),
        }


def replace_base_attention(module: nn.Module) -> int:
    """Recursively wrap base AttentionedAttention instances without editing them."""

    from .attention import AttentionedAttention

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, SourceBalancedAttentionAdapter):
            continue
        if isinstance(child, AttentionedAttention):
            setattr(module, name, SourceBalancedAttentionAdapter(child))
            replaced += 1
        else:
            replaced += replace_base_attention(child)
    return replaced


def set_source_context(module: nn.Module, source_ids: torch.Tensor | None, *, enabled: bool) -> int:
    configured = 0
    for child in module.modules():
        if isinstance(child, SourceBalancedAttentionAdapter):
            child.set_source_context(source_ids, enabled=enabled)
            configured += 1
    return configured


__all__ = [
    "SourceBalancedAttentionAdapter",
    "SourceBalancedSoftmax",
    "replace_base_attention",
    "set_source_context",
]
