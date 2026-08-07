from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class AttentionBlock(nn.Module):
    """Attentioned attention aggregation.

    This is the operator from the research plan:

        alpha_ij = softmax_j(q_i k_j)
        g_ij = gate_leak + (1 - gate_leak) * sigmoid(psi(q_i, k_j))
        h_i = sum_j alpha_ij * g_ij * v_j

    The gate equation is for valid query-key pairs; masked pairs are zeroed.

    Attention estimates relevance.  Attentioning estimates permission: whether a
    weighted value term is allowed to influence the current query's prediction.
    The same query-conditioned value gate is used across domains; adaptation
    happens through the model's learned query/key/value representations.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        value_dim: int,
        *,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        gate_init: float = 0.95,
        gate_leak: float = 0.05,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = value_dim if hidden_dim is None else int(hidden_dim)
        output_dim = value_dim if output_dim is None else int(output_dim)
        if min(query_dim, key_dim, value_dim, hidden_dim, output_dim) <= 0:
            raise ValueError("all dimensions must be positive")
        if not (0.0 <= gate_leak <= 1.0):
            raise ValueError("gate_leak must satisfy 0 <= gate_leak <= 1")
        if not (0.0 < gate_init < 1.0):
            raise ValueError("gate_init must satisfy 0 < gate_init < 1")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

        self.query_dim = int(query_dim)
        self.key_dim = int(key_dim)
        self.value_dim = int(value_dim)
        self.output_dim = int(output_dim)
        self.gate_leak = float(gate_leak)

        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.scorer = nn.Linear(hidden_dim, 1)
        self.output_proj = nn.Identity() if output_dim == value_dim else nn.Linear(value_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self._init_gate_bias(float(gate_init))

    def _init_gate_bias(self, gate_init: float) -> None:
        if self.gate_leak >= 1.0:
            prob = 0.5
        else:
            prob = (gate_init - self.gate_leak) / (1.0 - self.gate_leak)
            prob = min(max(prob, 1.0e-4), 1.0 - 1.0e-4)
        nn.init.zeros_(self.scorer.weight)
        nn.init.constant_(self.scorer.bias, math.log(prob / (1.0 - prob)))

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attention_logits: torch.Tensor | None = None,
        attention_weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(output, attention_weights, gate, attention_logits)``.

        Shapes:
            query: ``[batch, query_len, query_dim]`` or ``[batch, query_dim]``
            key: ``[batch, key_len, key_dim]``
            value: ``[batch, key_len, value_dim]``
            mask: optional bool ``[batch, key_len]`` or ``[batch, query_len, key_len]``

        If an existing attention block already computed attention weights, pass
        them through ``attention_weights``.  Otherwise this module computes the
        usual scaled dot-product relevance distribution.
        """

        squeeze_query = False
        if query.ndim == 2:
            query = query.unsqueeze(1)
            squeeze_query = True
        if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
            raise ValueError("query, key, and value must be 3D tensors, except query may be 2D")
        if key.shape[:2] != value.shape[:2]:
            raise ValueError("key and value must have the same [batch, length] shape")
        if query.shape[0] != key.shape[0]:
            raise ValueError("query, key, and value must have the same batch size")

        batch_size, query_len, _ = query.shape
        key_len = key.shape[1]
        pair_mask = self._pair_mask(mask, batch_size, query_len, key_len, query.device)

        gate_logits = self._gate_logits(query, key)
        gate = self.gate_leak + (1.0 - self.gate_leak) * torch.sigmoid(gate_logits)
        gate = torch.where(pair_mask, gate, torch.zeros_like(gate))
        gate_logits = torch.where(pair_mask, gate_logits, torch.zeros_like(gate_logits))

        if attention_weights is None:
            if attention_logits is None:
                attention_logits = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(float(key.shape[-1]))
            attention_logits = attention_logits.masked_fill(~pair_mask, torch.finfo(attention_logits.dtype).min)
            empty_rows = ~pair_mask.any(dim=-1, keepdim=True)
            attention_logits = torch.where(empty_rows, torch.zeros_like(attention_logits), attention_logits)
            attention_weights = torch.softmax(attention_logits, dim=-1)
        else:
            if attention_weights.shape != (batch_size, query_len, key_len):
                raise ValueError(
                    "attention_weights must have shape "
                    f"{(batch_size, query_len, key_len)}, got {tuple(attention_weights.shape)}"
                )
            attention_weights = attention_weights.to(device=query.device, dtype=value.dtype)
            attention_weights = attention_weights * pair_mask.to(attention_weights.dtype)

        attention_weights = self.dropout(attention_weights)
        attentioned_values = value.unsqueeze(1) * gate.unsqueeze(-1)
        output = torch.sum(attention_weights.unsqueeze(-1) * attentioned_values, dim=2)
        output = self.output_proj(output)

        if squeeze_query:
            output = output.squeeze(1)
            attention_weights = attention_weights.squeeze(1)
            gate = gate.squeeze(1)
            gate_logits = gate_logits.squeeze(1)
        return output, attention_weights, gate, gate_logits

    def _gate_logits(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.query_proj(query).unsqueeze(2) + self.key_proj(key).unsqueeze(1)
        return self.scorer(F.gelu(hidden)).squeeze(-1)

    @staticmethod
    def _pair_mask(
        mask: torch.Tensor | None,
        batch_size: int,
        query_len: int,
        key_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        if mask is None:
            return torch.ones(batch_size, query_len, key_len, dtype=torch.bool, device=device)
        mask = mask.to(device=device, dtype=torch.bool)
        if mask.shape == (batch_size, key_len):
            return mask.unsqueeze(1).expand(-1, query_len, -1)
        if mask.shape == (batch_size, query_len, key_len):
            return mask
        raise ValueError(
            f"mask must have shape {(batch_size, key_len)} or "
            f"{(batch_size, query_len, key_len)}, got {tuple(mask.shape)}"
        )

    def aux_loss(
        self,
        attention_logits: torch.Tensor,
        attention_targets: torch.Tensor | None = None,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Optional supervised attention loss.  Best value is 0."""

        if attention_targets is None:
            return attention_logits.new_zeros(())
        targets = attention_targets.to(device=attention_logits.device, dtype=attention_logits.dtype)
        if targets.shape != attention_logits.shape:
            raise ValueError(
                f"attention_targets must have shape {tuple(attention_logits.shape)}, got {tuple(targets.shape)}"
            )
        if mask is None:
            return F.binary_cross_entropy_with_logits(attention_logits, targets)
        mask = mask.to(device=attention_logits.device, dtype=torch.bool)
        if mask.shape != attention_logits.shape:
            raise ValueError(f"mask must have shape {tuple(attention_logits.shape)}, got {tuple(mask.shape)}")
        if not bool(mask.any()):
            return attention_logits.new_zeros(())
        return F.binary_cross_entropy_with_logits(attention_logits[mask], targets[mask])


class QueryPatchValidityPredictor(nn.Module):
    """Additive query-history validity scorer used by edge-conditioned attentioning."""

    def __init__(
        self,
        hidden_dim: int,
        time_dim: int,
        *,
        stat_dim: int = 3,
        dropout: float = 0.0,
        gate_init: float = 0.95,
        gate_leak: float = 0.05,
    ) -> None:
        super().__init__()
        self.gate_leak = float(gate_leak)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.patch_proj = nn.Linear(hidden_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.stat_proj = nn.Linear(stat_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self._init_gate_head(float(gate_init))

    def _init_gate_head(self, gate_init: float) -> None:
        if self.gate_leak >= 1.0:
            prob = 0.5
        else:
            prob = (gate_init - self.gate_leak) / (1.0 - self.gate_leak)
            prob = min(max(prob, 1.0e-4), 1.0 - 1.0e-4)
        last = self.scorer[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.constant_(last.bias, math.log(prob / (1.0 - prob)))

    def forward(
        self,
        query: torch.Tensor,
        patch_keys: torch.Tensor,
        delta_t_enc: torch.Tensor,
        stats: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_h = self.query_proj(query).unsqueeze(1)
        hidden = self.patch_proj(patch_keys) + query_h + self.time_proj(delta_t_enc) + self.stat_proj(stats)
        logits = self.scorer(hidden).squeeze(-1)
        gate = self.gate_leak + (1.0 - self.gate_leak) * torch.sigmoid(logits)
        gate = gate * valid_mask.to(gate.dtype)
        logits = torch.where(valid_mask, logits, torch.zeros_like(logits))
        return gate, logits


class EdgeConditionedAttentionBlock(nn.Module):
    """Candidate-edge-conditioned value routing over source and destination history.

    This mirrors the FLIP-V VITAL usage: the queried candidate edge is visible
    to the validity stream, both endpoint histories are retrieved, and the
    gated contexts are injected into the endpoint representations before final
    scoring.
    """

    def __init__(
        self,
        hidden_dim: int,
        time_dim: int,
        *,
        stat_dim: int = 3,
        dropout: float = 0.0,
        gate_init: float = 0.95,
        gate_leak: float = 0.05,
        context_scale_init: float = 0.10,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.time_dim = int(time_dim)
        self.stat_dim = int(stat_dim)
        self.query_encoder = nn.Sequential(
            nn.Linear(2 * hidden_dim + time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.token_encoder = nn.Sequential(
            nn.Linear(hidden_dim + stat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_time_bias = nn.Linear(time_dim, 1)
        self.validity = QueryPatchValidityPredictor(
            hidden_dim,
            time_dim,
            stat_dim=stat_dim,
            dropout=dropout,
            gate_init=gate_init,
            gate_leak=gate_leak,
        )
        self.context_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        self.context_scale = nn.Parameter(torch.tensor(float(context_scale_init), dtype=torch.float32))
        self.last_validity_mean: torch.Tensor | None = None
        self.last_attention_entropy: torch.Tensor | None = None

    def _semantic_attention(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        delta_t_enc: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(query).unsqueeze(1)
        k = self.k_proj(tokens)
        logits = (q * k).sum(dim=-1) / math.sqrt(float(max(1, self.hidden_dim)))
        logits = logits + self.attn_time_bias(delta_t_enc).squeeze(-1)
        logits = logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)
        empty = ~valid_mask.any(dim=-1, keepdim=True)
        logits = torch.where(empty, torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=-1) * valid_mask.to(logits.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)

    def _side_context(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        delta_t_enc: torch.Tensor,
        stats: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.token_encoder(torch.cat([memory, stats], dim=-1))
        tokens = tokens * valid_mask.unsqueeze(-1).to(tokens.dtype)
        weights = self._semantic_attention(query, tokens, delta_t_enc, valid_mask)
        gate, logits = self.validity(query, tokens, delta_t_enc, stats, valid_mask)
        context = (tokens * weights.unsqueeze(-1) * gate.unsqueeze(-1)).sum(dim=1)
        return context, gate, weights

    def forward(
        self,
        *,
        src_h: torch.Tensor,
        dst_h: torch.Tensor,
        time_h: torch.Tensor,
        src_memory: torch.Tensor,
        dst_memory: torch.Tensor,
        src_delta_t_enc: torch.Tensor,
        dst_delta_t_enc: torch.Tensor,
        src_stats: torch.Tensor,
        dst_stats: torch.Tensor,
        src_mask: torch.Tensor,
        dst_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        query = self.query_encoder(torch.cat([src_h, dst_h, time_h], dim=-1))
        src_context, src_gate, src_weights = self._side_context(query, src_memory, src_delta_t_enc, src_stats, src_mask)
        dst_context, dst_gate, dst_weights = self._side_context(query, dst_memory, dst_delta_t_enc, dst_stats, dst_mask)
        delta = self.context_proj(torch.cat([src_context, dst_context], dim=-1))
        src_delta, dst_delta = delta.chunk(2, dim=-1)

        valid_count = src_mask.float().sum() + dst_mask.float().sum()
        validity_mean = (src_gate.sum() + dst_gate.sum()) / valid_count.clamp_min(1.0)
        entropy = 0.5 * (
            -(src_weights.clamp_min(1.0e-8) * src_weights.clamp_min(1.0e-8).log()).sum(dim=-1).mean()
            - (dst_weights.clamp_min(1.0e-8) * dst_weights.clamp_min(1.0e-8).log()).sum(dim=-1).mean()
        )
        self.last_validity_mean = validity_mean.detach()
        self.last_attention_entropy = entropy.detach()
        info = {
            "edge_attention_gate_mean": validity_mean.detach(),
            "edge_attention_attention_entropy": entropy.detach(),
        }
        return self.context_scale * src_delta, self.context_scale * dst_delta, info


__all__ = ["EdgeConditionedAttentionBlock", "QueryPatchValidityPredictor", "AttentionBlock"]
