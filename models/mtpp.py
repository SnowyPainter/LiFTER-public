from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .attention import SinusoidalTimeEncoding, AttentionedAttention
from .base import ModelResult, AttentionModel
from .attention_block import AttentionBlock


def _last_valid(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x[:, -1]
    positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand_as(mask)
    indices = torch.where(mask, positions, torch.full_like(positions, -1)).max(dim=1).values.clamp_min(0)
    return x[torch.arange(x.shape[0], device=x.device), indices]


def _causal_mask(valid: torch.Tensor | None, seq_len: int, device: torch.device) -> torch.Tensor:
    causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    if valid is None:
        return causal.unsqueeze(0)
    return causal.unsqueeze(0) & valid.unsqueeze(1)


class _TemporalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        use_attention: bool,
        dropout: float,
        gate_init: float,
        gate_leak: float,
    ) -> None:
        super().__init__()
        self.attn = AttentionedAttention(
            hidden_dim,
            hidden_dim,
            use_attention=use_attention,
            dropout=dropout,
            gate_init=gate_init,
            gate_leak=gate_leak,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor, pair_mask: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y, info = self.attn(self.norm1(x), self.norm1(x), mask=pair_mask)
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x, info


class _MTPPBase(AttentionModel):
    model_name = "mtpp_base"

    def __init__(
        self,
        *,
        num_event_types: int,
        hidden_dim: int = 128,
        time_dim: int = 32,
        num_layers: int = 2,
        use_attention: bool = False,
        dropout: float = 0.1,
        gate_init: float = 0.95,
        gate_leak: float = 0.05,
    ) -> None:
        super().__init__(use_attention=use_attention)
        self.num_event_types = int(num_event_types)
        self.hidden_dim = int(hidden_dim)
        self.mark_embedding = nn.Embedding(num_event_types + 1, hidden_dim, padding_idx=num_event_types)
        self.time_encoder = SinusoidalTimeEncoding(time_dim)
        self.input_proj = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                _TemporalSelfAttentionBlock(
                    hidden_dim,
                    use_attention=False,
                    dropout=dropout,
                    gate_init=gate_init,
                    gate_leak=gate_leak,
                )
                for _ in range(num_layers)
            ]
        )
        self.mark_head = nn.Linear(hidden_dim, num_event_types)
        self.candidate_query_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.candidate_score = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.candidate_score[-1].weight)
        nn.init.zeros_(self.candidate_score[-1].bias)
        self.candidate_attention: AttentionBlock | None = None
        self.candidate_value_norm: nn.LayerNorm | None = None
        self.candidate_value_proj: nn.Linear | None = None
        self.candidate_value_scale: nn.Parameter | None = None
        if self.use_attention:
            self.candidate_attention = AttentionBlock(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                gate_init=gate_init,
                gate_leak=gate_leak,
                dropout=dropout,
            )
            self.candidate_attention._attention_audit_active = True
            self.candidate_value_norm = nn.LayerNorm(hidden_dim)
            self.candidate_value_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.candidate_value_scale = nn.Parameter(torch.tensor(-3.0))
        self.time_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Softplus())

    def event_embedding(self, event_types: torch.Tensor, times: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            event_types = event_types.masked_fill(~mask, self.num_event_types)
            times = times.masked_fill(~mask, 0.0)
        mark_h = self.mark_embedding(event_types.long().clamp(min=0, max=self.num_event_types))
        return self.input_proj(torch.cat([mark_h, self.time_encoder(times.float())], dim=-1))

    def encode_events(
        self,
        event_types: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.event_embedding(event_types, times, mask)
        pair_mask = _causal_mask(mask, x.shape[1], x.device)
        last_info: dict[str, torch.Tensor] = {}
        for layer in self.layers:
            x, last_info = layer(x, pair_mask)
        return x, last_info

    def intensity_state(self, state: torch.Tensor, next_delta: torch.Tensor | None = None) -> torch.Tensor:
        return state

    def base_mark_logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.mark_head(state)

    def _apply_candidate_attention(
        self,
        base_logits: torch.Tensor,
        state: torch.Tensor,
        encoded: torch.Tensor,
        event_types: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None,
        info: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self.candidate_attention is None:
            return base_logits, info

        batch_size = event_types.shape[0]
        candidate_ids = torch.arange(self.num_event_types, device=event_types.device)
        candidate_h = self.mark_embedding(candidate_ids).unsqueeze(0).expand(batch_size, -1, -1)
        state_h = state.unsqueeze(1).expand(-1, self.num_event_types, -1)
        query = self.candidate_query_proj(torch.cat([state_h, candidate_h], dim=-1))
        pair_mask = (
            torch.ones(batch_size, self.num_event_types, event_types.shape[1], dtype=torch.bool, device=event_types.device)
            if mask is None
            else mask.to(torch.bool).unsqueeze(1).expand(-1, self.num_event_types, -1)
        )

        attentioned_context, attention, gate, attention_logits = self.candidate_attention(
            query,
            encoded,
            encoded,
            mask=pair_mask,
        )
        if self.candidate_value_norm is None or self.candidate_value_proj is None or self.candidate_value_scale is None:
            delta_logits = self.candidate_score(torch.cat([state_h, candidate_h, attentioned_context], dim=-1)).squeeze(-1)
        else:
            attentioned_value = self.candidate_value_proj(self.candidate_value_norm(attentioned_context))
            scale = 0.5 * torch.sigmoid(self.candidate_value_scale)
            delta_logits = scale * (candidate_h * attentioned_value).sum(dim=-1) / (float(self.hidden_dim) ** 0.5)
        merged_info = {
            **info,
            "attention_gate": gate,
            "attention_logits": attention_logits,
            "candidate_attention_attention": attention.detach(),
        }
        return base_logits + delta_logits, merged_info

    def forward(self, event_types: torch.Tensor, times: torch.Tensor, mask: torch.Tensor | None = None) -> ModelResult:
        encoded, info = self.encode_events(event_types, times, mask)
        state = _last_valid(encoded, mask)
        state = self.intensity_state(state)
        base_logits = self.base_mark_logits(state)
        logits, info = self._apply_candidate_attention(base_logits, state, encoded, event_types, times, mask, info)
        return ModelResult(logits=logits, aux={**self.aux_from_info(info), "next_time": self.time_head(state).squeeze(-1)})


class SAHP(_MTPPBase):
    model_name = "SAHP"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mu = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.eta = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.gamma = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim, bias=False), nn.Softplus())

    def intensity_state(self, state: torch.Tensor, next_delta: torch.Tensor | None = None) -> torch.Tensor:
        duration = 0.0 if next_delta is None else next_delta.unsqueeze(-1)
        mu = F.gelu(self.mu(state))
        eta = F.gelu(self.eta(state))
        gamma = self.gamma(state)
        return mu + (eta - mu) * torch.exp(-gamma * duration)


class THP(_MTPPBase):
    model_name = "THP"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.factor_intensity_base = nn.Parameter(torch.zeros(1, self.num_event_types))
        self.factor_intensity_decay = nn.Parameter(torch.zeros(1, self.num_event_types))
        self.thp_head = nn.Linear(self.hidden_dim, self.num_event_types)

    def forward(self, event_types: torch.Tensor, times: torch.Tensor, mask: torch.Tensor | None = None) -> ModelResult:
        encoded, info = self.encode_events(event_types, times, mask)
        state = _last_valid(encoded, mask)
        base_logits = self.thp_head(state) + self.factor_intensity_base
        logits, info = self._apply_candidate_attention(base_logits, state, encoded, event_types, times, mask, info)
        return ModelResult(logits=logits, aux={**self.aux_from_info(info), "next_time": self.time_head(state).squeeze(-1)})


class AttNHP(_MTPPBase):
    model_name = "AttNHP"

    def __init__(self, *args, num_heads: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_mix = nn.Parameter(torch.ones(num_heads) / max(1, num_heads))
        self.recurrent_state = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)

    def encode_events(
        self,
        event_types: torch.Tensor,
        times: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded, info = super().encode_events(event_types, times, mask)
        recurrent, _ = self.recurrent_state(encoded)
        return encoded + recurrent, info
