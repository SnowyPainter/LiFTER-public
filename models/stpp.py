from __future__ import annotations

import torch
from torch import nn

from .attention import SinusoidalTimeEncoding, AttentionedAttention
from .base import ModelResult, AttentionModel
from .mtpp import _causal_mask, _last_valid
from .attention_block import AttentionBlock


class _SpatioTemporalSelfAttentionBlock(nn.Module):
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

    def forward(
        self,
        x: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y, info = self.attn(self.norm1(x), self.norm1(x), mask=pair_mask)
        x = x + y
        x = x + self.ffn(self.norm2(x))
        return x, info


class _STPPBase(AttentionModel):
    model_name = "stpp_base"

    def __init__(
        self,
        *,
        num_marks: int = 1024,
        hidden_dim: int = 128,
        time_dim: int = 32,
        num_layers: int = 2,
        use_attention: bool = False,
        dropout: float = 0.1,
        gate_init: float = 0.95,
        gate_leak: float = 0.05,
    ) -> None:
        super().__init__(use_attention=use_attention)
        self.num_marks = int(num_marks)
        self.hidden_dim = int(hidden_dim)
        self.mark_embedding = nn.Embedding(num_marks + 1, hidden_dim, padding_idx=num_marks)
        self.time_encoder = SinusoidalTimeEncoding(time_dim)
        self.space_encoder = nn.Sequential(nn.Linear(2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.input_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim + time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.layers = nn.ModuleList(
            [
                _SpatioTemporalSelfAttentionBlock(
                    hidden_dim,
                    use_attention=False,
                    dropout=dropout,
                    gate_init=gate_init,
                    gate_leak=gate_leak,
                )
                for _ in range(num_layers)
            ]
        )
        self.prediction_query = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.prediction_attention: AttentionBlock | None = None
        self.prediction_attention_norm = nn.LayerNorm(hidden_dim)
        self.prediction_attention_scale = nn.Parameter(torch.tensor(-4.0)) if self.use_attention else None
        self.location_head = nn.Linear(hidden_dim, 2)
        nn.init.zeros_(self.location_head.weight)
        nn.init.zeros_(self.location_head.bias)
        self.location_log_scale = nn.Parameter(torch.zeros(2))
        self.mark_head = nn.Linear(hidden_dim, num_marks)
        self.time_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.time_head[-1].weight)
        nn.init.zeros_(self.time_head[-1].bias)
        self.time_log_scale = nn.Parameter(torch.zeros(()))
        if self.use_attention:
            self.prediction_attention = AttentionBlock(
                query_dim=hidden_dim,
                key_dim=hidden_dim,
                value_dim=hidden_dim,
                gate_init=gate_init,
                gate_leak=gate_leak,
                dropout=0.0,
            )
            self.prediction_attention._attention_audit_active = True

    def event_embedding(self, times: torch.Tensor, coords: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
        return self.input_proj(
            torch.cat(
                [
                    self.time_encoder(times.float()),
                    self.space_encoder(coords.float()),
                    self.mark_embedding(marks.long().clamp(min=0, max=self.num_marks)),
                ],
                dim=-1,
            )
        )

    def encode_events(
        self,
        times: torch.Tensor,
        coords: torch.Tensor,
        marks: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.event_embedding(times, coords, marks)
        pair_mask = _causal_mask(mask, x.shape[1], x.device)
        last_info: dict[str, torch.Tensor] = {}
        for layer in self.layers:
            x, last_info = layer(x, pair_mask)
        return x, last_info

    def forward(
        self,
        times: torch.Tensor,
        coords: torch.Tensor,
        marks: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> ModelResult:
        encoded, info = self.encode_events(times, coords, marks, mask)
        state = _last_valid(encoded, mask)
        dynamics_state = state
        if self.prediction_attention is not None:
            query = self.prediction_query(state)
            pred_context, pred_attention, pred_gate, pred_logits = self.prediction_attention(
                query,
                encoded,
                encoded,
                mask=mask,
            )
            scale = 0.2 * torch.sigmoid(self.prediction_attention_scale)
            dynamics_state = state + scale * self.prediction_attention_norm(pred_context)
            info = {
                **info,
                "attention_gate": pred_gate,
                "attention_logits": pred_logits,
                "prediction_attention_attention": pred_attention.detach(),
            }
        aux = self.aux_from_info(info)
        if "attention_logits" in info:
            aux["raw_attention_logits"] = info["attention_logits"]
        if getattr(self, "emit_prediction_attention_tensors", False):
            if "prediction_attention_attention" in info:
                aux["prediction_attention_attention"] = info["prediction_attention_attention"].detach()
            if "attention_gate" in info:
                aux["prediction_attention_gate"] = info["attention_gate"].detach()
        last_coord = _last_valid(coords, mask)
        aux["next_location"] = last_coord + self.location_head(dynamics_state)
        time_log_mean = self.time_head(dynamics_state).squeeze(-1)
        aux["time_log_mean"] = time_log_mean
        aux["time_log_scale"] = self.time_log_scale
        aux["location_log_scale"] = self.location_log_scale
        aux["next_time"] = torch.expm1(time_log_mean).clamp_min(0.0)
        return ModelResult(logits=self.mark_head(state), aux=aux)


class TransformerSTPP(_STPPBase):
    model_name = "Transformer-STPP"


class NSTPP(_STPPBase):
    model_name = "NSTPP"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.flow_state = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.flow_gate = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Sigmoid())

    def encode_events(
        self,
        times: torch.Tensor,
        coords: torch.Tensor,
        marks: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded, info = super().encode_events(times, coords, marks, mask)
        flow, _ = self.flow_state(encoded)
        return encoded + self.flow_gate(encoded) * flow, info


class DeepSTPP(_STPPBase):
    model_name = "DeepSTPP"

    def __init__(self, *args, latent_dim: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        latent_dim = self.hidden_dim if latent_dim is None else int(latent_dim)
        self.to_latent = nn.Linear(self.hidden_dim, latent_dim)
        self.from_latent = nn.Sequential(
            nn.Linear(latent_dim, 4 * self.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * self.hidden_dim, self.hidden_dim),
        )

    def encode_events(
        self,
        times: torch.Tensor,
        coords: torch.Tensor,
        marks: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        encoded, info = super().encode_events(times, coords, marks, mask)
        return encoded + self.from_latent(self.to_latent(encoded)), info
