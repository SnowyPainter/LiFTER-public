from __future__ import annotations

import torch
from torch import nn

from .attention import AttentionedAttention
from .base import ModelResult, AttentionModel


class _RAGBase(AttentionModel):
    model_name = "rag_base"

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_dim: int = 256,
        use_attention: bool = False,
        dropout: float = 0.1,
        gate_init: float = 0.95,
        gate_leak: float = 0.05,
        max_passages: int | None = None,
        question_len: int | None = None,
        passage_len: int | None = None,
        batch_size: int | None = None,
        eval_batch_size: int | None = None,
        support_loss_weight: float | None = None,
        gate_loss_weight: float | None = None,
    ) -> None:
        super().__init__(use_attention=use_attention)
        self.vocab_size = int(vocab_size)
        self.hidden_dim = int(hidden_dim)
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.question_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.passage_encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.attn = AttentionedAttention(
            hidden_dim,
            hidden_dim,
            use_attention=use_attention,
            dropout=dropout,
            gate_init=gate_init,
            gate_leak=gate_leak,
        )
        self.passage_scorer = nn.Sequential(
            nn.Linear(4 * hidden_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.passage_attention_scale = nn.Parameter(torch.tensor(-3.0)) if self.use_attention else None

    def encode_question(self, question_ids: torch.Tensor) -> torch.Tensor:
        _, h = self.question_encoder(self.embedding(question_ids.long()))
        return h.squeeze(0)

    def encode_passages(self, passage_ids: torch.Tensor) -> torch.Tensor:
        batch_size, passages, tokens = passage_ids.shape
        flat = passage_ids.reshape(batch_size * passages, tokens).long()
        _, h = self.passage_encoder(self.embedding(flat))
        return h.squeeze(0).reshape(batch_size, passages, self.hidden_dim)

    def _passage_valid_mask(self, passage_ids: torch.Tensor, passage_mask: torch.Tensor | None = None) -> torch.Tensor:
        valid = passage_ids.long().ne(0).any(dim=-1)
        if passage_mask is not None:
            valid = valid & passage_mask.to(torch.bool)
        return valid

    def _support_logits(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        context: torch.Tensor,
        info: dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mass = (info["attention"] * info.get("attention_gate", torch.ones_like(info["attention"]))).unsqueeze(-1)
        query_p = query.unsqueeze(1).expand_as(memory)
        context_p = context.unsqueeze(1).expand_as(memory)
        features = torch.cat([query_p, memory, query_p * memory, context_p, mass], dim=-1)
        logits = self.passage_scorer(features).squeeze(-1)
        if self.passage_attention_scale is not None:
            passage_count = valid_mask.to(mass.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
            attention_bias = 0.5 * torch.sigmoid(self.passage_attention_scale) * torch.log1p(
                mass.squeeze(-1).clamp_min(0.0) * passage_count
            )
            logits = logits + attention_bias
        return logits.masked_fill(~valid_mask, torch.finfo(logits.dtype).min)

    def forward(self, question_ids: torch.Tensor, passage_ids: torch.Tensor, passage_mask: torch.Tensor | None = None) -> ModelResult:
        query = self.encode_question(question_ids)
        memory = self.encode_passages(passage_ids)
        valid_mask = self._passage_valid_mask(passage_ids, passage_mask)
        context, info = self.attn(query, memory, mask=valid_mask)
        support_logits = self._support_logits(query, memory, context, info, valid_mask)
        aux = self.aux_from_info(info)
        aux["rag_context"] = context.detach()
        aux["passage_logits"] = support_logits
        aux["passage_mask"] = valid_mask.detach()
        aux["attention_mass"] = info["attention"].detach()
        aux["attention_mass"] = (info["attention"] * info["attention_gate"]).detach()
        aux["raw_attention_logits"] = info["attention_logits"]
        return ModelResult(logits=support_logits, aux=aux)


class FiD(_RAGBase):
    model_name = "FiD"

    def __init__(self, *args, fusion_layers: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layer = nn.TransformerEncoderLayer(self.hidden_dim, 4, batch_first=True, activation="gelu")
        self.fusion = nn.TransformerEncoder(layer, num_layers=fusion_layers)

    def encode_passages(self, passage_ids: torch.Tensor) -> torch.Tensor:
        return self.fusion(super().encode_passages(passage_ids))


class LED(_RAGBase):
    model_name = "LED"

    def __init__(self, *args, encoder_layers: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        layer = nn.TransformerEncoderLayer(self.hidden_dim, 8, batch_first=True, activation="gelu")
        self.long_encoder = nn.TransformerEncoder(layer, num_layers=encoder_layers)

    def encode_passages(self, passage_ids: torch.Tensor) -> torch.Tensor:
        return self.long_encoder(super().encode_passages(passage_ids))
