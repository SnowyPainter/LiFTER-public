from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .attention import SinusoidalTimeEncoding, AttentionedAttention
from .base import ModelResult, AttentionModel


class _TKGBase(AttentionModel):
    """Common utilities for temporal KG tail forecasting.

    The local TKG references use the same benchmark interface:
    ``(head, relation, time, history_head, history_relation, history_tail,
    history_time, history_mask) -> tail logits``.  The subclasses keep the
    paper-level modeling bias distinct:

    - RE-NET: recurrent history encoder over relation/entity events.
    - xERTE: query-conditioned multi-hop temporal evidence expansion.
    - CyGNet: generate/copy mixture over entities and historical vocabulary.
    """

    model_name = "tkg_base"

    def __init__(
        self,
        *,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 128,
        time_dim: int = 32,
        use_attention: bool = False,
        dropout: float = 0.1,
        gate_init: float = 0.95,
        gate_leak: float = 0.05,
        learning_rate: float | None = None,
        attention_learning_rate: float | None = None,
        gradient_clip: float | None = None,
    ) -> None:
        super().__init__(use_attention=use_attention)
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)
        self.hidden_dim = int(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.entity_embedding = nn.Embedding(self.num_entities, hidden_dim)
        self.relation_embedding = nn.Embedding(self.num_relations, hidden_dim)
        self.time_encoder = SinusoidalTimeEncoding(time_dim)

        self.query_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim + time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.event_proj = nn.Sequential(
            nn.Linear(3 * hidden_dim + time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.state_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.entity_bias = nn.Parameter(torch.zeros(self.num_entities))
        self.attention_copy_scale = nn.Parameter(torch.tensor(1.0)) if self.use_attention else None
        self.tail_path_mode = "attention" if self.use_attention else "closed"

    def query_state(self, head: torch.Tensor, relation: torch.Tensor, timestamp: torch.Tensor) -> torch.Tensor:
        h = self.entity_embedding(head.long().clamp(min=0, max=self.num_entities - 1))
        r = self.relation_embedding(relation.long().clamp(min=0, max=self.num_relations - 1))
        t = self.time_encoder(timestamp.float())
        return self.query_proj(torch.cat([h, r, t], dim=-1))

    def encode_events(
        self,
        history_head: torch.Tensor,
        history_relation: torch.Tensor,
        history_tail: torch.Tensor,
        history_time: torch.Tensor,
        query_time: torch.Tensor,
    ) -> torch.Tensor:
        hh = self.entity_embedding(history_head.long().clamp(min=0, max=self.num_entities - 1))
        hr = self.relation_embedding(history_relation.long().clamp(min=0, max=self.num_relations - 1))
        ht = self.entity_embedding(history_tail.long().clamp(min=0, max=self.num_entities - 1))
        dt = self.time_encoder((query_time.unsqueeze(1) - history_time.float()).clamp_min(0.0))
        return self.event_proj(torch.cat([hh, hr, ht, dt], dim=-1))

    def score_tails(self, state: torch.Tensor, relation: torch.Tensor) -> torch.Tensor:
        rel = self.relation_embedding(relation.long().clamp(min=0, max=self.num_relations - 1))
        conditioned = self.state_proj(torch.cat([state, rel], dim=-1))
        return torch.matmul(conditioned * rel, self.entity_embedding.weight.transpose(0, 1)) + self.entity_bias

    def history_copy_mass(
        self,
        history_tail: torch.Tensor,
        history_time: torch.Tensor,
        timestamp: torch.Tensor,
        history_mask: torch.Tensor | None,
        *,
        evidence_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid_tail = history_tail.long().clamp(min=0, max=self.num_entities - 1)
        gap = (timestamp.float().unsqueeze(1) - history_time.float()).clamp_min(0.0)
        recency = torch.exp(-torch.log1p(gap) / 8.0).to(self.entity_bias.dtype)
        if history_mask is not None:
            recency = recency.masked_fill(~history_mask, 0.0)
        if evidence_weight is not None:
            recency = recency * evidence_weight.to(recency.dtype)
        recency = recency / recency.sum(dim=1, keepdim=True).clamp_min(1.0e-9)
        return torch.zeros(
            history_tail.shape[0],
            self.num_entities,
            device=history_tail.device,
            dtype=recency.dtype,
        ).scatter_add(1, valid_tail, recency)

    def attentioned_tail_bias(
        self,
        history_tail: torch.Tensor,
        history_time: torch.Tensor,
        timestamp: torch.Tensor,
        history_mask: torch.Tensor | None,
        info: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.attention_copy_scale is None:
            empty = torch.zeros(
                history_tail.shape[0],
                self.num_entities,
                device=history_tail.device,
                dtype=self.entity_bias.dtype,
            )
            return empty, empty
        evidence_weight = self.tail_evidence_weight(info)
        copy_mass = self.history_copy_mass(
            history_tail,
            history_time,
            timestamp,
            history_mask,
            evidence_weight=evidence_weight,
        )
        scale = torch.nn.functional.softplus(self.attention_copy_scale)
        bias = scale * torch.log1p(copy_mass * float(self.num_entities))
        return bias, copy_mass

    def tail_evidence_weight(self, info: dict[str, torch.Tensor]) -> torch.Tensor:
        attention = info["attention"]
        if self.tail_path_mode == "open_no_gate":
            return attention
        return attention * info["attention_gate"]


class RENET(_TKGBase):
    """RE-NET-style recurrent event-history model."""

    model_name = "RE-NET"

    def __init__(self, *args, dropout: float = 0.1, gate_init: float = 0.95, gate_leak: float = 0.05, **kwargs) -> None:
        super().__init__(*args, dropout=dropout, gate_init=gate_init, gate_leak=gate_leak, **kwargs)
        self.local_encoder = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first=True)
        self.relation_aux_head = nn.Linear(self.hidden_dim, self.num_relations)
        self.attn = AttentionedAttention(
            self.hidden_dim,
            self.hidden_dim,
            use_attention=self.use_attention,
            dropout=dropout,
            gate_init=gate_init,
            gate_leak=gate_leak,
        )

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        timestamp: torch.Tensor,
        history_head: torch.Tensor,
        history_relation: torch.Tensor,
        history_tail: torch.Tensor,
        history_time: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> ModelResult:
        query = self.query_state(head, relation, timestamp)
        events = self.encode_events(history_head, history_relation, history_tail, history_time, timestamp)
        encoded, _ = self.local_encoder(events)
        context, info = self.attn(query, encoded, mask=history_mask)
        state = query + context
        logits = self.score_tails(state, relation)
        tail_bias, copy_mass = self.attentioned_tail_bias(history_tail, history_time, timestamp, history_mask, info)
        logits = logits + tail_bias
        aux = self.aux_from_info(info)
        aux["raw_attention_logits"] = info["attention_logits"]
        aux["relation_logits"] = self.relation_aux_head(state)
        aux["attention_tail_mass_mean"] = copy_mass.detach().sum(dim=1).mean()
        return ModelResult(logits=logits, aux=aux)


class XERTE(_TKGBase):
    """xERTE-style query-conditioned multi-hop temporal expansion."""

    model_name = "xERTE"

    def __init__(
        self,
        *args,
        expansion_layers: int = 2,
        dropout: float = 0.1,
        gate_init: float = 0.95,
        gate_leak: float = 0.05,
        **kwargs,
    ) -> None:
        super().__init__(*args, dropout=dropout, gate_init=gate_init, gate_leak=gate_leak, **kwargs)
        self.expansion_layers = int(expansion_layers)
        self.hop_attn = nn.ModuleList(
            [
                AttentionedAttention(
                    self.hidden_dim,
                    self.hidden_dim,
                    use_attention=self.use_attention,
                    dropout=dropout,
                    gate_init=gate_init,
                    gate_leak=gate_leak,
                )
                for _ in range(self.expansion_layers)
            ]
        )
        self.hop_update = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                    nn.LayerNorm(self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(self.expansion_layers)
            ]
        )

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        timestamp: torch.Tensor,
        history_head: torch.Tensor,
        history_relation: torch.Tensor,
        history_tail: torch.Tensor,
        history_time: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> ModelResult:
        query = self.query_state(head, relation, timestamp)
        frontier = self.encode_events(history_head, history_relation, history_tail, history_time, timestamp)
        infos: list[dict[str, torch.Tensor]] = []
        state = query
        for attn, update in zip(self.hop_attn, self.hop_update):
            context, info = attn(state, frontier, mask=history_mask)
            state = state + update(torch.cat([state, context], dim=-1))
            frontier = frontier + state.unsqueeze(1)
            infos.append(info)
        logits = self.score_tails(state, relation)
        if infos:
            tail_bias, copy_mass = self.attentioned_tail_bias(history_tail, history_time, timestamp, history_mask, infos[-1])
            logits = logits + tail_bias
        else:
            copy_mass = torch.zeros(
                history_tail.shape[0],
                self.num_entities,
                device=history_tail.device,
                dtype=self.entity_bias.dtype,
            )
        aux = self.aux_from_info(infos[-1] if infos else {})
        if infos:
            aux["attention_gate_mean"] = torch.stack([info["attention_gate"].detach().mean() for info in infos]).mean()
            aux["raw_attention_logits"] = infos[-1]["attention_logits"]
            aux["attention_tail_mass_mean"] = copy_mass.detach().sum(dim=1).mean()
        return ModelResult(logits=logits, aux=aux)


class CyGNet(_TKGBase):
    """CyGNet-style generate/copy model over historical entity vocabulary."""

    model_name = "CyGNet"

    def __init__(
        self,
        *args,
        dropout: float = 0.1,
        gate_init: float = 0.95,
        gate_leak: float = 0.05,
        learning_rate: float | None = None,
        attention_learning_rate: float | None = None,
        gradient_clip: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, dropout=dropout, gate_init=gate_init, gate_leak=gate_leak, **kwargs)
        self.copy_attn = AttentionedAttention(
            self.hidden_dim,
            self.hidden_dim,
            use_attention=self.use_attention,
            dropout=dropout,
            gate_init=gate_init,
            gate_leak=gate_leak,
        )
        self.copy_gate = nn.Sequential(nn.Linear(self.hidden_dim, 1), nn.Sigmoid())
        self.generate_temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        head: torch.Tensor,
        relation: torch.Tensor,
        timestamp: torch.Tensor,
        history_head: torch.Tensor,
        history_relation: torch.Tensor,
        history_tail: torch.Tensor,
        history_time: torch.Tensor,
        history_mask: torch.Tensor | None = None,
    ) -> ModelResult:
        query = self.query_state(head, relation, timestamp)
        events = self.encode_events(history_head, history_relation, history_tail, history_time, timestamp)
        context, info = self.copy_attn(query, events, mask=history_mask)
        state = query + context

        temperature = F.softplus(self.generate_temperature).clamp_min(0.2)
        gen_logits = (self.score_tails(query, relation) / temperature).clamp(min=-40.0, max=40.0)
        evidence_weight = self.tail_evidence_weight(info)
        copy_mass = self.history_copy_mass(
            history_tail,
            history_time,
            timestamp,
            history_mask,
            evidence_weight=evidence_weight,
        )
        copy_logits = torch.log(copy_mass.clamp_min(1.0e-9)).clamp(min=-40.0, max=40.0)
        gate = self.copy_gate(state)
        logits = torch.logaddexp(
            torch.log1p(-gate).clamp_min(-20.0) + gen_logits,
            torch.log(gate).clamp_min(-20.0) + copy_logits,
        )
        aux = self.aux_from_info(info)
        aux["raw_attention_logits"] = info["attention_logits"]
        aux["copy_gate_mean"] = gate.detach().mean()
        return ModelResult(logits=logits, aux=aux)
