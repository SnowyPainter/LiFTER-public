"""Common factorized event-intensity interface for CTDG link scorers.

The interface models a source-conditioned event process

    lambda(v, t | u, H) = Lambda(t | u, H) * pi(v | u, t, H),

where an existing CTDG backbone supplies the destination logits and a shared,
candidate-independent module supplies the ground-event rate.  Backbones must
be trained through this wrapper; arbitrary BCE/BPR logits are not interpreted
as intensities.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class FactorizedIntensityOutput:
    mark_logits: torch.Tensor
    ground_rate: torch.Tensor


@dataclass
class FactorizedLikelihood:
    loss: torch.Tensor
    mark_nll: torch.Tensor
    time_nll: torch.Tensor


class CandidateIndependentGroundRate(nn.Module):
    """Predict the next source-event rate without candidate identity.

    Only the source ID and counterpart IDs observed before the event are used.
    Query-time elapsed features are deliberately excluded: under the pilot's
    piecewise-constant assumption, the rate is fixed after the previous source
    event and remains constant until the next one.
    """

    def __init__(
        self,
        *,
        num_nodes: int,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        minimum_rate: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.minimum_rate = float(minimum_rate)
        self.node_embedding = nn.Embedding(self.num_nodes, int(hidden_dim))
        self.network = nn.Sequential(
            nn.Linear(2 * int(hidden_dim) + 1, int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(
        self,
        source: torch.Tensor,
        history_nodes: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        source_state = self.node_embedding(
            source.long().clamp(0, self.num_nodes - 1)
        )
        history_state = self.node_embedding(
            history_nodes.long().clamp(0, self.num_nodes - 1)
        )
        valid = (
            torch.ones_like(history_nodes, dtype=torch.bool)
            if history_mask is None
            else history_mask.to(torch.bool)
        )
        weights = valid.to(history_state.dtype)
        count = weights.sum(dim=1, keepdim=True)
        pooled = (
            history_state * weights.unsqueeze(-1)
        ).sum(dim=1) / count.clamp_min(1.0)
        empty = count == 0
        pooled = torch.where(empty, torch.zeros_like(pooled), pooled)
        features = torch.cat((source_state, pooled, torch.log1p(count)), dim=-1)
        return F.softplus(self.network(features).squeeze(-1)) + self.minimum_rate


class FactorizedCTDGIntensity(nn.Module):
    """Attach a common ground-rate model to any CTDG destination scorer."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        num_nodes: int,
        ground_hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.ground_rate = CandidateIndependentGroundRate(
            num_nodes=num_nodes,
            hidden_dim=ground_hidden_dim,
            dropout=dropout,
        )

    def mark_logits(self, *backbone_args: torch.Tensor, **kwargs: object) -> torch.Tensor:
        return self.backbone(*backbone_args, **kwargs).logits

    def rate(
        self,
        source: torch.Tensor,
        history_nodes: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.ground_rate(source, history_nodes, history_mask)

    @staticmethod
    def likelihood(
        mark_logits: torch.Tensor,
        positive_index: torch.Tensor,
        ground_rate: torch.Tensor,
        normalized_delta: torch.Tensor,
    ) -> FactorizedLikelihood:
        """Exact mark likelihood plus exponential ground-process likelihood."""
        if mark_logits.ndim != 2:
            raise ValueError("mark_logits must have shape [events, candidates]")
        mark_nll = F.cross_entropy(
            mark_logits, positive_index.long(), reduction="none"
        )
        rate = ground_rate.clamp_min(1e-12)
        time_nll = -torch.log(rate) + rate * normalized_delta
        return FactorizedLikelihood(
            loss=(mark_nll + time_nll).mean(),
            mark_nll=mark_nll,
            time_nll=time_nll,
        )
