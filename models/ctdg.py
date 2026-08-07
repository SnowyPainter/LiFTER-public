from __future__ import annotations

import importlib.util
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .attention import SinusoidalTimeEncoding, AttentionedAttention
from .base import ModelResult, AttentionModel
from .attention_block import EdgeConditionedAttentionBlock


class _PairScorer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.scale = math.sqrt(float(hidden_dim))

    def forward(self, src_state: torch.Tensor, dst_state: torch.Tensor, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        dot = (src_state * dst_state).sum(dim=-1) / self.scale
        mlp = self.mlp(torch.cat([src_state, dst_state, query, context], dim=-1)).squeeze(-1)
        return dot + mlp


class MultiScaleKernelAggregation(nn.Module):
    """Multiplicity-preserving candidate/history aggregation.

    For destination embedding q and historical neighbor embeddings k_j:

        K(q, k_j) = exp(tau * (cos(q, k_j) - 1))
        E_m(q, H) = log(1 + sum_j K(q, k_j) exp(-age_j / scale_m))

    Unlike softmax attention, E_m is a measure rather than a probability:
    repeating a compatible event increases its mass. Multiple learnable scales
    adapt the operator to dataset-specific recurrence horizons, while the
    projection learns how much each horizon matters.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        history_len: int,
        scale_count: int = 4,
        scale_init: str = "full",
        learnable_scales: bool = False,
        evidence_mode: str = "mass",
        kernel_temperature: float = 12.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        scales = self.initial_scales(history_len, scale_count, scale_init)
        self.register_buffer("initial_scale_values", scales.clone())
        self.log_scales = nn.Parameter(torch.log(scales), requires_grad=learnable_scales)
        self.scale_init = str(scale_init)
        self.learnable_scales = bool(learnable_scales)
        if evidence_mode not in {"mass", "mean", "softmax"}:
            raise ValueError("evidence_mode must be mass, mean, or softmax")
        self.evidence_mode = str(evidence_mode)
        self.kernel_temperature = float(kernel_temperature)
        self.projection = nn.Sequential(
            nn.Linear(scale_count, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def initial_scales(history_len: int, scale_count: int, scale_init: str) -> torch.Tensor:
        if history_len < 1:
            raise ValueError("history_len must be positive")
        if scale_count < 1:
            raise ValueError("scale_count must be positive")
        if scale_init not in {"short", "full", "long"}:
            raise ValueError("scale_init must be short, full, or long")
        midpoint = math.sqrt(float(history_len))
        lower, upper = {
            "short": (1.0, midpoint),
            "full": (1.0, float(history_len)),
            "long": (midpoint, float(history_len)),
        }[scale_init]
        if scale_count == 1:
            return torch.tensor([math.sqrt(lower * upper)], dtype=torch.float32)
        return torch.exp(torch.linspace(math.log(lower), math.log(upper), scale_count))

    def forward(
        self,
        candidate: torch.Tensor,
        history: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = (
            torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
            if mask is None
            else mask.to(device=history.device, dtype=torch.bool)
        )
        candidate_unit = F.normalize(candidate.float(), dim=-1)
        history_unit = F.normalize(history.float(), dim=-1)
        cosine = torch.sum(candidate_unit.unsqueeze(1) * history_unit, dim=-1).clamp(-1.0, 1.0)
        kernel = torch.exp(self.kernel_temperature * (cosine - 1.0)) * valid.to(cosine.dtype)

        length = history.shape[1]
        age = torch.arange(length - 1, -1, -1, device=history.device, dtype=cosine.dtype)
        learned_scales = torch.exp(self.log_scales).to(device=history.device, dtype=cosine.dtype).clamp_min(1.0)
        decay = torch.exp(-age[:, None] / learned_scales[None, :])
        weighted = kernel[:, :, None] * decay[None, :, :]
        if self.evidence_mode == "mass":
            raw_evidence = weighted.sum(dim=1)
        elif self.evidence_mode == "mean":
            normalizer = (
                valid[:, :, None].to(weighted.dtype) * decay[None, :, :]
            ).sum(dim=1).clamp_min(1e-12)
            raw_evidence = weighted.sum(dim=1) / normalizer
        else:
            logits = (
                self.kernel_temperature * (cosine[:, :, None] - 1.0)
                - age[None, :, None] / learned_scales[None, None, :]
            )
            logits = logits.masked_fill(~valid[:, :, None], -1e9)
            attention = torch.softmax(logits, dim=1) * valid[:, :, None].to(logits.dtype)
            attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-12)
            raw_evidence = (attention * kernel[:, :, None]).sum(dim=1)
        evidence = torch.log1p(raw_evidence)
        context = self.output_scale * self.projection(evidence.to(candidate.dtype))
        return context, evidence


class TargetAgnosticMultiScalePooling(nn.Module):
    """Multi-timescale source-history pooling without candidate access."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        history_len: int,
        scale_count: int = 4,
        include_activity: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        scales = MultiScaleKernelAggregation.initial_scales(
            history_len, scale_count, "full"
        )
        self.register_buffer("scale_values", scales)
        self.include_activity = bool(include_activity)
        input_dim = scale_count * hidden_dim + (scale_count if include_activity else 0)
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        history: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = (
            torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
            if mask is None
            else mask.to(device=history.device, dtype=torch.bool)
        )
        length = history.shape[1]
        age = torch.arange(
            length - 1, -1, -1, device=history.device, dtype=history.dtype
        )
        scales = self.scale_values.to(device=history.device, dtype=history.dtype)
        decay = torch.exp(-age[:, None] / scales[None, :])
        weights = valid[:, :, None].to(history.dtype) * decay[None, :, :]
        normalizer = weights.sum(dim=1).clamp_min(1e-12)
        content = torch.einsum("blm,blh->bmh", weights, history)
        content = content / normalizer[:, :, None]
        features = [content.flatten(start_dim=1)]
        activity = torch.log1p(normalizer)
        if self.include_activity:
            features.append(activity)
        context = self.output_scale * self.projection(torch.cat(features, dim=-1))
        return context, activity


class _CTDGBase(AttentionModel):
    model_name = "ctdg_base"

    def __init__(
        self,
        *,
        num_nodes: int,
        edge_feat_dim: int = 1,
        hidden_dim: int = 128,
        time_dim: int = 32,
        use_attention: bool = False,
        dropout: float = 0.1,
        gate_init: float = 0.95,
        gate_leak: float = 0.05,
        aggregation_operator: str = "softmax",
        neural_history_len: int | None = None,
        recurrence_history_len: int = 320,
        recurrence_scale_count: int = 4,
        recurrence_scale_init: str = "full",
        learnable_recurrence_scales: bool = False,
        recurrence_evidence_mode: str = "mass",
        target_agnostic_include_activity: bool = True,
        kernel_temperature: float = 12.0,
    ) -> None:
        super().__init__(use_attention=use_attention)
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.edge_feat_dim = int(edge_feat_dim)
        self.aggregation_operator = str(aggregation_operator)
        self.neural_history_len = None if neural_history_len is None else int(neural_history_len)
        if self.neural_history_len is not None and self.neural_history_len <= 0:
            raise ValueError("neural_history_len must be positive")
        if self.aggregation_operator not in {
            "softmax",
            "multiscale_kernel",
            "multiscale_pool",
        }:
            raise ValueError(
                "aggregation_operator must be softmax, multiscale_kernel, or multiscale_pool"
            )
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        self.time_encoder = SinusoidalTimeEncoding(time_dim)
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)
        self.memory_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim + time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.query_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim + time_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.attn = AttentionedAttention(
            hidden_dim,
            hidden_dim,
            use_attention=use_attention,
            dropout=dropout,
            gate_init=gate_init,
            gate_leak=gate_leak,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.scorer = _PairScorer(hidden_dim, dropout)
        self.kernel_aggregation = (
            MultiScaleKernelAggregation(
                hidden_dim,
                history_len=int(recurrence_history_len),
                scale_count=int(recurrence_scale_count),
                scale_init=str(recurrence_scale_init),
                learnable_scales=bool(learnable_recurrence_scales),
                evidence_mode=str(recurrence_evidence_mode),
                kernel_temperature=kernel_temperature,
                dropout=dropout,
            )
            if self.aggregation_operator == "multiscale_kernel"
            else None
        )
        self.target_agnostic_pool = (
            TargetAgnosticMultiScalePooling(
                hidden_dim,
                history_len=int(recurrence_history_len),
                scale_count=int(recurrence_scale_count),
                include_activity=bool(target_agnostic_include_activity),
                dropout=dropout,
            )
            if self.aggregation_operator == "multiscale_pool"
            else None
        )
        self.edge_attention = (
            EdgeConditionedAttentionBlock(
                hidden_dim,
                time_dim,
                dropout=dropout,
                gate_init=gate_init,
                gate_leak=gate_leak,
            )
            if use_attention
            else None
        )

    def _truncate_neural_history(
        self,
        history_nodes: torch.Tensor,
        history_times: torch.Tensor,
        history_edge_feats: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.neural_history_len is None or history_nodes.shape[1] <= self.neural_history_len:
            return history_nodes, history_times, history_edge_feats, history_mask
        width = self.neural_history_len
        return (
            history_nodes[:, -width:],
            history_times[:, -width:],
            history_edge_feats[:, -width:],
            None if history_mask is None else history_mask[:, -width:],
        )

    def _encode_raw_history(
        self,
        history_nodes: torch.Tensor,
        history_times: torch.Tensor,
        history_edge_feats: torch.Tensor,
        query_time: torch.Tensor,
    ) -> torch.Tensor:
        node_h = self.node_embedding(history_nodes.long().clamp_min(0).clamp_max(self.num_nodes - 1))
        edge_h = self.edge_encoder(history_edge_feats.float())
        dt_h = self.time_encoder((query_time.unsqueeze(1) - history_times.float()).clamp_min(0.0))
        return self.memory_proj(torch.cat([node_h, edge_h, dt_h], dim=-1))

    def encode_memory(
        self,
        history_nodes: torch.Tensor,
        history_times: torch.Tensor,
        history_edge_feats: torch.Tensor,
        query_time: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        history_nodes, history_times, history_edge_feats, history_mask = self._truncate_neural_history(
            history_nodes, history_times, history_edge_feats, history_mask
        )
        return self._encode_raw_history(history_nodes, history_times, history_edge_feats, query_time), history_mask

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        timestamp: torch.Tensor,
        history_nodes: torch.Tensor,
        history_times: torch.Tensor,
        history_edge_feats: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        dst_history_nodes: torch.Tensor | None = None,
        dst_history_times: torch.Tensor | None = None,
        dst_history_edge_feats: torch.Tensor | None = None,
        dst_history_mask: torch.Tensor | None = None,
    ) -> ModelResult:
        src_h = self.node_embedding(src.long().clamp_min(0).clamp_max(self.num_nodes - 1))
        dst_h = self.node_embedding(dst.long().clamp_min(0).clamp_max(self.num_nodes - 1))
        time_h = self.time_encoder(timestamp.float())
        query = self.query_proj(torch.cat([src_h, dst_h, time_h], dim=-1))
        raw_memory = self._encode_raw_history(history_nodes, history_times, history_edge_feats, timestamp)
        if self.target_agnostic_pool is None:
            memory, memory_mask = self.encode_memory(
                history_nodes,
                history_times,
                history_edge_feats,
                timestamp,
                history_mask,
            )
            context, info = self.attn(query, memory, mask=memory_mask)
        else:
            context, pool_activity = self.target_agnostic_pool(
                raw_memory, history_mask
            )
            info = {"target_agnostic_activity": pool_activity}
        if self.kernel_aggregation is not None:
            history_identity = self.node_embedding(
                history_nodes.long().clamp_min(0).clamp_max(self.num_nodes - 1)
            )
            kernel_context, kernel_evidence = self.kernel_aggregation(dst_h, history_identity, history_mask)
            context = context + kernel_context
            info["kernel_evidence"] = kernel_evidence
        src_state = self.output_norm(src_h + context)
        if self.edge_attention is not None and dst_history_nodes is not None and dst_history_times is not None:
            if dst_history_edge_feats is None:
                dst_history_edge_feats = torch.zeros_like(history_edge_feats)
            if dst_history_mask is None:
                dst_history_mask = torch.ones_like(dst_history_nodes, dtype=torch.bool)
            dst_memory = self._encode_raw_history(dst_history_nodes, dst_history_times, dst_history_edge_feats, timestamp)
            src_delta_t = (timestamp.unsqueeze(1) - history_times.float()).clamp_min(0.0)
            dst_delta_t = (timestamp.unsqueeze(1) - dst_history_times.float()).clamp_min(0.0)
            src_stats = self._edge_attention_stats(
                center_ids=src,
                counterpart_ids=dst,
                history_nodes=history_nodes,
                delta_t=src_delta_t,
                mask=history_mask,
            )
            dst_stats = self._edge_attention_stats(
                center_ids=dst,
                counterpart_ids=src,
                history_nodes=dst_history_nodes,
                delta_t=dst_delta_t,
                mask=dst_history_mask,
            )
            src_delta, dst_delta, edge_info = self.edge_attention(
                src_h=src_state,
                dst_h=dst_h,
                time_h=time_h,
                src_memory=raw_memory,
                dst_memory=dst_memory,
                src_delta_t_enc=self.time_encoder(src_delta_t),
                dst_delta_t_enc=self.time_encoder(dst_delta_t),
                src_stats=src_stats,
                dst_stats=dst_stats,
                src_mask=history_mask if history_mask is not None else torch.ones_like(history_nodes, dtype=torch.bool),
                dst_mask=dst_history_mask,
            )
            src_state = self.output_norm(src_state + src_delta)
            dst_h = self.output_norm(dst_h + dst_delta)
            info.update(edge_info)
        logits = self.scorer(src_state, dst_h, query, context)
        return ModelResult(logits=logits, aux=self.aux_from_info(info))

    def _edge_attention_stats(
        self,
        *,
        center_ids: torch.Tensor,
        counterpart_ids: torch.Tensor,
        history_nodes: torch.Tensor,
        delta_t: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        valid = mask if mask is not None else torch.ones_like(history_nodes, dtype=torch.bool)
        counterpart_hit = (history_nodes.long() == counterpart_ids.long().unsqueeze(1)).to(delta_t.dtype) * valid.to(delta_t.dtype)
        recency = torch.exp(-torch.log1p(delta_t.float()).clamp_min(0.0) / 8.0) * counterpart_hit
        valid_ratio = valid.to(delta_t.dtype).mean(dim=1, keepdim=True).expand_as(delta_t)
        return torch.stack([counterpart_hit, recency, valid_ratio], dim=-1)


class TGAT(_CTDGBase):
    """TGAT-style temporal neighbor attention.

    The benchmark-native version keeps TGAT's important contract: query-time
    node-pair attention over source temporal neighbors with functional time
    encodings.  Attention on/off changes the value aggregation inside that
    temporal attention path.
    """

    model_name = "TGAT"


class DyGFormer(_CTDGBase):
    """DyGFormer-style patch encoder over temporal neighbor sequences."""

    model_name = "DyGFormer"

    def __init__(
        self,
        *args,
        num_layers: int = 2,
        num_heads: int = 4,
        patch_size: int = 2,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.patch_size = max(1, int(patch_size))
        self.patch_proj = nn.Sequential(
            nn.Linear(self.patch_size * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * self.hidden_dim,
            dropout=kwargs.get("dropout", 0.1),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.patch_transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

    def encode_memory(
        self,
        history_nodes: torch.Tensor,
        history_times: torch.Tensor,
        history_edge_feats: torch.Tensor,
        query_time: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        history_nodes, history_times, history_edge_feats, history_mask = self._truncate_neural_history(
            history_nodes, history_times, history_edge_feats, history_mask
        )
        memory = self._encode_raw_history(history_nodes, history_times, history_edge_feats, query_time)
        batch_size, seq_len, hidden_dim = memory.shape
        pad = (-seq_len) % self.patch_size
        if pad:
            memory = F.pad(memory, (0, 0, pad, 0))
            if history_mask is not None:
                history_mask = F.pad(history_mask, (pad, 0), value=False)
        patches = memory.reshape(batch_size, -1, self.patch_size * hidden_dim)
        patch_memory = self.patch_proj(patches)
        if history_mask is None:
            patch_mask = None
        else:
            patch_mask = history_mask.reshape(batch_size, -1, self.patch_size).any(dim=-1)
            empty = ~patch_mask.any(dim=1)
            if bool(empty.any()):
                patch_mask = patch_mask.clone()
                patch_mask[empty, 0] = True
                patch_memory = patch_memory.clone()
                patch_memory[empty, 0] = 0.0
        src_key_padding_mask = None if patch_mask is None else ~patch_mask
        patch_memory = self.patch_transformer(patch_memory, src_key_padding_mask=src_key_padding_mask)
        return patch_memory, patch_mask


class DyGFormerLHA(DyGFormer):
    """DyGFormer backbone augmented with TAMI Link History Aggregation."""

    model_name = "DyGFormer w/ LHA"

    def __init__(self, *args: Any, gamma: float = 0.9, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gamma = float(gamma)
        self.current_pair = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.historical_decoder = nn.Linear(2 * self.hidden_dim, 1)
        self.pair_memory: dict[tuple[int, int], torch.Tensor] = {}
        self.pair_memory_by_source: dict[int, dict[int, torch.Tensor]] = {}

    def reset_pair_memory(self) -> None:
        self.pair_memory.clear()
        self.pair_memory_by_source.clear()

    def backup_pair_memory(self) -> dict[tuple[int, int], torch.Tensor]:
        return {key: value.clone() for key, value in self.pair_memory.items()}

    def restore_pair_memory(
        self, memory: dict[tuple[int, int], torch.Tensor]
    ) -> None:
        self.pair_memory = {key: value.clone() for key, value in memory.items()}
        self.pair_memory_by_source = {}
        for (source, destination), value in self.pair_memory.items():
            self.pair_memory_by_source.setdefault(source, {})[destination] = value

    def _historical_states(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if self.training or len(src) == 1:
            keys = zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist())
            return torch.stack(
                [
                    self.pair_memory.get(
                        key, torch.zeros(self.hidden_dim)
                    ).to(reference.device)
                    for key in keys
                ]
            )

        historical = torch.zeros_like(reference)
        for source in torch.unique(src).detach().cpu().tolist():
            source_memory = self.pair_memory_by_source.get(source)
            if not source_memory:
                continue
            source_rows = torch.nonzero(src == source, as_tuple=False).squeeze(-1)
            dense = torch.zeros(
                self.num_nodes,
                self.hidden_dim,
                dtype=reference.dtype,
                device=reference.device,
            )
            destinations = torch.tensor(
                list(source_memory), dtype=torch.long, device=reference.device
            )
            values = torch.stack(
                [source_memory[int(destination)] for destination in destinations.cpu()]
            ).to(reference.device)
            dense[destinations] = values
            historical[source_rows] = dense[dst[source_rows].long()]
        return historical

    def _node_states(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        timestamp: torch.Tensor,
        history_nodes: torch.Tensor,
        history_times: torch.Tensor,
        history_edge_feats: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_h = self.node_embedding(
            src.long().clamp_min(0).clamp_max(self.num_nodes - 1)
        )
        dst_h = self.node_embedding(
            dst.long().clamp_min(0).clamp_max(self.num_nodes - 1)
        )
        time_h = self.time_encoder(timestamp.float())
        query = self.query_proj(torch.cat([src_h, dst_h, time_h], dim=-1))
        # Full-catalog evaluation repeats the same event history for every
        # candidate destination. Encode each contiguous, identical history
        # once while retaining a distinct candidate-conditioned query.
        if not self.training and len(src) > 1:
            same_as_previous = (
                (src[1:] == src[:-1])
                & (timestamp[1:] == timestamp[:-1])
                & (history_nodes[1:] == history_nodes[:-1]).all(dim=1)
                & (history_times[1:] == history_times[:-1]).all(dim=1)
            )
            if history_mask is not None:
                same_as_previous &= (
                    history_mask[1:] == history_mask[:-1]
                ).all(dim=1)
            group_start = torch.ones(
                len(src), dtype=torch.bool, device=src.device
            )
            group_start[1:] = ~same_as_previous
            representatives = torch.nonzero(
                group_start, as_tuple=False
            ).squeeze(-1)
            compact_memory, compact_mask = self.encode_memory(
                history_nodes[representatives],
                history_times[representatives],
                history_edge_feats[representatives],
                timestamp[representatives],
                (
                    history_mask[representatives]
                    if history_mask is not None
                    else None
                ),
            )
            group_index = group_start.long().cumsum(dim=0) - 1
            if not self.attn.use_attention:
                projected_query = self.attn.query_proj(query)
                context = torch.empty_like(projected_query)
                for group in range(len(representatives)):
                    rows = torch.nonzero(
                        group_index == group, as_tuple=False
                    ).squeeze(-1)
                    keys = self.attn.key_proj(compact_memory[group])
                    values = self.attn.value_proj(compact_memory[group])
                    logits = projected_query[rows] @ keys.transpose(0, 1)
                    logits = logits / math.sqrt(float(keys.shape[-1]))
                    mask = compact_mask[group]
                    if mask is not None:
                        logits = logits.masked_fill(
                            ~mask.unsqueeze(0),
                            torch.finfo(logits.dtype).min,
                        )
                        if not bool(mask.any()):
                            logits = torch.zeros_like(logits)
                    weights = torch.softmax(logits, dim=-1)
                    if mask is not None:
                        weights = weights * mask.unsqueeze(0).to(weights.dtype)
                    context[rows] = self.attn.dropout(weights) @ values
                return self.output_norm(src_h + context), dst_h
            memory = compact_memory[group_index]
            memory_mask = compact_mask[group_index]
        else:
            memory, memory_mask = self.encode_memory(
                history_nodes,
                history_times,
                history_edge_feats,
                timestamp,
                history_mask,
            )
        context, _ = self.attn(query, memory, mask=memory_mask)
        return self.output_norm(src_h + context), dst_h

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        timestamp: torch.Tensor,
        history_nodes: torch.Tensor,
        history_times: torch.Tensor,
        history_edge_feats: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        dst_history_nodes: torch.Tensor | None = None,
        dst_history_times: torch.Tensor | None = None,
        dst_history_edge_feats: torch.Tensor | None = None,
        dst_history_mask: torch.Tensor | None = None,
        *,
        update_memory: bool = False,
    ) -> ModelResult:
        del (
            dst_history_nodes,
            dst_history_times,
            dst_history_edge_feats,
            dst_history_mask,
        )
        src_state, dst_state = self._node_states(
            src,
            dst,
            timestamp,
            history_nodes,
            history_times,
            history_edge_feats,
            history_mask,
        )
        current = self.current_pair(torch.cat([src_state, dst_state], dim=-1))
        keys = list(zip(src.detach().cpu().tolist(), dst.detach().cpu().tolist()))
        historical = self._historical_states(src, dst, current)
        logits = self.historical_decoder(
            torch.cat([current, historical], dim=-1)
        ).squeeze(-1)
        if update_memory:
            updated = (
                self.gamma * current.detach()
                + (1.0 - self.gamma) * historical.detach()
            ).cpu()
            for key, value in zip(keys, updated):
                self.pair_memory[key] = value
                self.pair_memory_by_source.setdefault(key[0], {})[key[1]] = value
        return ModelResult(
            logits=logits,
            aux={"pair_history_norm": historical.norm(dim=-1)},
        )


class _MixerBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.channel = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.token_gate = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.channel(x)
        scores = self.token_gate(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            empty = ~mask.any(dim=-1, keepdim=True)
            scores = torch.where(empty, torch.zeros_like(scores), scores)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        global_token = (weights * x).sum(dim=1, keepdim=True)
        return x + global_token


class GraphMixer(_CTDGBase):
    """GraphMixer-style MLP mixer over fixed recent temporal tokens."""

    model_name = "GraphMixer"

    def __init__(self, *args, mixer_layers: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dropout = float(kwargs.get("dropout", 0.1))
        self.mixers = nn.ModuleList([_MixerBlock(self.hidden_dim, dropout) for _ in range(mixer_layers)])

    def encode_memory(
        self,
        history_nodes: torch.Tensor,
        history_times: torch.Tensor,
        history_edge_feats: torch.Tensor,
        query_time: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        history_nodes, history_times, history_edge_feats, history_mask = self._truncate_neural_history(
            history_nodes, history_times, history_edge_feats, history_mask
        )
        memory = self._encode_raw_history(history_nodes, history_times, history_edge_feats, query_time)
        for mixer in self.mixers:
            memory = mixer(memory, history_mask)
        return memory, history_mask


EXTERNAL_ROOT = Path(__file__).resolve().parents[1] / "external"
TGN_OFFICIAL_COMMIT = "d55bbe678acabb9fc3879c408fd1f2e15919667c"
GRAPHMIXER_OFFICIAL_COMMIT = "c84f1e0bee4eed848872a966b8166d741e240713"


@contextmanager
def _prepend_external_path(path: Path) -> Iterator[None]:
    value = str(path)
    sys.path.insert(0, value)
    try:
        yield
    finally:
        if value in sys.path:
            sys.path.remove(value)


def _load_official_graphmixer_module() -> Any:
    path = EXTERNAL_ROOT / "GraphMixer" / "model.py"
    if not path.exists():
        raise FileNotFoundError(
            f"official GraphMixer checkout missing: {path}; "
            "clone https://github.com/CongWeilin/GraphMixer"
        )
    spec = importlib.util.spec_from_file_location("official_graphmixer_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_official_tgn_class() -> type[nn.Module]:
    root = EXTERNAL_ROOT / "tgn"
    path = root / "model" / "tgn.py"
    if not path.exists():
        raise FileNotFoundError(
            f"official TGN checkout missing: {path}; "
            "clone https://github.com/twitter-research/tgn"
        )
    with _prepend_external_path(root):
        spec = importlib.util.spec_from_file_location("official_tgn_model", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.TGN


class OfficialGraphMixerBaseline(nn.Module):
    """Official GraphMixer modules adapted to the repository CTDG batch API.

    Source and destination histories are encoded independently, so the
    destination never selects or reweights the source history.
    """

    model_name = "GraphMixer"
    official_commit = GRAPHMIXER_OFFICIAL_COMMIT

    def __init__(
        self,
        *,
        num_nodes: int,
        history_len: int,
        hidden_dim: int = 64,
        time_dim: int = 32,
        mixer_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        official = _load_official_graphmixer_module()
        self.num_nodes = int(num_nodes)
        self.history_len = int(history_len)
        self.hidden_dim = int(hidden_dim)
        self.node_embedding = nn.Embedding(self.num_nodes, self.hidden_dim)
        self.history_encoder = official.MLPMixer(
            per_graph_size=self.history_len,
            time_channels=int(time_dim),
            input_channels=1,
            hidden_channels=self.hidden_dim,
            out_channels=self.hidden_dim,
            num_layers=int(mixer_layers),
            dropout=float(dropout),
            token_expansion_factor=0.5,
            channel_expansion_factor=4,
        )
        self.src_fc = nn.Linear(2 * self.hidden_dim, 100)
        self.dst_fc = nn.Linear(2 * self.hidden_dim, 100)
        self.out_fc = nn.Linear(100, 1)

    def _encode_history(
        self,
        history_times: torch.Tensor,
        history_mask: torch.Tensor | None,
        query_time: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, length = history_times.shape
        if length != self.history_len:
            raise ValueError(f"expected history length {self.history_len}, got {length}")
        valid = (
            torch.ones_like(history_times, dtype=torch.bool)
            if history_mask is None
            else history_mask.to(dtype=torch.bool)
        )
        edge_features = torch.zeros(
            (batch_size * length, 1),
            dtype=history_times.dtype,
            device=history_times.device,
        )
        elapsed = (
            query_time[:, None].to(history_times.dtype) - history_times
        ).clamp_min(0.0)
        indices = valid.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        return self.history_encoder(
            edge_features[indices],
            elapsed.reshape(-1)[indices],
            batch_size,
            indices,
        )

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        timestamp: torch.Tensor,
        history_nodes: torch.Tensor,
        history_times: torch.Tensor,
        history_edge_feats: torch.Tensor,
        history_mask: torch.Tensor | None = None,
        dst_history_nodes: torch.Tensor | None = None,
        dst_history_times: torch.Tensor | None = None,
        dst_history_edge_feats: torch.Tensor | None = None,
        dst_history_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> ModelResult:
        del history_nodes, history_edge_feats, dst_history_nodes, dst_history_edge_feats
        if dst_history_times is None:
            raise ValueError("GraphMixer requires destination history")
        src_history = self._encode_history(history_times, history_mask, timestamp)
        dst_history = self._encode_history(
            dst_history_times, dst_history_mask, timestamp
        )
        src_node = self.node_embedding(src.long().clamp(0, self.num_nodes - 1))
        dst_node = self.node_embedding(dst.long().clamp(0, self.num_nodes - 1))
        src_state = self.src_fc(torch.cat([src_history, src_node], dim=-1))
        dst_state = self.dst_fc(torch.cat([dst_history, dst_node], dim=-1))
        logits = self.out_fc(torch.relu(src_state + dst_state)).squeeze(-1)
        return ModelResult(logits=logits, aux={})


class OfficialTGNBaseline(nn.Module):
    """Official TGN memory and GRU update adapted to the CTDG batch API."""

    model_name = "TGN"
    official_commit = TGN_OFFICIAL_COMMIT

    def __init__(
        self,
        *,
        num_nodes: int,
        device: torch.device,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        tgn_class = _load_official_tgn_class()
        node_features = np.zeros((int(num_nodes), int(hidden_dim)), dtype=np.float32)
        edge_features = np.zeros((1, 1), dtype=np.float32)
        self.model = tgn_class(
            neighbor_finder=None,
            node_features=node_features,
            edge_features=edge_features,
            device=device,
            n_layers=1,
            n_heads=2,
            dropout=float(dropout),
            use_memory=True,
            memory_update_at_start=True,
            message_dimension=int(hidden_dim),
            memory_dimension=int(hidden_dim),
            embedding_module_type="identity",
            message_function="mlp",
            aggregator_type="last",
            memory_updater_type="gru",
            n_neighbors=1,
        )

    def reset_memory(self) -> None:
        self.model.memory.__init_memory__()

    def backup_memory(self) -> Any:
        return self.model.memory.backup_memory()

    def restore_memory(self, backup: Any) -> None:
        self.model.memory.restore_memory(backup)

    def detach_memory(self) -> None:
        self.model.memory.detach_memory()

    def _current_memory(self) -> torch.Tensor:
        nodes = list(range(self.model.n_nodes))
        memory, _ = self.model.get_updated_memory(nodes, self.model.memory.messages)
        return memory

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        timestamp: torch.Tensor,
        *_: Any,
        update_memory: bool = False,
        **__: Any,
    ) -> ModelResult:
        src_np = src.detach().cpu().numpy().astype(np.int64)
        dst_np = dst.detach().cpu().numpy().astype(np.int64)
        ts_np = timestamp.detach().cpu().numpy().astype(np.float64)
        if update_memory:
            src_state, dst_state, _ = self.model.compute_temporal_embeddings(
                src_np,
                dst_np,
                dst_np,
                ts_np,
                np.zeros(len(src_np), dtype=np.int64),
                n_neighbors=1,
            )
        else:
            memory = self._current_memory()
            src_state = memory[src.long()]
            dst_state = memory[dst.long()]
        logits = self.model.affinity_score(src_state, dst_state).squeeze(-1)
        return ModelResult(logits=logits, aux={})
