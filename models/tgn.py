"""Benchmark-native Temporal Graph Network.

The original TGN maintains a mutable global node-memory table.  This version
replays each query's strictly historical, padded event sequence into the same
message-function -> last-message -> GRU-memory pipeline.  It is independent of
batch order and exposes differentiable event weights for post-hoc explainers.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .attention import SinusoidalTimeEncoding
from .base import AttentionModel, ModelResult


class NativeTGN(AttentionModel):
    model_name = "TGN"
    reproduction_ready = True
    implementation_scope = "TGN memory/message architecture adapted by causal history replay"

    def __init__(
        self,
        *,
        num_nodes: int,
        edge_feat_dim: int = 1,
        hidden_dim: int = 64,
        time_dim: int = 32,
        message_dim: int | None = None,
        neural_history_len: int = 20,
        dropout: float = 0.1,
        use_attention: bool = False,
        **_: Any,
    ) -> None:
        super().__init__(use_attention=use_attention)
        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.neural_history_len = int(neural_history_len)
        if self.neural_history_len < 1:
            raise ValueError("neural_history_len must be positive")
        self.node_embedding = nn.Embedding(self.num_nodes, self.hidden_dim)
        self.edge_encoder = nn.Linear(int(edge_feat_dim), self.hidden_dim)
        self.time_encoder = SinusoidalTimeEncoding(int(time_dim))
        raw_dim = 3 * self.hidden_dim + int(time_dim)
        msg_dim = self.hidden_dim if message_dim is None else int(message_dim)
        self.message_function = nn.Sequential(
            nn.Linear(raw_dim, raw_dim // 2), nn.ReLU(),
            nn.Dropout(float(dropout)), nn.Linear(raw_dim // 2, msg_dim),
        )
        self.memory_updater = nn.GRU(msg_dim, self.hidden_dim, batch_first=True)
        self.embedding = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.ReLU(),
        )
        self.affinity = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.ReLU(),
            nn.Dropout(float(dropout)), nn.Linear(self.hidden_dim, 1),
        )

    def _replay(
        self,
        center: torch.Tensor,
        timestamp: torch.Tensor,
        history_nodes: torch.Tensor,
        history_times: torch.Tensor,
        history_edge_feats: torch.Tensor,
        history_mask: torch.Tensor | None,
        history_weights: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # TGN's recurrent memory needs the recent neural context, not the full
        # 320-event bank retained for long-range heuristic/rule baselines.
        # Truncation happens before all encoders, reducing the sequential GRU
        # work while preserving the configured model input exactly.
        if history_nodes.shape[1] > self.neural_history_len:
            history_nodes = history_nodes[:, -self.neural_history_len:]
            history_times = history_times[:, -self.neural_history_len:]
            history_edge_feats = history_edge_feats[:, -self.neural_history_len:]
            if history_mask is not None:
                history_mask = history_mask[:, -self.neural_history_len:]
            if history_weights is not None:
                history_weights = history_weights[:, -self.neural_history_len:]
        batch, length = history_nodes.shape
        center_state = self.node_embedding(center.long().clamp(0, self.num_nodes - 1))
        memory = torch.zeros(batch, self.hidden_dim, device=center.device, dtype=center_state.dtype)
        valid = torch.ones_like(history_nodes, dtype=torch.bool) if history_mask is None else history_mask.bool()
        if bool((history_times[valid] > timestamp[:, None].expand_as(history_times)[valid]).any()):
            raise ValueError("TGN received a future event in its pre-query history")
        weights = valid.to(center_state.dtype) if history_weights is None else history_weights.to(center_state.dtype) * valid
        neighbor = self.node_embedding(history_nodes.long().clamp(0, self.num_nodes - 1))
        edge = self.edge_encoder(history_edge_feats.float())
        age = (timestamp[:, None] - history_times.float()).clamp_min(0)
        encoded_time = self.time_encoder(age)
        raw = torch.cat((
            center_state[:, None, :].expand(-1, length, -1),
            neighbor, edge, encoded_time,
        ), dim=-1)
        messages = self.message_function(raw)

        if history_weights is None:
            # Stable compaction preserves temporal order while moving padding
            # and deleted events behind the valid prefix.  A single fused GRU
            # kernel then executes the entire temporal scan on GPU using the
            # exact GRUCell parameters; no Python loop or per-event launch is
            # present on the forecasting path.
            positions = torch.arange(length, device=center.device)[None, :]
            order = torch.argsort((~valid).long() * length + positions, dim=1)
            compact = messages.gather(
                1, order[:, :, None].expand(-1, -1, messages.shape[-1])
            )
            compact_valid = valid.gather(1, order)
            compact = compact * compact_valid[:, :, None].to(compact.dtype)
            self.memory_updater.flatten_parameters()
            trace, _ = self.memory_updater(compact, memory[None])
            lengths = compact_valid.sum(1)
            selected = (lengths - 1).clamp_min(0)
            memory = trace[torch.arange(batch, device=center.device), selected]
            memory = torch.where(lengths[:, None] > 0, memory, torch.zeros_like(memory))
        else:
            # Continuous masks are used only while fitting post-hoc explainers.
            # They require state interpolation and therefore retain the exact
            # differentiable recurrence rather than the hard-mask fast path.
            states = []
            for index in range(length):
                input_gates = F.linear(
                    messages[:, index], self.memory_updater.weight_ih_l0,
                    self.memory_updater.bias_ih_l0,
                )
                hidden_gates = F.linear(
                    memory, self.memory_updater.weight_hh_l0,
                    self.memory_updater.bias_hh_l0,
                )
                input_reset, input_update, input_new = input_gates.chunk(3, 1)
                hidden_reset, hidden_update, hidden_new = hidden_gates.chunk(3, 1)
                reset = torch.sigmoid(input_reset + hidden_reset)
                update = torch.sigmoid(input_update + hidden_update)
                new = torch.tanh(input_new + reset * hidden_new)
                proposal = new + update * (memory - new)
                weight = weights[:, index:index + 1].clamp(0, 1)
                memory = weight * proposal + (1 - weight) * memory
                states.append(memory)
            trace = torch.stack(states, 1)
        return self.embedding(torch.cat((center_state, memory), -1)), trace

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
        history_weights: torch.Tensor | None = None,
        dst_history_weights: torch.Tensor | None = None,
        **_: Any,
    ) -> ModelResult:
        if dst_history_nodes is None or dst_history_times is None:
            raise ValueError("TGN requires both source and destination histories")
        if dst_history_edge_feats is None:
            dst_history_edge_feats = torch.zeros_like(history_edge_feats)
        src_state, src_trace = self._replay(
            src, timestamp, history_nodes, history_times, history_edge_feats,
            history_mask, history_weights,
        )
        dst_state, dst_trace = self._replay(
            dst, timestamp, dst_history_nodes, dst_history_times,
            dst_history_edge_feats, dst_history_mask, dst_history_weights,
        )
        logits = self.affinity(torch.cat((src_state, dst_state), -1)).squeeze(-1)
        return ModelResult(logits, {"src_memory_trace": src_trace, "dst_memory_trace": dst_trace})


__all__ = ["NativeTGN"]
