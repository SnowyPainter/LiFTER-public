from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .base import ModelResult, AttentionModel


def _activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    activations: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        "gelu": F.gelu,
        "relu": F.relu,
        "swish": F.silu,
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
    }
    try:
        return activations[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported hidden activation: {name}") from error


class CRAFTCrossAttentionLayer(nn.Module):
    """Cross-attention layer matching the authors' public implementation.

    Candidate destinations form the queries. The source node's recent
    neighbors form both keys and values. The LayerNorm placement intentionally
    follows ``luyi256/CRAFT`` rather than ``torch.nn.TransformerDecoderLayer``.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        *,
        attention_dropout: float,
        hidden_dropout: float,
        hidden_act: str,
        layer_norm_eps: float,
        ffn_multiplier: int,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.attention_output = nn.Linear(hidden_dim, hidden_dim)
        self.attention_output_dropout = nn.Dropout(hidden_dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.ffn_in = nn.Linear(hidden_dim, ffn_multiplier * hidden_dim)
        self.ffn_out = nn.Linear(ffn_multiplier * hidden_dim, hidden_dim)
        self.ffn_dropout = nn.Dropout(hidden_dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.hidden_activation = _activation(hidden_act)

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim)

    def forward(
        self,
        destination: torch.Tensor,
        source_history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self._heads(self.query(destination)).permute(0, 2, 1, 3)
        key = self._heads(self.key(source_history)).permute(0, 2, 3, 1)
        value = self._heads(self.value(source_history)).permute(0, 2, 1, 3)
        scores = torch.matmul(query, key) / self.scale
        # The reference uses -10000 instead of -inf, so an entirely empty
        # history remains finite. Its training loop normally skips such cases.
        scores = scores.masked_fill(~history_mask[:, None, None, :], -10000.0)
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, value).permute(0, 2, 1, 3).contiguous()
        context = context.view(destination.shape[0], destination.shape[1], -1)
        update = self.attention_output_dropout(self.attention_output(context))
        destination = destination + self.attention_norm(update)

        update = self.ffn_out(self.hidden_activation(self.ffn_in(destination)))
        update = self.ffn_dropout(update)
        destination = destination + self.ffn_norm(update)
        return destination, weights


class _ReferenceMLP(nn.Module):
    """The small MLP used by the official CRAFT implementation."""

    def __init__(self, input_dim: int, output_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_output_layers must be positive")
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend((nn.Linear(current_dim, input_dim), nn.PReLU(init=0.15), nn.Dropout(dropout)))
            current_dim = input_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class CRAFT(AttentionModel):
    """CRAFT adapted to this repository's CTDG model interface.

    The network follows the public implementation at
    https://github.com/luyi256/CRAFT accompanying
    "Future Link Prediction Without Memory or Aggregation" (NeurIPS 2025):
    learned node identifiers, positional source-history embeddings,
    destination-to-history cross-attention, destination elapsed time, and a
    prediction MLP. No source memory state or post-hoc neighbor aggregation is
    constructed.

    This repository stores histories left-padded and evaluates one candidate
    per call. The adapter compacts valid neighbors to the left (as in the
    authors' sampler) before applying the reference architecture.
    """

    model_name = "CRAFT"

    def __init__(
        self,
        *,
        num_nodes: int,
        hidden_dim: int = 172,
        num_heads: int = 2,
        num_layers: int = 2,
        max_neighbors: int = 20,
        hidden_dropout: float = 0.5,
        attention_dropout: float = 0.5,
        embedding_dropout: float = 0.1,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        ffn_multiplier: int = 4,
        num_output_layers: int = 1,
        use_position: bool = True,
        use_elapsed_time: bool = True,
        use_repeat_count: bool = False,
        repeat_operator: str = "concat",
        projection_control: str = "membership",
        unseen_elapsed_value: float = -100000.0,
        use_attention: bool = False,
        dropout: float | None = None,
        # Backward-compatible aliases for early local configurations.
        use_repeat_time: bool | None = None,
        **_: object,
    ) -> None:
        super().__init__(use_attention=use_attention)
        if num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if dropout is not None:
            hidden_dropout = float(dropout)
        if use_repeat_time is not None:
            use_repeat_count = bool(use_repeat_time)

        self.num_nodes = int(num_nodes)
        self.hidden_dim = int(hidden_dim)
        self.max_neighbors = int(max_neighbors)
        self.use_position = bool(use_position)
        self.use_elapsed_time = bool(use_elapsed_time)
        self.use_repeat_count = bool(use_repeat_count)
        self.repeat_operator = str(repeat_operator)
        self.projection_control = str(projection_control)
        if self.repeat_operator not in {"concat", "membership_orthogonal"}:
            raise ValueError(
                "repeat_operator must be one of: concat, membership_orthogonal"
            )
        if self.projection_control not in {"membership", "random", "shuffled"}:
            raise ValueError(
                "projection_control must be one of: membership, random, shuffled"
            )
        self.unseen_elapsed_value = float(unseen_elapsed_value)
        self.padding_id = self.num_nodes

        # Native node ID 0 is valid in this repository, hence a dedicated last
        # row is reserved for padding instead of reusing the authors' ID 0.
        self.node_embedding = nn.Embedding(self.num_nodes + 1, self.hidden_dim, padding_idx=self.padding_id)
        self.position_embedding = nn.Embedding(self.max_neighbors, self.hidden_dim)
        self.embedding_norm = nn.LayerNorm(self.hidden_dim, eps=layer_norm_eps)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.layers = nn.ModuleList(
            [
                CRAFTCrossAttentionLayer(
                    self.hidden_dim,
                    num_heads,
                    attention_dropout=attention_dropout,
                    hidden_dropout=hidden_dropout,
                    hidden_act=hidden_act,
                    layer_norm_eps=layer_norm_eps,
                    ffn_multiplier=ffn_multiplier,
                )
                for _ in range(num_layers)
            ]
        )
        if self.use_elapsed_time:
            self.elapsed_projection: nn.Linear | None = nn.Linear(1, self.hidden_dim)
            self.elapsed_norm: nn.LayerNorm | None = nn.LayerNorm(self.hidden_dim, eps=layer_norm_eps)
        else:
            self.elapsed_projection = None
            self.elapsed_norm = None
        if self.use_repeat_count:
            self.repeat_projection: nn.Linear | None = nn.Linear(1, self.hidden_dim)
            self.repeat_norm: nn.LayerNorm | None = nn.LayerNorm(self.hidden_dim, eps=layer_norm_eps)
        else:
            self.repeat_projection = None
            self.repeat_norm = None
        self.feature_dropout = nn.Dropout(hidden_dropout)
        representation_width = self.hidden_dim * (
            1 + int(self.use_elapsed_time) + int(self.use_repeat_count)
        )
        if self.repeat_operator == "membership_orthogonal":
            self.register_buffer(
                "membership_direction", torch.zeros(representation_width)
            )
            self.projection_active = False
        else:
            self.register_buffer("membership_direction", None)
            self.projection_active = False
        predictor_width = representation_width
        self.predictor = _ReferenceMLP(
            predictor_width,
            1,
            num_layers=num_output_layers,
            dropout=hidden_dropout,
        )
        self.reset_parameters(initializer_range)

    def reset_parameters(self, initializer_range: float = 0.02) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.normal_(module.weight, mean=0.0, std=initializer_range)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.node_embedding.weight[self.padding_id].zero_()

    def _node_ids(self, values: torch.Tensor) -> torch.Tensor:
        return values.long().clamp(min=0, max=self.num_nodes - 1)

    def _compact_history(
        self,
        history_nodes: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if history_nodes.shape[1] > self.max_neighbors:
            history_nodes = history_nodes[:, -self.max_neighbors :]
            if history_mask is not None:
                history_mask = history_mask[:, -self.max_neighbors :]
        valid = (
            torch.ones_like(history_nodes, dtype=torch.bool)
            if history_mask is None
            else history_mask.to(torch.bool)
        )
        # Local histories are left-padded; the reference sampler puts valid
        # neighbors at the front. Stable sorting reproduces that convention.
        length = history_nodes.shape[1]
        positions = torch.arange(length, device=history_nodes.device).expand_as(history_nodes)
        sort_key = torch.where(valid, positions, positions + length)
        order = torch.argsort(sort_key, dim=1, stable=True)
        compact_nodes = history_nodes.gather(1, order)
        compact_mask = valid.gather(1, order)
        compact_nodes = torch.where(
            compact_mask,
            self._node_ids(compact_nodes),
            torch.full_like(compact_nodes, self.padding_id),
        )
        return compact_nodes, compact_mask

    def _source_history(
        self,
        history_nodes: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nodes, valid = self._compact_history(history_nodes, history_mask)
        history = self.node_embedding(nodes)
        if self.use_position:
            positions = torch.arange(nodes.shape[1], device=nodes.device)
            history = history + self.position_embedding(positions).unsqueeze(0)
        history = self.embedding_dropout(self.embedding_norm(history))
        return history, valid

    def _destination_elapsed(
        self,
        timestamp: torch.Tensor,
        dst_history_times: torch.Tensor | None,
        dst_history_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if dst_history_times is None:
            seen = torch.zeros_like(timestamp, dtype=torch.bool)
            elapsed = torch.full_like(timestamp.float(), self.unseen_elapsed_value)
            return elapsed, seen
        valid = (
            torch.ones_like(dst_history_times, dtype=torch.bool)
            if dst_history_mask is None
            else dst_history_mask.to(torch.bool)
        )
        last_activity = torch.where(
            valid,
            dst_history_times.float(),
            torch.full_like(dst_history_times.float(), -torch.inf),
        ).amax(dim=1)
        seen = torch.isfinite(last_activity)
        elapsed = timestamp.float() - torch.where(seen, last_activity, timestamp.float())
        elapsed = torch.where(
            seen,
            elapsed,
            torch.full_like(elapsed, self.unseen_elapsed_value),
        )
        return elapsed, seen

    @staticmethod
    def _repeat_count(
        dst: torch.Tensor,
        history_nodes: torch.Tensor,
        history_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        valid = (
            torch.ones_like(history_nodes, dtype=torch.bool)
            if history_mask is None
            else history_mask.to(torch.bool)
        )
        return ((history_nodes.long() == dst.long().unsqueeze(1)) & valid).sum(dim=1).float()

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
        del src, history_times, history_edge_feats, dst_history_nodes, dst_history_edge_feats
        source_history, valid = self._source_history(history_nodes, history_mask)
        destination = self.node_embedding(self._node_ids(dst)).unsqueeze(1)
        destination = self.embedding_dropout(self.embedding_norm(destination))

        layer_weights = []
        for layer in self.layers:
            destination, weights = layer(destination, source_history, valid)
            layer_weights.append(weights)
        edge_representation = destination.squeeze(1)
        features = [edge_representation]
        aux: dict[str, torch.Tensor] = {
            "attention_weights": torch.stack(layer_weights, dim=1),
            "edge_representation": edge_representation.detach(),
        }

        if self.elapsed_projection is not None and self.elapsed_norm is not None:
            elapsed, destination_seen = self._destination_elapsed(
                timestamp, dst_history_times, dst_history_mask
            )
            elapsed_feature = self.elapsed_projection(elapsed.unsqueeze(-1))
            features.append(self.feature_dropout(self.elapsed_norm(elapsed_feature)))
            aux["destination_elapsed"] = elapsed.detach()
            aux["destination_seen"] = destination_seen.detach()

        repeat_count = None
        if self.use_repeat_count:
            repeat_count = self._repeat_count(dst, history_nodes, history_mask)
            aux["repeat_count"] = repeat_count.detach()

        if self.repeat_projection is not None and self.repeat_norm is not None:
            assert repeat_count is not None
            repeat_feature = self.repeat_projection(repeat_count.unsqueeze(-1))
            features.append(self.feature_dropout(self.repeat_norm(repeat_feature)))

        fused = torch.cat(features, dim=-1)
        if self.membership_direction is not None:
            # The seen-minus-unseen mean difference defines the Euclidean
            # direction removed from the link representation.
            direction = F.normalize(
                self.membership_direction.detach(), dim=-1, eps=1e-12
            ).unsqueeze(0)
            unprojected_fused = fused
            membership_coordinate = F.linear(unprojected_fused, direction)
            if self.projection_active:
                fused = unprojected_fused - membership_coordinate * direction
            aux["unprojected_fused"] = unprojected_fused.detach()
            aux["membership_coordinate"] = membership_coordinate.detach().squeeze(-1)
        logits = self.predictor(fused).squeeze(-1)
        return ModelResult(logits=logits, aux=aux)

    @staticmethod
    def bpr_loss(positive_logits: torch.Tensor, negative_logits: torch.Tensor) -> torch.Tensor:
        """Bayesian Personalized Ranking loss used by the paper."""

        if positive_logits.shape != negative_logits.shape:
            raise ValueError("positive_logits and negative_logits must have identical shapes")
        return -F.logsigmoid(positive_logits - negative_logits).mean()


class CRAFTR(CRAFT):
    """CRAFT-R, which adds candidate-edge repeat-count encoding."""

    model_name = "CRAFT-R"

    def __init__(self, **kwargs: object) -> None:
        kwargs.setdefault("use_repeat_count", True)
        super().__init__(**kwargs)
