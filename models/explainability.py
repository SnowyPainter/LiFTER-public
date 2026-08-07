"""Native temporal-graph explanation models.

These implementations contain no imports from ``external``.  The downloaded
repositories are retained solely as provenance/reference material.  The
interfaces below use the benchmark's padded temporal history representation:
``history_nodes``, ``history_times``, ``history_edge_feats`` and a boolean mask.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .attention import SinusoidalTimeEncoding
from .base import AttentionModel, ModelResult


def _valid(mask: torch.Tensor | None, nodes: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(nodes, dtype=torch.bool) if mask is None else mask.bool()


class _EventEncoder(nn.Module):
    def __init__(self, num_nodes: int, edge_feat_dim: int, hidden_dim: int, time_dim: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.node = nn.Embedding(num_nodes, hidden_dim)
        self.edge = nn.Linear(edge_feat_dim, hidden_dim)
        self.time = SinusoidalTimeEncoding(time_dim)
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden_dim + time_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )

    def events(self, nodes: torch.Tensor, times: torch.Tensor, edge_feats: torch.Tensor,
               query_time: torch.Tensor) -> torch.Tensor:
        node = self.node(nodes.long().clamp(0, self.num_nodes - 1))
        edge = self.edge(edge_feats.float())
        age = (query_time[:, None] - times.float()).clamp_min(0)
        return self.fuse(torch.cat((node, edge, self.time(age)), dim=-1))


@dataclass
class TemporalExplanation:
    """A machine-readable event rationale returned by a native explainer."""

    event_scores: torch.Tensor
    selected_mask: torch.Tensor
    fidelity_drop: torch.Tensor | None = None
    motif_scores: torch.Tensor | None = None


@dataclass
class _SearchNode:
    coalition: tuple[int, ...]
    parent: "_SearchNode | None" = None
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    children: list["_SearchNode"] | None = None

    @property
    def value(self) -> float:
        return self.value_sum / max(1, self.visits)


class TGNNExplainer(nn.Module):
    """Explorer–navigator event-subset explainer (ICLR 2023).

    The explorer performs batched Monte-Carlo tree search over deletion masks.
    Its reward combines prediction fidelity and subset sparsity.  The navigator
    learns which event to remove from successful search trajectories, allowing
    later searches to be guided instead of exhaustively enumerated.
    """

    def __init__(self, predictor: nn.Module, *, rollout: int = 20, min_events: int = 1,
                 exploration: float = 2.0, sparsity_weight: float = 0.05) -> None:
        super().__init__()
        self.predictor = predictor
        self.rollout = int(rollout)
        self.min_events = int(min_events)
        self.exploration = float(exploration)
        self.sparsity_weight = float(sparsity_weight)
        self.navigator = nn.Sequential(nn.LazyLinear(32), nn.ReLU(), nn.Linear(32, 1))
        self.reproduction_ready = True
        self.implementation_scope = "explorer-navigator MCTS over pre-query histories"

    @staticmethod
    def _call(predictor: nn.Module, inputs: dict[str, torch.Tensor], mask: torch.Tensor,
              dst_mask: torch.Tensor | None = None) -> torch.Tensor:
        overrides = {"history_mask": mask}
        if dst_mask is not None:
            overrides["dst_history_mask"] = dst_mask
        result = predictor(**{**inputs, **overrides})
        return result.logits if isinstance(result, ModelResult) else result

    @staticmethod
    def _event_features(inputs: dict[str, torch.Tensor], row: int) -> torch.Tensor:
        query_time = inputs["timestamp"][row]
        blocks = []
        for side, prefix in ((0.0, ""), (1.0, "dst_")):
            nodes = inputs[f"{prefix}history_nodes"][row].float()
            times = inputs[f"{prefix}history_times"][row].float()
            edges = inputs[f"{prefix}history_edge_feats"][row].float()
            scale = nodes.abs().amax().clamp_min(1)
            age = torch.log1p((query_time - times).clamp_min(0))
            edge_norm = edges.norm(dim=-1)
            blocks.append(torch.stack((nodes / scale, age, edge_norm,
                                       torch.full_like(nodes, side)), -1))
        return torch.cat(blocks, 0)

    def _score_coalition(self, inputs: dict[str, torch.Tensor], row: int,
                         coalition: tuple[int, ...], original: torch.Tensor,
                         src_length: int, total_valid: int) -> float:
        one = {key: value[row:row + 1] for key, value in inputs.items()}
        src_mask = torch.zeros_like(one["history_nodes"], dtype=torch.bool)
        dst_mask = torch.zeros_like(one["dst_history_nodes"], dtype=torch.bool)
        for index in coalition:
            if index < src_length:
                src_mask[0, index] = True
            else:
                dst_mask[0, index - src_length] = True
        score = self._call(self.predictor, one, src_mask, dst_mask)
        fidelity = -float((original[row] - score[0]).abs().cpu())
        sparsity = 1.0 - len(coalition) / max(1, total_valid)
        return fidelity + self.sparsity_weight * sparsity

    @torch.no_grad()
    def explain(self, inputs: dict[str, torch.Tensor], *, top_k: int | None = None) -> TemporalExplanation:
        nodes = inputs["history_nodes"]
        valid = _valid(inputs.get("history_mask"), nodes)
        dst_nodes = inputs.get("dst_history_nodes")
        if dst_nodes is None:
            # Retain support for source-only black boxes used in diagnostics.
            original = self._call(self.predictor, inputs, valid)
            drops = torch.full_like(valid, -torch.inf, dtype=torch.float)
            for j in range(valid.shape[1]):
                candidate = valid.clone(); candidate[:, j] = False
                drops[:, j] = (original - self._call(self.predictor, inputs, candidate)).abs()
            target = max(1, min(int(top_k or self.min_events), valid.shape[1]))
            selected = torch.zeros_like(valid)
            selected.scatter_(1, drops.topk(target, 1).indices, True); selected &= valid
            return TemporalExplanation(drops, selected,
                                       (original-self._call(self.predictor, inputs, selected)).abs())
        dst_valid = _valid(inputs.get("dst_history_mask"), dst_nodes)
        original = self._call(self.predictor, inputs, valid, dst_valid)
        batch, length = valid.shape
        all_scores = torch.full((batch, length + dst_valid.shape[1]), -torch.inf, device=nodes.device)
        selected_all = torch.zeros_like(all_scores, dtype=torch.bool)
        for row in range(batch):
            valid_ids = torch.cat((valid[row], dst_valid[row])).nonzero().flatten().tolist()
            target = max(1, min(int(top_k or self.min_events), len(valid_ids)))
            features = self._event_features(inputs, row)
            priors = torch.softmax(self.navigator(features).squeeze(-1), 0).detach().cpu().tolist()
            root = _SearchNode(tuple(valid_ids))
            cache: dict[tuple[int, ...], float] = {}
            best: _SearchNode | None = None
            for _ in range(self.rollout):
                node, path = root, [root]
                while node.children:
                    log_parent = math.log(max(2, node.visits))
                    node = max(node.children, key=lambda child: child.value + self.exploration * child.prior * math.sqrt(log_parent / (1 + child.visits)))
                    path.append(node)
                if len(node.coalition) > target:
                    node.children = [
                        _SearchNode(tuple(x for x in node.coalition if x != removed), node, priors[removed])
                        for removed in node.coalition
                    ]
                    node = max(node.children, key=lambda child: child.prior)
                    path.append(node)
                if node.coalition not in cache:
                    cache[node.coalition] = self._score_coalition(
                        inputs, row, node.coalition, original, length, len(valid_ids))
                reward = cache[node.coalition]
                for ancestor in path:
                    ancestor.visits += 1; ancestor.value_sum += reward
                if len(node.coalition) <= target and (best is None or node.value > best.value):
                    best = node
            if best is None:
                best = _SearchNode(tuple(sorted(valid_ids, key=lambda idx: priors[idx], reverse=True)[:target]))
            selected_all[row, list(best.coalition)] = True
            # Search-backed inclusion value.  Unlike a singleton-deletion
            # surrogate, this compares rewards among the coalitions actually
            # visited by the explorer.
            stack = [root]
            visited_nodes: list[_SearchNode] = []
            while stack:
                node = stack.pop()
                visited_nodes.append(node)
                stack.extend(node.children or [])
            for event in valid_ids:
                with_event = [item.value for item in visited_nodes if event in item.coalition and item.visits]
                without_event = [item.value for item in visited_nodes if event not in item.coalition and item.visits]
                if with_event:
                    baseline = max(without_event) if without_event else min(with_event)
                    all_scores[row, event] = max(with_event) - baseline
        src_selected, dst_selected = selected_all[:, :length], selected_all[:, length:]
        masked_score = self._call(self.predictor, inputs, src_selected, dst_selected)
        return TemporalExplanation(all_scores, selected_all, (original - masked_score).abs())

    def navigator_loss(self, event_features: torch.Tensor, search_rewards: torch.Tensor,
                       mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.navigator(event_features).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        target = torch.softmax(search_rewards.detach(), dim=-1)
        return -(target * torch.log_softmax(logits, dim=-1)).sum(-1).mean()

    def navigator_training_loss(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Train the navigator to propose low-cost removals for the MCTS explorer.

        The predictor is treated as frozen.  For each valid event, the target is
        the change in the predictor logit after deleting that event.  Events with
        a small change are preferred removals, exactly matching the navigator's
        role in the explorer: it orders actions but never supplies the coalition
        reward used by MCTS.
        """
        valid = _valid(inputs.get("history_mask"), inputs["history_nodes"])
        dst_valid = _valid(inputs.get("dst_history_mask"), inputs["dst_history_nodes"])
        combined = torch.cat((valid, dst_valid), dim=1)
        with torch.no_grad():
            original = self._call(self.predictor, inputs, valid, dst_valid)
            deletion_cost = torch.zeros_like(combined, dtype=torch.float)
            width = valid.shape[1]
            for event in range(combined.shape[1]):
                src_mask, dst_mask = valid.clone(), dst_valid.clone()
                if event < width:
                    src_mask[:, event] = False
                else:
                    dst_mask[:, event - width] = False
                deletion_cost[:, event] = (original - self._call(
                    self.predictor, inputs, src_mask, dst_mask)).abs()
        features = torch.stack([
            self._event_features(inputs, row) for row in range(combined.shape[0])
        ])
        # A high target reward means that the event is safe to remove first.
        return self.navigator_loss(features, -deletion_cost, combined)


class TempME(nn.Module):
    """Temporal-motif explainer (NeurIPS 2023).

    Events are encoded into ordered two-edge temporal motifs.  A Concrete
    bottleneck selects motifs, and event importance is exactly the probability
    that an event participates in at least one selected motif.
    """

    def __init__(self, predictor: nn.Module | None = None, *, num_nodes: int, edge_feat_dim: int = 1, hidden_dim: int = 64,
                 time_dim: int = 32, temperature: float = 0.07, prior: float = 0.1,
                 **_: Any) -> None:
        super().__init__()
        self.predictor = predictor
        self.encoder = _EventEncoder(num_nodes, edge_feat_dim, hidden_dim, time_dim)
        self.temperature = float(temperature)
        self.prior = float(prior)
        self.motif_score = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.reproduction_ready = predictor is not None
        self.implementation_scope = "ordered temporal motifs with Concrete bottleneck and TGN mask coupling"

    def forward(self, *, timestamp: torch.Tensor, history_nodes: torch.Tensor,
                history_times: torch.Tensor, history_edge_feats: torch.Tensor,
                history_mask: torch.Tensor | None = None, hard: bool = False,
                **_: Any) -> TemporalExplanation:
        events = self.encoder.events(history_nodes, history_times, history_edge_feats, timestamp)
        batch, length, hidden = events.shape
        left = events[:, :, None, :].expand(-1, -1, length, -1)
        right = events[:, None, :, :].expand(-1, length, -1, -1)
        delta = (history_times[:, :, None] - history_times[:, None, :]).abs()
        temporal = torch.exp(-delta / (delta.detach().mean(dim=(1, 2), keepdim=True) + 1e-6))
        motif_logits = self.motif_score(torch.cat((left, right, left * right), -1)).squeeze(-1)
        valid = _valid(history_mask, history_nodes)
        ordered = history_times[:, :, None] < history_times[:, None, :]
        motif_valid = valid[:, :, None] & valid[:, None, :] & ordered
        motif_logits = motif_logits + torch.log(temporal.clamp_min(1e-8))
        motif_logits = motif_logits.masked_fill(~motif_valid, -30.0)
        if self.training and not hard:
            motif_prob = torch.sigmoid((motif_logits + torch.empty_like(motif_logits).uniform_(1e-6, 1-1e-6).logit()) / self.temperature)
        else:
            motif_prob = torch.sigmoid(motif_logits / self.temperature)
        motif_prob = motif_prob * motif_valid
        event_prob = 1 - (1 - motif_prob).prod(1) * (1 - motif_prob).prod(2)
        selected = (event_prob >= 0.5) & valid
        return TemporalExplanation(event_prob, selected, motif_scores=motif_prob)

    def bottleneck_loss(self, explanation: TemporalExplanation) -> torch.Tensor:
        q = explanation.motif_scores.clamp(1e-6, 1 - 1e-6)
        p = torch.as_tensor(self.prior, device=q.device, dtype=q.dtype)
        return (q * (q / p).log() + (1-q) * ((1-q)/(1-p)).log()).mean()

    def explain_and_predict(self, inputs: dict[str, torch.Tensor], *, hard: bool = False) -> tuple[ModelResult, TemporalExplanation]:
        if self.predictor is None:
            raise RuntimeError("TempME requires a frozen TGN predictor for prediction coupling")
        src = self(**inputs, hard=hard)
        dst_inputs = {
            **inputs,
            "history_nodes": inputs["dst_history_nodes"],
            "history_times": inputs["dst_history_times"],
            "history_edge_feats": inputs["dst_history_edge_feats"],
            "history_mask": inputs.get("dst_history_mask"),
        }
        dst = self(**dst_inputs, hard=hard)
        result = self.predictor(**inputs, history_weights=src.event_scores,
                                dst_history_weights=dst.event_scores)
        combined = TemporalExplanation(
            torch.cat((src.event_scores, dst.event_scores), 1),
            torch.cat((src.selected_mask, dst.selected_mask), 1),
            motif_scores=torch.cat((src.motif_scores.flatten(1), dst.motif_scores.flatten(1)), 1),
        )
        return result, combined

    def training_loss(self, inputs: dict[str, torch.Tensor], labels: torch.Tensor,
                      *, beta: float = 0.01) -> torch.Tensor:
        result, explanation = self.explain_and_predict(inputs)
        return F.binary_cross_entropy_with_logits(result.logits, labels.float()) + beta * self.bottleneck_loss(explanation)


class TGIB(AttentionModel):
    """Temporal Graph Information Bottleneck self-explainable predictor."""
    model_name = "TGIB"
    is_self_explainable = True

    def __init__(self, *, num_nodes: int, edge_feat_dim: int = 1, hidden_dim: int = 64,
                 time_dim: int = 32, temperature: float = 0.5, dropout: float = 0.1,
                 bottleneck_prior: float = 0.3, bottleneck_beta: float = 0.01,
                 use_attention: bool = False, **_: Any) -> None:
        super().__init__(use_attention=use_attention)
        self.encoder = _EventEncoder(num_nodes, edge_feat_dim, hidden_dim, time_dim)
        self.temperature = float(temperature)
        self.bottleneck_prior = float(bottleneck_prior)
        self.bottleneck_beta = float(bottleneck_beta)
        self.attention_q = nn.Linear(hidden_dim, hidden_dim)
        self.attention_k = nn.Linear(hidden_dim, hidden_dim)
        self.attention_v = nn.Linear(hidden_dim, hidden_dim)
        self.selector = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.score = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.reproduction_ready = True
        self.implementation_scope = "time-aware multivariate-Bernoulli graph information bottleneck"

    def forward(self, src: torch.Tensor, dst: torch.Tensor, timestamp: torch.Tensor,
                history_nodes: torch.Tensor, history_times: torch.Tensor,
                history_edge_feats: torch.Tensor, history_mask: torch.Tensor | None = None,
                dst_history_nodes: torch.Tensor | None = None,
                dst_history_times: torch.Tensor | None = None,
                dst_history_edge_feats: torch.Tensor | None = None,
                dst_history_mask: torch.Tensor | None = None,
                **_: Any) -> ModelResult:
        src_events = self.encoder.events(history_nodes, history_times, history_edge_feats, timestamp)
        src_valid = _valid(history_mask, history_nodes)
        if dst_history_nodes is not None and dst_history_times is not None:
            dst_edge = torch.zeros_like(history_edge_feats) if dst_history_edge_feats is None else dst_history_edge_feats
            dst_events = self.encoder.events(dst_history_nodes, dst_history_times, dst_edge, timestamp)
            events = torch.cat((src_events, dst_events), 1)
            valid = torch.cat((src_valid, _valid(dst_history_mask, dst_history_nodes)), 1)
        else:
            events, valid = src_events, src_valid
        src_h, dst_h = self.encoder.node(src.long()), self.encoder.node(dst.long())
        q, k, value = self.attention_q(events), self.attention_k(events), self.attention_v(events)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(events.shape[-1])
        attention_logits = attention_logits.masked_fill(~valid[:, None, :], -30)
        attention = torch.softmax(attention_logits, -1) * valid[:, None, :]
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-12)
        attended = attention @ value
        events = events + attended
        query = torch.cat((src_h, dst_h, src_h * dst_h), -1)
        query_events = torch.cat((events, (src_h + dst_h)[:, None].expand_as(events),
                                  events * (src_h + dst_h)[:, None]), -1)
        logits = self.selector(query_events).squeeze(-1)
        logits = logits.masked_fill(~valid, -30)
        if self.training:
            gate = F.gumbel_softmax(torch.stack((torch.zeros_like(logits), logits), -1), tau=self.temperature, hard=False)[..., 1]
        else:
            gate = torch.sigmoid(logits / self.temperature)
        gate = gate * valid
        context = (events * gate[..., None]).sum(1) / gate.sum(1, keepdim=True).clamp_min(1)
        prediction = self.score(torch.cat((src_h, dst_h, context), -1)).squeeze(-1)
        probability = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
        prior = torch.as_tensor(self.bottleneck_prior, device=gate.device, dtype=gate.dtype)
        ib = probability * (probability / prior).log() + (1-probability) * ((1-probability)/(1-prior)).log()
        ib = (ib * valid).sum() / valid.sum().clamp_min(1)
        return ModelResult(prediction, {"event_importance": probability * valid,
                           "sampled_event_gate": gate, "information_bottleneck": ib,
                           "regularization_loss": self.bottleneck_beta * ib})


class SIG(AttentionModel):
    """Self-Interpretable temporal Graph predictor with causal interventions."""
    model_name = "SIG"
    is_self_explainable = True

    def __init__(self, *, num_nodes: int, edge_feat_dim: int = 1, hidden_dim: int = 64,
                 time_dim: int = 32, dropout: float = 0.1, use_attention: bool = False,
                 causal_ratio: float = 0.4, confounder_count: int = 10,
                 **_: Any) -> None:
        super().__init__(use_attention=use_attention)
        self.encoder = _EventEncoder(num_nodes, edge_feat_dim, hidden_dim, time_dim)
        self.causal_ratio = float(causal_ratio)
        self.temporal_mixer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.temporal_query = nn.Linear(hidden_dim, hidden_dim)
        self.structural_query = nn.Linear(hidden_dim, hidden_dim)
        self.confounders = nn.Parameter(torch.randn(int(confounder_count), 2 * hidden_dim) / math.sqrt(hidden_dim))
        self.confounder_query = nn.Linear(2 * hidden_dim, 2 * hidden_dim)
        self.iid_classifier = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.temporal_classifier = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.structural_classifier = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.reproduction_ready = True
        self.implementation_scope = "separate temporal/structural causal subgraphs with ICCM-NWGM intervention"

    def _topk_st(self, scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        soft = torch.softmax(scores.masked_fill(~valid, -30), -1) * valid
        counts = valid.sum(-1)
        keep = torch.ceil(counts.float() * self.causal_ratio).long().clamp_min(1)
        # Per-row variable top-k without a Python loop or device synchronizing
        # ``.item()`` calls. Invalid rows remain zero through the final mask.
        order = torch.argsort(soft, dim=-1, descending=True)
        ranks = torch.empty_like(order)
        rank_values = torch.arange(scores.shape[-1], device=scores.device)[None].expand_as(order)
        ranks.scatter_(1, order, rank_values)
        hard = ((ranks < keep[:, None]) & valid).to(soft.dtype)
        return hard + soft - soft.detach() if self.training else hard

    @staticmethod
    def _pool(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (values * weights[..., None]).sum(1) / weights.sum(1, keepdim=True).clamp_min(1)

    def forward(self, src: torch.Tensor, dst: torch.Tensor, timestamp: torch.Tensor,
                history_nodes: torch.Tensor, history_times: torch.Tensor,
                history_edge_feats: torch.Tensor, history_mask: torch.Tensor | None = None,
                dst_history_nodes: torch.Tensor | None = None,
                dst_history_times: torch.Tensor | None = None,
                dst_history_edge_feats: torch.Tensor | None = None,
                dst_history_mask: torch.Tensor | None = None,
                **_: Any) -> ModelResult:
        if dst_history_nodes is None or dst_history_times is None:
            raise ValueError("SIG requires source and destination causal neighborhoods")
        dst_edge = torch.zeros_like(history_edge_feats) if dst_history_edge_feats is None else dst_history_edge_feats
        src_events = self.temporal_mixer(self.encoder.events(history_nodes, history_times, history_edge_feats, timestamp))
        dst_events = self.temporal_mixer(self.encoder.events(dst_history_nodes, dst_history_times, dst_edge, timestamp))
        src_valid, dst_valid = _valid(history_mask, history_nodes), _valid(dst_history_mask, dst_history_nodes)
        src_h, dst_h = self.encoder.node(src.long()), self.encoder.node(dst.long())
        # Eq. 10: cross-query temporal masks between endpoint histories.
        src_t_score = (self.temporal_query(dst_h)[:, None] * src_events).sum(-1) / math.sqrt(src_events.shape[-1])
        dst_t_score = (self.temporal_query(src_h)[:, None] * dst_events).sum(-1) / math.sqrt(dst_events.shape[-1])
        src_t_gate, dst_t_gate = self._topk_st(src_t_score, src_valid), self._topk_st(dst_t_score, dst_valid)
        temporal = torch.cat((self._pool(src_events, src_t_gate), self._pool(dst_events, dst_t_gate)), -1)
        # Eq. 12-13: structural masks operate on neighbor identity features.
        src_nodes_h = self.encoder.node(history_nodes.long().clamp(0, self.encoder.num_nodes-1))
        dst_nodes_h = self.encoder.node(dst_history_nodes.long().clamp(0, self.encoder.num_nodes-1))
        src_s_score = (self.structural_query(dst_h)[:, None] * src_nodes_h).sum(-1) / math.sqrt(src_h.shape[-1])
        dst_s_score = (self.structural_query(src_h)[:, None] * dst_nodes_h).sum(-1) / math.sqrt(src_h.shape[-1])
        src_s_gate, dst_s_gate = self._topk_st(src_s_score, src_valid), self._topk_st(dst_s_score, dst_valid)
        structural = torch.cat((src_h + self._pool(src_nodes_h, src_s_gate),
                                dst_h + self._pool(dst_nodes_h, dst_s_gate)), -1)
        # Eq. 14: query-conditioned confounder expectation and NWGM logits.
        joint = structural + temporal
        conf_attn = torch.softmax(self.confounder_query(joint) @ self.confounders.T / math.sqrt(joint.shape[-1]), -1)
        confounder = conf_attn @ self.confounders
        iid = self.iid_classifier(torch.cat((structural, temporal), -1)).squeeze(-1)
        temporal_do = self.temporal_classifier(torch.cat((temporal, confounder), -1)).squeeze(-1)
        structural_do = self.structural_classifier(torch.cat((structural, confounder), -1)).squeeze(-1)
        importance = torch.cat(((src_t_gate + src_s_gate) / 2, (dst_t_gate + dst_s_gate) / 2), 1)
        return ModelResult(iid, {"event_importance": importance,
            "temporal_event_importance": torch.cat((src_t_gate, dst_t_gate), 1),
            "structural_event_importance": torch.cat((src_s_gate, dst_s_gate), 1),
            "auxiliary_logits": torch.stack((temporal_do, structural_do), -1),
            "confounder_attention": conf_attn})


EXPLAINABILITY_REGISTRY = {
    "T-GNNExplainer": TGNNExplainer,
    "TempME": TempME,
    "TGIB": TGIB,
    "SIG": SIG,
}


def build_explainability_model(name: str, *args: Any, **kwargs: Any) -> nn.Module:
    try:
        cls = EXPLAINABILITY_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown explainability model {name!r}. Available: {', '.join(EXPLAINABILITY_REGISTRY)}") from exc
    return cls(*args, **kwargs)


def get_explainability_reference(name: str) -> type[nn.Module]:
    """Backward-compatible name; returns the native class, never external code."""
    try:
        return EXPLAINABILITY_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown explainability model {name!r}") from exc


__all__ = ["EXPLAINABILITY_REGISTRY", "SIG", "TGIB", "TGNNExplainer", "TempME",
           "TemporalExplanation", "build_explainability_model", "get_explainability_reference"]
