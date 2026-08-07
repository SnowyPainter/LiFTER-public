from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel


@dataclass
class BertAttentionOutput:
    logits: torch.Tensor
    aux: dict[str, torch.Tensor]


class CLSAttentionRobertaSelfAttention(nn.Module):
    """RoBERTa self-attention with Attention on final-layer CLS value terms.

    Only the CLS query row is gated.  Non-CLS query rows use the original
    attention output.  ``candidate_token_mask`` marks sentence-side tokens; the
    gate is applied only to those tokens so question/special tokens are not
    treated as candidate evidence.
    """

    def __init__(
        self,
        base_attention: nn.Module,
        *,
        variant: str,
        gate_init: float = 0.95,
        gate_leak: float = 0.05,
    ) -> None:
        super().__init__()
        self.query = base_attention.query
        self.key = base_attention.key
        self.value = base_attention.value
        self.dropout = base_attention.dropout
        self.config = base_attention.config
        self.num_attention_heads = int(base_attention.num_attention_heads)
        self.attention_head_size = int(base_attention.attention_head_size)
        self.all_head_size = int(base_attention.all_head_size)
        self.scaling = getattr(base_attention, "scaling", self.attention_head_size**-0.5)
        self.is_decoder = bool(getattr(base_attention, "is_decoder", False))
        self.is_causal = bool(getattr(base_attention, "is_causal", False))
        self.layer_idx = getattr(base_attention, "layer_idx", None)
        self.variant = str(variant)
        self.gate_leak = float(gate_leak)
        hidden = int(self.config.hidden_size)
        self.gate_query_fuse = nn.Linear(hidden * 3, hidden)
        self.gate_query = nn.Linear(hidden, hidden)
        self.gate_key = nn.Linear(hidden, hidden)
        self.gate_scorer = nn.Linear(hidden, 1)
        self._init_gate(float(gate_init))
        self.last_attention: torch.Tensor | None = None
        self.last_gate: torch.Tensor | None = None
        self.last_candidate_mask: torch.Tensor | None = None

    def _init_gate(self, gate_init: float) -> None:
        prob = (gate_init - self.gate_leak) / max(1.0e-8, 1.0 - self.gate_leak)
        prob = min(max(prob, 1.0e-4), 1.0 - 1.0e-4)
        nn.init.normal_(self.gate_scorer.weight, mean=0.0, std=1.0e-3)
        nn.init.constant_(self.gate_scorer.bias, math.log(prob / (1.0 - prob)))

    def transpose_for_scores(self, tensor: torch.Tensor) -> torch.Tensor:
        new_shape = tensor.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return tensor.view(new_shape).transpose(1, 2)

    def _gate(self, hidden_states: torch.Tensor, candidate_token_mask: torch.Tensor | None) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        cls_query = hidden_states[:, :1, :]
        if self.variant == "query_only_gate":
            hidden = F.gelu(self.gate_query(cls_query))
            logits = self.gate_scorer(hidden).expand(batch, seq_len)
        else:
            hidden = self.gate_query(cls_query) + self.gate_key(hidden_states)
            logits = self.gate_scorer(F.gelu(hidden)).squeeze(-1)
            if self.variant == "shuffled_attention" and batch > 1:
                logits = torch.roll(logits, shifts=1, dims=0)
        gate = self.gate_leak + (1.0 - self.gate_leak) * torch.sigmoid(logits)
        if candidate_token_mask is None:
            candidate = torch.ones(batch, seq_len, dtype=torch.bool, device=hidden_states.device)
            candidate[:, 0] = False
        else:
            candidate = candidate_token_mask.to(device=hidden_states.device, dtype=torch.bool)
        # Question/special tokens retain the original value path.
        gate = torch.where(candidate, gate, torch.ones_like(gate))
        return gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        candidate_token_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        attention_probs = torch.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        output = torch.matmul(attention_probs, value_layer)
        gate = self._gate(hidden_states, candidate_token_mask)
        attentioned_value = value_layer * gate[:, None, :, None]
        attentioned_output = torch.matmul(attention_probs, attentioned_value)
        # Only replace the CLS query row; all other tokens remain base attention.
        output[:, :, 0, :] = attentioned_output[:, :, 0, :]
        output = output.transpose(1, 2).reshape(*input_shape, self.all_head_size).contiguous()
        self.last_attention = attention_probs[:, :, 0, :].mean(dim=1)
        self.last_gate = gate
        self.last_candidate_mask = None if candidate_token_mask is None else candidate_token_mask
        return output, attention_probs


class CandidateAttentionRobertaSelfAttention(nn.Module):
    """RoBERTa self-attention with Attention on candidate-marker value terms.

    ``candidate_marker_mask`` is ``[B, C, L]`` and marks the query token for
    each candidate. ``candidate_token_mask`` is ``[B, C, L]`` and marks the
    value tokens belonging to that candidate.  Only marker query rows are
    replaced; all other token outputs remain the base self-attention output.
    """

    def __init__(
        self,
        base_attention: nn.Module,
        *,
        variant: str,
        gate_init: float = 0.95,
        gate_leak: float = 0.05,
    ) -> None:
        super().__init__()
        self.query = base_attention.query
        self.key = base_attention.key
        self.value = base_attention.value
        self.dropout = base_attention.dropout
        self.config = base_attention.config
        self.num_attention_heads = int(base_attention.num_attention_heads)
        self.attention_head_size = int(base_attention.attention_head_size)
        self.all_head_size = int(base_attention.all_head_size)
        self.scaling = getattr(base_attention, "scaling", self.attention_head_size**-0.5)
        self.is_decoder = bool(getattr(base_attention, "is_decoder", False))
        self.is_causal = bool(getattr(base_attention, "is_causal", False))
        self.layer_idx = getattr(base_attention, "layer_idx", None)
        self.variant = str(variant)
        self.gate_leak = float(gate_leak)
        hidden = int(self.config.hidden_size)
        self.gate_query_fuse = nn.Linear(hidden * 3, hidden)
        self.gate_query = nn.Linear(hidden, hidden)
        self.gate_key = nn.Linear(hidden, hidden)
        self.gate_scorer = nn.Linear(hidden, 1)
        self._init_gate(float(gate_init))
        self.last_attention: torch.Tensor | None = None
        self.last_gate: torch.Tensor | None = None
        self.last_candidate_mask: torch.Tensor | None = None
        self.last_base_marker_context: torch.Tensor | None = None
        self.last_attentioned_marker_context: torch.Tensor | None = None

    def _init_gate(self, gate_init: float) -> None:
        prob = (gate_init - self.gate_leak) / max(1.0e-8, 1.0 - self.gate_leak)
        prob = min(max(prob, 1.0e-4), 1.0 - 1.0e-4)
        nn.init.normal_(self.gate_scorer.weight, mean=0.0, std=1.0e-3)
        nn.init.constant_(self.gate_scorer.bias, math.log(prob / (1.0 - prob)))

    def transpose_for_scores(self, tensor: torch.Tensor) -> torch.Tensor:
        new_shape = tensor.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        return tensor.view(new_shape).transpose(1, 2)

    def _gate(
        self,
        hidden_states: torch.Tensor,
        candidate_marker_mask: torch.Tensor,
        candidate_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        marker = candidate_marker_mask.to(hidden_states.device, dtype=hidden_states.dtype)
        token_mask = candidate_token_mask.to(hidden_states.device, dtype=torch.bool)
        if self.variant == "open_path_no_gate":
            return torch.ones(
                token_mask.shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
        marker_hidden = torch.einsum("bcl,bld->bcd", marker, hidden_states)
        cls_hidden = hidden_states[:, :1, :].expand(-1, marker_hidden.shape[1], -1)
        query_repr = self.gate_query_fuse(
            torch.cat([cls_hidden, marker_hidden, cls_hidden * marker_hidden], dim=-1)
        )
        if self.variant == "query_only_gate":
            hidden = F.gelu(self.gate_query(query_repr)).unsqueeze(2)
            logits = self.gate_scorer(hidden).squeeze(-1).expand(-1, -1, hidden_states.shape[1])
        else:
            hidden = self.gate_query(query_repr).unsqueeze(2) + self.gate_key(hidden_states).unsqueeze(1)
            logits = self.gate_scorer(F.gelu(hidden)).squeeze(-1)
            if self.variant == "shuffled_attention" and hidden_states.shape[0] > 1:
                logits = torch.roll(logits, shifts=1, dims=0)
        gate = self.gate_leak + (1.0 - self.gate_leak) * torch.sigmoid(logits)
        return torch.where(token_mask, gate, torch.ones_like(gate))

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        candidate_marker_mask: torch.Tensor | None = None,
        candidate_token_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) * self.scaling
        if attention_mask is not None:
            scores = scores + attention_mask
        attention_probs = torch.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        output = torch.matmul(attention_probs, value_layer)
        if candidate_marker_mask is not None and candidate_token_mask is not None:
            marker = candidate_marker_mask.to(hidden_states.device, dtype=torch.bool)
            gate = self._gate(hidden_states, candidate_marker_mask, candidate_token_mask)
            marker_attention = torch.einsum("bcl,bhls->bchs", marker.to(attention_probs.dtype), attention_probs)
            attentioned_value = value_layer[:, None, :, :, :] * gate[:, :, None, :, None]
            attentioned_marker_output = torch.einsum("bchs,bchsd->bchd", marker_attention, attentioned_value)
            base_marker_output = torch.einsum("bchs,bhsd->bchd", marker_attention, value_layer)

            self.last_base_marker_context = base_marker_output.contiguous().view(
                base_marker_output.shape[0],
                base_marker_output.shape[1],
                self.all_head_size,
            )
            self.last_attentioned_marker_context = attentioned_marker_output.contiguous().view(
                attentioned_marker_output.shape[0],
                attentioned_marker_output.shape[1],
                self.all_head_size,
            )

            for cand_idx in range(marker.shape[1]):
                row_mask = marker[:, cand_idx, :][:, None, :, None]
                replacement = attentioned_marker_output[:, cand_idx, :, :][:, :, None, :]
                output = torch.where(row_mask, replacement, output)
            self.last_attention = marker_attention.mean(dim=2)
            self.last_gate = gate
            self.last_candidate_mask = candidate_token_mask
        else:
            self.last_attention = None
            self.last_gate = None
            self.last_candidate_mask = None
            self.last_base_marker_context = None
            self.last_attentioned_marker_context = None
        output = output.transpose(1, 2).reshape(*input_shape, self.all_head_size).contiguous()
        return output, attention_probs


class BertHotpotAttentionSelector(nn.Module):
    def __init__(
        self,
        *,
        model_name: str = "roberta-base",
        variant: str = "base",
        gate_init: float = 0.95,
        gate_leak: float = 0.1,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.variant = str(variant)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(self.encoder.config.hidden_size)
        self.post_mlp = (
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
            if self.variant == "post_mlp"
            else None
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.attention_attention: CLSAttentionRobertaSelfAttention | None = None
        if self.variant in {"full_attention", "shuffled_attention", "query_only_gate"}:
            last_self = self.encoder.encoder.layer[-1].attention.self
            self.attention_attention = CLSAttentionRobertaSelfAttention(
                last_self,
                variant=self.variant,
                gate_init=gate_init,
                gate_leak=gate_leak,
            )
            self.encoder.encoder.layer[-1].attention.self = self.attention_attention
        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                # Attention parameters and classifier stay trainable.
                if "gate_" not in name:
                    param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        candidate_token_mask: torch.Tensor | None = None,
    ) -> BertAttentionOutput:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            candidate_token_mask=candidate_token_mask,
        )
        cls = outputs.last_hidden_state[:, 0]
        if self.post_mlp is not None:
            cls = cls + self.post_mlp(cls)
        logits = self.classifier(cls).squeeze(-1)
        aux: dict[str, torch.Tensor] = {}
        if self.attention_attention is not None and self.attention_attention.last_attention is not None:
            attention = self.attention_attention.last_attention
            gate = self.attention_attention.last_gate
            candidate = (
                candidate_token_mask.to(attention.device, dtype=torch.bool)
                if candidate_token_mask is not None
                else torch.ones_like(attention, dtype=torch.bool)
            )
            aux["attention"] = attention
            aux["permission"] = gate
            aux["effective_mass"] = attention * gate
            aux["candidate_token_mask"] = candidate
        return BertAttentionOutput(logits=logits, aux=aux)


class BertHotpotMultiCandidateAttentionSelector(nn.Module):
    def __init__(
        self,
        *,
        model_name: str = "roberta-base",
        variant: str = "base",
        gate_init: float = 0.95,
        gate_leak: float = 0.1,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        vocab_size: int | None = None,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.variant = str(variant)
        self.encoder = AutoModel.from_pretrained(model_name)
        if vocab_size is not None:
            self.encoder.resize_token_embeddings(int(vocab_size))
        hidden = int(self.encoder.config.hidden_size)
        self.post_mlp = (
            nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
            if self.variant == "post_mlp"
            else None
        )
        self.post_attn_glu = (
            nn.ModuleDict(
                {
                    "value": nn.Linear(hidden, hidden),
                    "gate": nn.Linear(hidden, hidden),
                    "dropout": nn.Dropout(dropout),
                }
            )
            if self.variant == "post_attn_glu"
            else None
        )
        self.readout_fuse = (
            nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, hidden))
            if self.variant in {"full_attention", "open_path_no_gate"}
            else None
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.attention_attention: CandidateAttentionRobertaSelfAttention | None = None
        if self.variant in {
            "full_attention",
            "open_path_no_gate",
            "shuffled_attention",
            "query_only_gate",
            "attention_readout",
        }:
            last_self = self.encoder.encoder.layer[-1].attention.self
            self.attention_attention = CandidateAttentionRobertaSelfAttention(
                last_self,
                variant=self.variant,
                gate_init=gate_init,
                gate_leak=gate_leak,
            )
            self.encoder.encoder.layer[-1].attention.self = self.attention_attention
        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                if "gate_" not in name:
                    param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        candidate_marker_mask: torch.Tensor,
        candidate_token_mask: torch.Tensor,
        candidate_valid_mask: torch.Tensor,
    ) -> BertAttentionOutput:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            candidate_marker_mask=candidate_marker_mask,
            candidate_token_mask=candidate_token_mask,
        )
        marker = candidate_marker_mask.to(outputs.last_hidden_state.device, dtype=outputs.last_hidden_state.dtype)
        marker_state = torch.einsum("bcl,bld->bcd", marker, outputs.last_hidden_state)

        if (
            self.variant == "attention_readout"
            and self.attention_attention is not None
            and self.attention_attention.last_base_marker_context is not None
        ):
            marker_hidden = self.attention_attention.last_base_marker_context.to(outputs.last_hidden_state.device)
        elif (
            self.variant in {"full_attention", "open_path_no_gate"}
            and self.attention_attention is not None
            and self.attention_attention.last_attentioned_marker_context is not None
            and self.readout_fuse is not None
        ):
            attentioned_context = self.attention_attention.last_attentioned_marker_context.to(outputs.last_hidden_state.device)
            marker_hidden = marker_state + self.readout_fuse(
                torch.cat([marker_state, attentioned_context, marker_state * attentioned_context], dim=-1)
            )
        else:
            marker_hidden = marker_state

        if self.post_mlp is not None:
            marker_hidden = marker_hidden + self.post_mlp(marker_hidden)
        if self.post_attn_glu is not None:
            value = self.post_attn_glu["value"](marker_hidden)
            gate = torch.sigmoid(self.post_attn_glu["gate"](marker_hidden))
            marker_hidden = marker_hidden + self.post_attn_glu["dropout"](value * gate)
        logits = self.classifier(marker_hidden).squeeze(-1)
        logits = logits.masked_fill(~candidate_valid_mask.to(logits.device, dtype=torch.bool), -1.0e4)
        aux: dict[str, torch.Tensor] = {}
        if self.attention_attention is not None and self.attention_attention.last_attention is not None:
            attention = self.attention_attention.last_attention
            gate = self.attention_attention.last_gate
            token_mask = candidate_token_mask.to(attention.device, dtype=torch.bool)
            aux["attention"] = attention
            aux["permission"] = gate
            aux["effective_mass"] = attention * gate
            aux["candidate_token_mask"] = token_mask
        return BertAttentionOutput(logits=logits, aux=aux)
