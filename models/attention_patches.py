from __future__ import annotations

import math
import types
from typing import Any

import torch
from torch import nn
import numpy as np

from .attention_block import AttentionBlock


def masked_attention_for_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    dropout: nn.Module | None = None,
    attention: AttentionBlock | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention with optional value-level attention gating.

    Inputs use the common multi-head shape ``[batch, heads, length, dim]``.
    ``mask`` follows attention convention: truthy entries are blocked.
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(float(d_k))
    if mask is not None:
        scores = scores.masked_fill(mask.to(torch.bool), torch.finfo(scores.dtype).min)

    empty_rows = torch.isneginf(scores).all(dim=-1, keepdim=True)
    scores = torch.where(empty_rows, torch.zeros_like(scores), scores)
    attn = torch.softmax(scores, dim=-1)
    if mask is not None:
        attn = attn.masked_fill(mask.to(torch.bool), 0.0)
    if dropout is not None and attention is None:
        attn = dropout(attn)

    if attention is None:
        return torch.matmul(attn, value), attn

    batch_size, num_heads, query_len, head_dim = query.shape
    key_len = key.shape[-2]
    flat_query = query.reshape(batch_size * num_heads, query_len, head_dim)
    flat_key = key.reshape(batch_size * num_heads, key_len, head_dim)
    flat_value = value.reshape(batch_size * num_heads, key_len, value.shape[-1])
    flat_attn = attn.reshape(batch_size * num_heads, query_len, key_len)

    keep_mask = None
    if mask is not None:
        keep_mask = (~mask.to(torch.bool)).expand(batch_size, num_heads, query_len, key_len)
        keep_mask = keep_mask.reshape(batch_size * num_heads, query_len, key_len)

    output, _, _, _ = attention(
        flat_query,
        flat_key,
        flat_value,
        attention_weights=flat_attn,
        mask=keep_mask,
    )
    return output.reshape(batch_size, num_heads, query_len, value.shape[-1]), attn


class AttentionedEasyTPPMultiHeadAttention(nn.Module):
    """Drop-in replacement for EasyTPP's official MultiHeadAttention."""

    def __init__(
        self,
        n_head: int,
        d_input: int,
        d_model: int,
        dropout: float = 0.1,
        output_linear: bool = False,
        *,
        gate_init: float = 0.95,
        gate_leak: float = 0.05,
    ) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k
        self.d_model = d_model
        self.output_linear = output_linear
        if output_linear:
            self.linears = nn.ModuleList([nn.Linear(d_input, d_model) for _ in range(3)] + [nn.Linear(d_model, d_model)])
        else:
            self.linears = nn.ModuleList([nn.Linear(d_input, d_model) for _ in range(3)])
        self.dropout = nn.Dropout(p=dropout)
        self.attention = AttentionBlock(
            self.d_k,
            self.d_k,
            self.d_v,
            gate_init=gate_init,
            gate_leak=gate_leak,
            dropout=dropout,
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, output_weight: bool = False):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            lin_layer(x).view(nbatches, -1, self.n_head, self.d_k).transpose(1, 2)
            for lin_layer, x in zip(self.linears, (query, key, value))
        ]
        x, attn_weight = masked_attention_for_attention(
            query,
            key,
            value,
            mask=mask,
            dropout=self.dropout,
            attention=self.attention,
        )
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.n_head * self.d_k)
        if self.output_linear:
            x = self.linears[-1](x)
        return (x, attn_weight) if output_weight else x


class AttentionedDyGLibMultiHeadAttention(nn.Module):
    """Drop-in replacement for DyGLib/TGAT temporal MultiHeadAttention."""

    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        time_feat_dim: int,
        num_heads: int = 2,
        dropout: float = 0.1,
        *,
        gate_init: float = 0.95,
        gate_leak: float = 0.05,
    ) -> None:
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.time_feat_dim = time_feat_dim
        self.num_heads = num_heads
        self.query_dim = node_feat_dim + time_feat_dim
        self.key_dim = node_feat_dim + edge_feat_dim + time_feat_dim
        if self.query_dim % num_heads != 0:
            raise ValueError("node_feat_dim + time_feat_dim must be divisible by num_heads")
        self.head_dim = self.query_dim // num_heads
        self.query_projection = nn.Linear(self.query_dim, num_heads * self.head_dim, bias=False)
        self.key_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)
        self.value_projection = nn.Linear(self.key_dim, num_heads * self.head_dim, bias=False)
        self.scaling_factor = self.head_dim ** -0.5
        self.layer_norm = nn.LayerNorm(self.query_dim)
        self.residual_fc = nn.Linear(num_heads * self.head_dim, self.query_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = AttentionBlock(
            self.head_dim,
            self.head_dim,
            self.head_dim,
            gate_init=gate_init,
            gate_leak=gate_leak,
            dropout=dropout,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        node_time_features: torch.Tensor,
        neighbor_node_features: torch.Tensor,
        neighbor_node_time_features: torch.Tensor,
        neighbor_node_edge_features: torch.Tensor,
        neighbor_masks: np.ndarray,
    ):
        node_features = torch.unsqueeze(node_features, dim=1)
        query = residual = torch.cat([node_features, node_time_features], dim=2)
        query = self.query_projection(query).reshape(query.shape[0], query.shape[1], self.num_heads, self.head_dim)
        key = value = torch.cat([neighbor_node_features, neighbor_node_edge_features, neighbor_node_time_features], dim=2)
        key = self.key_projection(key).reshape(key.shape[0], key.shape[1], self.num_heads, self.head_dim)
        value = self.value_projection(value).reshape(value.shape[0], value.shape[1], self.num_heads, self.head_dim)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        attention_mask = torch.from_numpy(neighbor_masks).to(node_features.device).unsqueeze(dim=1) == 0
        attention_mask = torch.stack([attention_mask for _ in range(self.num_heads)], dim=1)
        attention_output, attention_scores = masked_attention_for_attention(
            query,
            key,
            value,
            mask=attention_mask,
            dropout=self.dropout,
            attention=self.attention,
        )
        attention_output = attention_output.permute(0, 2, 1, 3).flatten(start_dim=2)
        output = self.dropout(self.residual_fc(attention_output))
        output = self.layer_norm(output + residual).squeeze(dim=1)
        attention_scores = attention_scores.squeeze(dim=2)
        return output, attention_scores


class AttentionedNeuralSTPPMultiheadAttention(nn.Module):
    """Drop-in replacement for facebookresearch/neural_stpp MultiheadAttention."""

    def __init__(self, embed_dim: int, num_heads: int, *, gate_init: float = 0.95, gate_leak: float = 0.05) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attention = AttentionBlock(self.head_dim, self.head_dim, self.head_dim, gate_init=gate_init, gate_leak=gate_leak)

    def forward(self, x: torch.Tensor, attn_mask=None, rm_nonself_grads: bool = False, attn_multiplier=None):
        if rm_nonself_grads:
            raise NotImplementedError("Attentioned neural_stpp attention does not support rm_nonself_grads=True")
        time_len, batch_size, _ = x.shape
        q, k, v = map(
            lambda a: a.reshape(time_len, batch_size, self.num_heads, self.head_dim),
            torch.split(self.in_proj(x), self.embed_dim, dim=-1),
        )
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))

        block_mask = None
        if attn_mask is not None:
            additive = attn_mask.to(device=logits.device, dtype=logits.dtype).view(1, 1, time_len, time_len)
            logits = logits + additive
            block_mask = additive < -1.0e6

        attn_weights = torch.softmax(logits, dim=-1)
        if attn_multiplier is not None:
            multiplier = attn_multiplier.to(device=attn_weights.device, dtype=attn_weights.dtype)
            multiplier = multiplier.permute(1, 2, 0).unsqueeze(-2)
            attn_weights = attn_weights * multiplier
            attn_weights = attn_weights / attn_weights.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(attn_weights.dtype).eps)

        flat_q = q.reshape(batch_size * self.num_heads, time_len, self.head_dim)
        flat_k = k.reshape(batch_size * self.num_heads, time_len, self.head_dim)
        flat_v = v.reshape(batch_size * self.num_heads, time_len, self.head_dim)
        flat_attn = attn_weights.reshape(batch_size * self.num_heads, time_len, time_len)
        keep_mask = None
        if block_mask is not None:
            keep_mask = (~block_mask).expand(batch_size, self.num_heads, time_len, time_len)
            keep_mask = keep_mask.reshape(batch_size * self.num_heads, time_len, time_len)
        output, _, _, _ = self.attention(flat_q, flat_k, flat_v, attention_weights=flat_attn, mask=keep_mask)
        output = output.reshape(batch_size, self.num_heads, time_len, self.head_dim)
        output = output.permute(2, 0, 1, 3).reshape(time_len, batch_size, -1)
        return self.out_proj(output), attn_weights.permute(2, 3, 0, 1).detach()


def replace_torch_multihead_attention(module: nn.Module, *, gate_init: float = 0.95, gate_leak: float = 0.05) -> int:
    """Replace child ``nn.MultiheadAttention`` modules with a attentioned wrapper.

    The wrapper preserves the public forward contract used by PyTorch encoder
    layers, while applying AttentionBlock after the usual relevance softmax.
    """

    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, AttentionedTorchMultiheadAttention.from_attention(child, gate_init=gate_init, gate_leak=gate_leak))
            replaced += 1
        else:
            replaced += replace_torch_multihead_attention(child, gate_init=gate_init, gate_leak=gate_leak)
    return replaced


def replace_huggingface_attention(module: nn.Module, *, gate_init: float = 0.95, gate_leak: float = 0.05) -> int:
    replaced = 0
    for child in module.modules():
        class_name = child.__class__.__name__
        if class_name == "T5Attention" and all(hasattr(child, attr) for attr in ("q", "k", "v", "o")):
            child._attention_original_forward = child.forward
            child._attention_block = AttentionBlock(
                child.key_value_proj_dim,
                child.key_value_proj_dim,
                child.key_value_proj_dim,
                gate_init=gate_init,
                gate_leak=gate_leak,
                dropout=getattr(child, "dropout", 0.0),
            )
            child.forward = types.MethodType(_t5_attention_forward, child)
            replaced += 1
        elif class_name == "LEDDecoderAttention" and all(hasattr(child, attr) for attr in ("q_proj", "k_proj", "v_proj", "out_proj")):
            child._attention_original_forward = child.forward
            child._attention_block = AttentionBlock(
                child.head_dim,
                child.head_dim,
                child.head_dim,
                gate_init=gate_init,
                gate_leak=gate_leak,
                dropout=getattr(child, "dropout", 0.0),
            )
            child.forward = types.MethodType(_led_decoder_attention_forward, child)
            replaced += 1
    return replaced


def _t5_attention_forward(
    self,
    hidden_states,
    mask=None,
    key_value_states=None,
    position_bias=None,
    past_key_values=None,
    output_attentions=False,
    **kwargs,
):
    if past_key_values is not None:
        return self._attention_original_forward(
            hidden_states,
            mask=mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            **kwargs,
        )
    input_shape = hidden_states.shape[:-1]
    batch_size, query_len = input_shape
    query_states = self.q(hidden_states).view(batch_size, query_len, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    current_states = key_value_states if key_value_states is not None else hidden_states
    key_len = current_states.shape[1]
    key_states = self.k(current_states).view(batch_size, key_len, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    value_states = self.v(current_states).view(batch_size, key_len, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    scores = torch.matmul(query_states, key_states.transpose(3, 2))
    if position_bias is None:
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros((1, query_states.shape[1], query_len, key_len), device=scores.device, dtype=scores.dtype)
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(query_len, key_len, device=scores.device, past_seen_tokens=0)
        if mask is not None:
            position_bias = position_bias + mask[:, :, :, :key_len]
    scores = scores + position_bias
    attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    flat_q = query_states.reshape(batch_size * self.n_heads, query_len, self.key_value_proj_dim)
    flat_k = key_states.reshape(batch_size * self.n_heads, key_len, self.key_value_proj_dim)
    flat_v = value_states.reshape(batch_size * self.n_heads, key_len, self.key_value_proj_dim)
    flat_attn = attn_weights.reshape(batch_size * self.n_heads, query_len, key_len)
    keep_mask = None
    if mask is not None:
        keep_mask = mask[:, :, :, :key_len] > -1.0e6
        keep_mask = keep_mask.expand(batch_size, self.n_heads, query_len, key_len).reshape(batch_size * self.n_heads, query_len, key_len)
    attn_output, _, _, _ = self._attention_block(flat_q, flat_k, flat_v, attention_weights=flat_attn, mask=keep_mask)
    attn_output = attn_output.reshape(batch_size, self.n_heads, query_len, self.key_value_proj_dim)
    attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)
    attn_output = self.o(attn_output)
    outputs = (attn_output, position_bias)
    if output_attentions:
        outputs = outputs + (attn_weights,)
    return outputs


def _led_decoder_attention_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states: torch.Tensor | None = None,
    past_key_values=None,
    attention_mask: torch.Tensor | None = None,
    output_attentions: bool = False,
):
    if past_key_values is not None:
        return self._attention_original_forward(
            hidden_states,
            key_value_states=key_value_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
    is_cross_attention = key_value_states is not None
    batch_size, query_len, embed_dim = hidden_states.size()
    current_states = key_value_states if is_cross_attention else hidden_states
    key_len = current_states.shape[1]
    query_states = self.q_proj(hidden_states) * self.scaling
    key_states = self.k_proj(current_states)
    value_states = self.v_proj(current_states)
    query_states = query_states.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
    scores = torch.matmul(query_states, key_states.transpose(-2, -1))
    if attention_mask is not None:
        scores = scores + attention_mask
    attn_weights = nn.functional.softmax(scores, dim=-1)
    attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    flat_q = query_states.reshape(batch_size * self.num_heads, query_len, self.head_dim)
    flat_k = key_states.reshape(batch_size * self.num_heads, key_len, self.head_dim)
    flat_v = value_states.reshape(batch_size * self.num_heads, key_len, self.head_dim)
    flat_attn = attn_probs.reshape(batch_size * self.num_heads, query_len, key_len)
    keep_mask = None
    if attention_mask is not None:
        keep_mask = attention_mask > -1.0e6
        keep_mask = keep_mask.expand(batch_size, self.num_heads, query_len, key_len).reshape(batch_size * self.num_heads, query_len, key_len)
    attn_output, _, _, _ = self._attention_block(flat_q, flat_k, flat_v, attention_weights=flat_attn, mask=keep_mask)
    attn_output = attn_output.reshape(batch_size, self.num_heads, query_len, self.head_dim)
    attn_output = attn_output.transpose(1, 2).reshape(batch_size, query_len, embed_dim)
    attn_output = self.out_proj(attn_output)
    attn_weights_reshaped = attn_weights if output_attentions else None
    return attn_output, attn_weights_reshaped, past_key_values


class AttentionedTorchMultiheadAttention(nn.MultiheadAttention):
    @classmethod
    def from_attention(cls, source: nn.MultiheadAttention, *, gate_init: float, gate_leak: float):
        wrapped = cls(
            embed_dim=source.embed_dim,
            num_heads=source.num_heads,
            dropout=source.dropout,
            bias=source.in_proj_bias is not None,
            add_bias_kv=source.bias_k is not None,
            add_zero_attn=source.add_zero_attn,
            kdim=source.kdim,
            vdim=source.vdim,
            batch_first=source.batch_first,
            device=source.in_proj_weight.device,
            dtype=source.in_proj_weight.dtype,
        )
        wrapped.load_state_dict(source.state_dict(), strict=False)
        wrapped.attention = AttentionBlock(
            wrapped.head_dim,
            wrapped.head_dim,
            wrapped.head_dim,
            gate_init=gate_init,
            gate_leak=gate_leak,
            dropout=source.dropout,
        )
        return wrapped

    def forward(self, *args: Any, **kwargs: Any):
        query = args[0] if len(args) > 0 else kwargs.pop("query")
        key = args[1] if len(args) > 1 else kwargs.pop("key")
        value = args[2] if len(args) > 2 else kwargs.pop("value")
        key_padding_mask = kwargs.get("key_padding_mask")
        need_weights = kwargs.get("need_weights", True)
        attn_mask = kwargs.get("attn_mask")
        average_attn_weights = kwargs.get("average_attn_weights", True)

        if self.batch_first:
            batch_query, batch_key, batch_value = query, key, value
        else:
            batch_query, batch_key, batch_value = query.transpose(0, 1), key.transpose(0, 1), value.transpose(0, 1)

        if self._qkv_same_embed_dim:
            q_proj, k_proj, v_proj = torch.nn.functional._in_projection_packed(
                batch_query,
                batch_key,
                batch_value,
                self.in_proj_weight,
                self.in_proj_bias,
            )
        else:
            bias_q, bias_k, bias_v = (None, None, None) if self.in_proj_bias is None else self.in_proj_bias.chunk(3)
            q_proj, k_proj, v_proj = torch.nn.functional._in_projection(
                batch_query,
                batch_key,
                batch_value,
                self.q_proj_weight,
                self.k_proj_weight,
                self.v_proj_weight,
                bias_q,
                bias_k,
                bias_v,
            )

        batch_size, query_len, _ = q_proj.shape
        key_len = k_proj.shape[1]
        q_proj = q_proj.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = k_proj.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_proj = v_proj.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        block_mask = None
        additive_mask = None
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                block_mask = attn_mask
            else:
                additive_mask = attn_mask
        if block_mask is not None and block_mask.ndim == 2:
            block_mask = block_mask.unsqueeze(0).unsqueeze(0)
        elif block_mask is not None and block_mask.ndim == 3:
            block_mask = block_mask.view(batch_size, self.num_heads, query_len, key_len)
        if key_padding_mask is not None:
            pad_mask = key_padding_mask.to(torch.bool).view(batch_size, 1, 1, key_len)
            block_mask = pad_mask if block_mask is None else (block_mask | pad_mask)

        if additive_mask is not None:
            scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
            if additive_mask.ndim == 2:
                scores = scores + additive_mask.view(1, 1, query_len, key_len)
            elif additive_mask.ndim == 3:
                scores = scores + additive_mask.view(batch_size, self.num_heads, query_len, key_len)
            x, attn = masked_attention_for_attention(
                q_proj,
                k_proj,
                v_proj,
                mask=block_mask,
                dropout=self.dropout if isinstance(self.dropout, nn.Module) else None,
                attention=None,
            )
            # Preserve additive masks by recomputing the softmax path explicitly.
            if block_mask is not None:
                scores = scores.masked_fill(block_mask, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=-1)
            flat_q = q_proj.reshape(batch_size * self.num_heads, query_len, self.head_dim)
            flat_k = k_proj.reshape(batch_size * self.num_heads, key_len, self.head_dim)
            flat_v = v_proj.reshape(batch_size * self.num_heads, key_len, self.head_dim)
            flat_attn = attn.reshape(batch_size * self.num_heads, query_len, key_len)
            keep_mask = None if block_mask is None else (~block_mask).expand(batch_size, self.num_heads, query_len, key_len).reshape(batch_size * self.num_heads, query_len, key_len)
            x, _, _, _ = self.attention(flat_q, flat_k, flat_v, attention_weights=flat_attn, mask=keep_mask)
            x = x.reshape(batch_size, self.num_heads, query_len, self.head_dim)
        else:
            dropout = nn.Dropout(self.dropout) if self.training and self.dropout > 0 else None
            x, attn = masked_attention_for_attention(q_proj, k_proj, v_proj, mask=block_mask, dropout=dropout, attention=self.attention)

        x = x.transpose(1, 2).contiguous().view(batch_size, query_len, self.embed_dim)
        x = self.out_proj(x)
        if not self.batch_first:
            x = x.transpose(0, 1)
        if not need_weights:
            return x, None
        weights = attn.mean(dim=1) if average_attn_weights else attn
        return x, weights


__all__ = [
    "AttentionedEasyTPPMultiHeadAttention",
    "AttentionedDyGLibMultiHeadAttention",
    "AttentionedNeuralSTPPMultiheadAttention",
    "AttentionedTorchMultiheadAttention",
    "masked_attention_for_attention",
    "replace_torch_multihead_attention",
    "replace_huggingface_attention",
]
