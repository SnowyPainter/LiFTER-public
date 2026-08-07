from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelResult:
    logits: torch.Tensor
    aux: dict[str, torch.Tensor]


class AttentionModel(nn.Module):
    model_name = "attention_model"

    def __init__(self, *, use_attention: bool = False) -> None:
        super().__init__()
        self.use_attention = bool(use_attention)

    @staticmethod
    def aux_from_info(info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        gate = info.get("attention_gate")
        logits = info.get("attention_logits")
        aux: dict[str, torch.Tensor] = {}
        if gate is not None:
            aux["attention_gate_mean"] = gate.detach().mean()
        if logits is not None:
            aux["attention_logit_mean"] = logits.detach().mean()
        for key in ("edge_attention_gate_mean", "edge_attention_attention_entropy"):
            value = info.get(key)
            if value is not None:
                aux[key] = value.detach()
        return aux
