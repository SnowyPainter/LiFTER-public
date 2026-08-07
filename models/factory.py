from __future__ import annotations

from typing import Any

from .craft import CRAFT, CRAFTR
from .ctdg import DyGFormer, GraphMixer, TGAT
from .mtpp import AttNHP, SAHP, THP
from .official import build_official_model
from .rag import FiD, LED
from .stpp import DeepSTPP, NSTPP, TransformerSTPP
from .tkg import CyGNet, RENET, XERTE
from .attention_patches import replace_torch_multihead_attention
from .lifter import LiFTER
from .explainability import SIG, TGIB
from .tgn import NativeTGN


MODEL_REGISTRY = {
    "TGAT": TGAT,
    "DyGFormer": DyGFormer,
    "GraphMixer": GraphMixer,
    "CRAFT": CRAFT,
    "CRAFT-R": CRAFTR,
    "SAHP": SAHP,
    "THP": THP,
    "AttNHP": AttNHP,
    "Transformer-STPP": TransformerSTPP,
    "NSTPP": NSTPP,
    "DeepSTPP": DeepSTPP,
    "RE-NET": RENET,
    "xERTE": XERTE,
    "CyGNet": CyGNet,
    "FiD": FiD,
    "LED": LED,
    "LiFTER": LiFTER,
    "TGIB": TGIB,
    "SIG": SIG,
    "TGN": NativeTGN,
}


def build_model(name: str, *, use_attention: bool = False, implementation: str = "reference", **kwargs: Any):
    if implementation == "official":
        return build_official_model(name, use_attention=use_attention, **kwargs)
    if implementation not in {"reference", "native", "lightweight"}:
        raise ValueError("implementation must be one of: official, reference, native, lightweight")
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {name!r}. Available: {available}")
    model = MODEL_REGISTRY[name](use_attention=use_attention, **kwargs)
    replacements = 0
    if use_attention and name not in {"FiD", "LED"}:
        replacements = replace_torch_multihead_attention(
            model,
            gate_init=float(kwargs.get("gate_init", 0.95)),
            gate_leak=float(kwargs.get("gate_leak", 0.05)),
        )
    model.attention_replacements = replacements
    return model


__all__ = ["MODEL_REGISTRY", "build_model"]
