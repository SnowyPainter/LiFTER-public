from __future__ import annotations

import importlib
import contextlib
import io
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .base import ModelResult
from .external_utils import EXTERNAL_ROOT, import_from_file, isolated_top_level_package, prepend_sys_path
from .attention_patches import (
    AttentionedDyGLibMultiHeadAttention,
    AttentionedEasyTPPMultiHeadAttention,
    AttentionedNeuralSTPPMultiheadAttention,
    replace_huggingface_attention,
    replace_torch_multihead_attention,
)


def _attention_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "gate_init": float(kwargs.pop("gate_init", 0.95)),
        "gate_leak": float(kwargs.pop("gate_leak", 0.05)),
    }


class DyGLibNeighborSampler:
    """Small DyGLib-compatible temporal neighbor sampler for processed CSV data."""

    def __init__(
        self,
        adj_list: list[list[tuple[int, int, float]]],
        *,
        sample_neighbor_strategy: str = "recent",
        seed: int | None = None,
    ) -> None:
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed
        self.random_state = np.random.RandomState(seed) if seed is not None else None
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []
        for neighbors in adj_list:
            ordered = sorted(neighbors, key=lambda item: item[2])
            self.nodes_neighbor_ids.append(np.asarray([item[0] for item in ordered], dtype=np.longlong))
            self.nodes_edge_ids.append(np.asarray([item[1] for item in ordered], dtype=np.longlong))
            self.nodes_neighbor_times.append(np.asarray([item[2] for item in ordered], dtype=np.float32))

    def reset_random_state(self) -> None:
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def find_neighbors_before(self, node_id: int, interact_time: float, return_sampled_probabilities: bool = False):
        if node_id < 0 or node_id >= len(self.nodes_neighbor_ids):
            empty_i = np.asarray([], dtype=np.longlong)
            empty_f = np.asarray([], dtype=np.float32)
            return empty_i, empty_i, empty_f, None
        end = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)
        return (
            self.nodes_neighbor_ids[node_id][:end],
            self.nodes_edge_ids[node_id][:end],
            self.nodes_neighbor_times[node_id][:end],
            None,
        )

    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20):
        node_ids = np.asarray(node_ids, dtype=np.longlong)
        node_interact_times = np.asarray(node_interact_times, dtype=np.float64)
        neighbor_ids = np.zeros((len(node_ids), num_neighbors), dtype=np.longlong)
        edge_ids = np.zeros((len(node_ids), num_neighbors), dtype=np.longlong)
        times = np.zeros((len(node_ids), num_neighbors), dtype=np.float32)
        for idx, (node_id, ts) in enumerate(zip(node_ids, node_interact_times)):
            n_ids, e_ids, n_times, _ = self.find_neighbors_before(int(node_id), float(ts))
            if len(n_ids) == 0:
                continue
            if self.sample_neighbor_strategy == "uniform":
                rng = self.random_state if self.random_state is not None else np.random
                selected = rng.choice(len(n_ids), size=num_neighbors, replace=True)
                selected = selected[np.argsort(n_times[selected])]
                chosen_n, chosen_e, chosen_t = n_ids[selected], e_ids[selected], n_times[selected]
            else:
                chosen_n = n_ids[-num_neighbors:]
                chosen_e = e_ids[-num_neighbors:]
                chosen_t = n_times[-num_neighbors:]
            start = num_neighbors - len(chosen_n)
            neighbor_ids[idx, start:] = chosen_n
            edge_ids[idx, start:] = chosen_e
            times[idx, start:] = chosen_t
        return neighbor_ids, edge_ids, times

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray):
        node_lists, edge_lists, time_lists = [], [], []
        for node_id, ts in zip(np.asarray(node_ids, dtype=np.longlong), np.asarray(node_interact_times, dtype=np.float64)):
            n_ids, e_ids, n_times, _ = self.find_neighbors_before(int(node_id), float(ts))
            node_lists.append(n_ids)
            edge_lists.append(e_ids)
            time_lists.append(n_times)
        return node_lists, edge_lists, time_lists


def build_dyglib_inputs(
    frame: Any,
    *,
    node_feat_dim: int,
    edge_feat_dim: int,
    sample_neighbor_strategy: str = "recent",
    seed: int = 0,
) -> dict[str, Any]:
    src = np.asarray(frame["src"], dtype=np.longlong) + 1
    dst = np.asarray(frame["dst"], dtype=np.longlong) + 1
    timestamps = np.asarray(frame["timestamp"], dtype=np.float32)
    edge_ids = np.arange(1, len(src) + 1, dtype=np.longlong)
    max_node_id = int(max(src.max(initial=0), dst.max(initial=0)))

    node_raw_features = np.zeros((max_node_id + 1, int(node_feat_dim)), dtype=np.float32)
    if node_feat_dim > 0 and max_node_id > 0:
        dims = min(int(node_feat_dim), 16)
        node_ids = np.arange(max_node_id + 1, dtype=np.float32)[:, None]
        freqs = np.arange(1, dims + 1, dtype=np.float32)[None, :]
        node_raw_features[:, :dims] = np.sin(node_ids * freqs * 0.0001)
        node_raw_features[0, :] = 0.0

    edge_raw_features = np.zeros((len(edge_ids) + 1, int(edge_feat_dim)), dtype=np.float32)
    feat_columns = [column for column in getattr(frame, "columns", []) if str(column).startswith("feat_")]
    if feat_columns and edge_feat_dim > 0:
        values = frame[feat_columns].to_numpy(dtype=np.float32)
        width = min(values.shape[1], int(edge_feat_dim))
        edge_raw_features[1:, :width] = values[:, :width]

    adj_list: list[list[tuple[int, int, float]]] = [[] for _ in range(max_node_id + 1)]
    for s_id, d_id, e_id, ts in zip(src, dst, edge_ids, timestamps):
        adj_list[int(s_id)].append((int(d_id), int(e_id), float(ts)))
        adj_list[int(d_id)].append((int(s_id), int(e_id), float(ts)))

    return {
        "node_raw_features": node_raw_features,
        "edge_raw_features": edge_raw_features,
        "neighbor_sampler": DyGLibNeighborSampler(
            adj_list,
            sample_neighbor_strategy=sample_neighbor_strategy,
            seed=seed,
        ),
        "node_id_offset": 1,
    }


class OfficialModelAdapter(nn.Module):
    """Thin wrapper around an official/reference implementation."""

    official_name: str
    attention_replacements: int

    def __init__(
        self,
        model: nn.Module,
        *,
        official_name: str,
        use_attention: bool,
        attention_replacements: int = 0,
        attention_expected: bool | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.official_name = official_name
        self.use_attention = bool(use_attention)
        self.attention_replacements = int(attention_replacements)
        self.attention_expected = bool(use_attention) if attention_expected is None else bool(attention_expected)

    def forward(self, *args: Any, **kwargs: Any):
        return self.model(*args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            model = super().__getattr__("model")
            return getattr(model, name)


class OfficialEasyTPP(OfficialModelAdapter):
    MODEL_MODULES = {
        "SAHP": "easy_tpp.model.torch_model.torch_sahp",
        "THP": "easy_tpp.model.torch_model.torch_thp",
        "AttNHP": "easy_tpp.model.torch_model.torch_attnhp",
    }

    def __init__(
        self,
        model_id: str,
        *,
        num_event_types: int,
        hidden_dim: int = 128,
        time_dim: int = 32,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_attention: bool = False,
        gpu: int = -1,
        **kwargs: Any,
    ) -> None:
        attention_kwargs = _attention_kwargs(kwargs)
        root = EXTERNAL_ROOT / "EasyTemporalPointProcess"
        if not root.exists():
            raise FileNotFoundError(f"EasyTemporalPointProcess repo is missing: {root}")
        with prepend_sys_path(root):
            model_module = importlib.import_module(self.MODEL_MODULES[model_id])
            base_layer_module = importlib.import_module("easy_tpp.model.torch_model.torch_baselayer")
            if use_attention:
                model_module.MultiHeadAttention = lambda n_head, d_input, d_model, dropout=0.1, output_linear=False: AttentionedEasyTPPMultiHeadAttention(
                    n_head,
                    d_input,
                    d_model,
                    dropout,
                    output_linear,
                    **attention_kwargs,
                )
            else:
                model_module.MultiHeadAttention = base_layer_module.MultiHeadAttention
            config = SimpleNamespace(
                model_id=model_id,
                hidden_size=hidden_dim,
                time_emb_size=time_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout_rate=dropout,
                use_ln=bool(kwargs.pop("use_ln", False)),
                loss_integral_num_sample_per_step=int(kwargs.pop("loss_integral_num_sample_per_step", 20)),
                use_mc_samples=bool(kwargs.pop("use_mc_samples", True)),
                thinning=kwargs.pop("thinning", None),
                model_specs=kwargs.pop("model_specs", {}),
                pretrained_model_dir=kwargs.pop("pretrained_model_dir", None),
                rnn_type=kwargs.pop("rnn_type", "LSTM"),
                sharing_param_layer=kwargs.pop("sharing_param_layer", False),
                is_training=kwargs.pop("training", False),
                num_event_types=int(num_event_types),
                num_event_types_pad=int(kwargs.pop("num_event_types_pad", num_event_types + 1)),
                pad_token_id=int(kwargs.pop("event_pad_index", num_event_types)),
                gpu=gpu,
            )
            model_cls = getattr(model_module, model_id)
            model = model_cls(config)
        replacements = sum(1 for module in model.modules() if isinstance(module, AttentionedEasyTPPMultiHeadAttention))
        super().__init__(model, official_name=f"EasyTPP.{model_id}", use_attention=use_attention, attention_replacements=replacements)
        self.model_id = model_id
        self.num_event_types = int(num_event_types)
        output_dim = hidden_dim * num_heads if model_id == "AttNHP" else hidden_dim
        self.time_head = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, 1),
            nn.Softplus(),
        )

    def forward(self, event_types: torch.Tensor, times: torch.Tensor, mask: torch.Tensor | None = None) -> ModelResult:
        valid = mask.to(torch.bool) if mask is not None else torch.ones_like(event_types, dtype=torch.bool)
        event_seqs = event_types.long().clamp(min=0, max=self.num_event_types)
        event_seqs = event_seqs.masked_fill(~valid, self.model.pad_token_id)
        time_seqs = times.float().masked_fill(~valid, 0.0)
        time_delta = torch.zeros_like(time_seqs)
        time_delta[:, 1:] = (time_seqs[:, 1:] - time_seqs[:, :-1]).clamp_min(0.0)

        batch_size, seq_len = event_seqs.shape
        causal_block = torch.triu(torch.ones(seq_len, seq_len, device=event_seqs.device, dtype=torch.bool), diagonal=1)
        attention_mask = causal_block.unsqueeze(0).expand(batch_size, -1, -1).clone()
        attention_mask = attention_mask | (~valid).unsqueeze(1).expand(-1, seq_len, -1)

        if self.model_id == "SAHP":
            encoded = self.model(time_seqs, time_delta, event_seqs, attention_mask)
            last_state = self.model.get_logits_at_last_step(encoded, valid)
            decayed = self.model.state_decay(last_state, torch.zeros(last_state.shape[0], 1, device=last_state.device))
            logits = torch.log(self.model.softplus(decayed).clamp_min(self.model.eps))
        elif self.model_id == "THP":
            encoded = self.model(time_seqs, event_seqs, attention_mask)
            last_state = self.model.get_logits_at_last_step(encoded, valid)
            intensity = self.model.layer_intensity_hidden(last_state) + self.model.factor_intensity_base
            logits = torch.log(self.model.softplus(intensity).clamp_min(self.model.eps))
        else:
            encoded = self.model(time_seqs, event_seqs, attention_mask)
            last_state = self.model.get_logits_at_last_step(encoded, valid)
            logits = torch.log(self.model.layer_intensity(last_state).clamp_min(self.model.eps))
        return ModelResult(logits=logits, aux={"next_time": self.time_head(last_state).squeeze(-1)})


class OfficialDyGLib(OfficialModelAdapter):
    MODEL_FILES = {
        "TGAT": "TGAT",
        "DyGFormer": "DyGFormer",
        "GraphMixer": "GraphMixer",
    }

    def __init__(self, model_id: str, *, use_attention: bool = False, **kwargs: Any) -> None:
        attention_kwargs = _attention_kwargs(kwargs)
        root = EXTERNAL_ROOT / "DyGLib"
        if not root.exists():
            raise FileNotFoundError(f"DyGLib repo is missing: {root}")
        if "time_dim" in kwargs and "time_feat_dim" not in kwargs:
            kwargs["time_feat_dim"] = kwargs.pop("time_dim")
        hidden_dim = kwargs.pop("hidden_dim", None)
        kwargs.pop("gpu", None)
        kwargs.pop("edge_feat_dim", None)
        kwargs.pop("num_nodes", None)
        node_id_offset = int(kwargs.pop("node_id_offset", 0))
        num_neighbors = int(kwargs.pop("num_neighbors", kwargs.get("num_tokens", 20)))
        time_gap = int(kwargs.pop("time_gap", 2000))
        if model_id == "DyGFormer":
            kwargs.setdefault("channel_embedding_dim", hidden_dim or 128)
        elif model_id == "GraphMixer":
            kwargs.setdefault("num_tokens", num_neighbors)
            if "mixer_layers" in kwargs and "num_layers" not in kwargs:
                kwargs["num_layers"] = kwargs.pop("mixer_layers")
            kwargs.pop("num_heads", None)
        elif model_id == "TGAT":
            kwargs.pop("channel_embedding_dim", None)
            kwargs.pop("num_tokens", None)
        if "node_raw_features" not in kwargs or "edge_raw_features" not in kwargs or "neighbor_sampler" not in kwargs:
            raise ValueError(
                f"Official DyGLib {model_id} requires node_raw_features, edge_raw_features, and neighbor_sampler. "
                "Build these from the processed CTDG dataset before constructing the official model."
            )
        with prepend_sys_path(root), isolated_top_level_package("models", root / "models"):
            modules = importlib.reload(importlib.import_module("models.modules"))
            patched_tgat_attention = None
            if use_attention and model_id == "TGAT":
                patched_tgat_attention = lambda node_feat_dim, edge_feat_dim, time_feat_dim, num_heads=2, dropout=0.1: AttentionedDyGLibMultiHeadAttention(
                    node_feat_dim,
                    edge_feat_dim,
                    time_feat_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    **attention_kwargs,
                )
                modules.MultiHeadAttention = patched_tgat_attention
            model_module = importlib.reload(importlib.import_module(f"models.{self.MODEL_FILES[model_id]}"))
            if patched_tgat_attention is not None:
                model_module.MultiHeadAttention = patched_tgat_attention
            model_cls = getattr(model_module, model_id)
            model = model_cls(**kwargs)
            link_predictor = modules.MergeLayer(
                input_dim1=model.node_feat_dim,
                input_dim2=model.node_feat_dim,
                hidden_dim=model.node_feat_dim,
                output_dim=1,
            )
        replacements = 0
        if use_attention and model_id == "DyGFormer":
            replacements = replace_torch_multihead_attention(model, **attention_kwargs)
        elif use_attention and model_id == "TGAT":
            replacements = len(getattr(model, "temporal_conv_layers", []))
        elif use_attention and model_id == "GraphMixer":
            replacements = replace_torch_multihead_attention(model, **attention_kwargs)
        super().__init__(
            model,
            official_name=f"DyGLib.{model_id}",
            use_attention=use_attention,
            attention_replacements=replacements,
            attention_expected=bool(use_attention and model_id != "GraphMixer"),
        )
        self.model_id = model_id
        self.num_neighbors = num_neighbors
        self.time_gap = time_gap
        self.node_id_offset = node_id_offset
        self.link_predictor = link_predictor

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        timestamp: torch.Tensor,
        *_: Any,
        **__: Any,
    ) -> ModelResult:
        src_np = (src.detach().cpu().numpy().astype(np.longlong) + self.node_id_offset)
        dst_np = (dst.detach().cpu().numpy().astype(np.longlong) + self.node_id_offset)
        ts_np = timestamp.detach().cpu().numpy().astype(np.float64)
        if self.model_id == "TGAT":
            src_emb, dst_emb = self.model.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src_np,
                dst_node_ids=dst_np,
                node_interact_times=ts_np,
                num_neighbors=self.num_neighbors,
            )
        elif self.model_id == "GraphMixer":
            src_emb, dst_emb = self.model.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src_np,
                dst_node_ids=dst_np,
                node_interact_times=ts_np,
                num_neighbors=self.num_neighbors,
                time_gap=self.time_gap,
            )
        else:
            src_emb, dst_emb = self.model.compute_src_dst_node_temporal_embeddings(
                src_node_ids=src_np,
                dst_node_ids=dst_np,
                node_interact_times=ts_np,
            )
        logits = self.link_predictor(input_1=src_emb, input_2=dst_emb).squeeze(-1)
        return ModelResult(logits=logits, aux={})


class OfficialRENET(OfficialModelAdapter):
    def __init__(
        self,
        *,
        num_entities: int,
        num_relations: int,
        hidden_dim: int = 128,
        use_attention: bool = False,
        **kwargs: Any,
    ) -> None:
        root = EXTERNAL_ROOT / "RE-Net"
        with prepend_sys_path(root):
            try:
                module = import_from_file("attention_external_renet_model", root / "model.py")
            except ModuleNotFoundError as exc:
                if exc.name == "dgl":
                    raise ImportError(
                        "Official RE-NET requires DGL. Install it in this environment, for example "
                        "`/opt/conda/bin/pip install dgl`, then rebuild the model."
                    ) from exc
                raise
            module.defaultdict = defaultdict
            model = module.RENet(
                in_dim=int(num_entities),
                h_dim=int(hidden_dim),
                num_rels=int(num_relations),
                dropout=float(kwargs.pop("dropout", 0.0)),
                model=int(kwargs.pop("rgcn_model", kwargs.pop("model", 0))),
                seq_len=int(kwargs.pop("seq_len", 10)),
                num_k=int(kwargs.pop("num_k", 10)),
            )
        super().__init__(model, official_name="RE-Net.RENet", use_attention=use_attention, attention_replacements=0)


class OfficialCyGNet(OfficialModelAdapter):
    def __init__(
        self,
        *,
        num_entities: int,
        num_relations: int,
        num_times: int = 365,
        hidden_dim: int = 128,
        use_attention: bool = False,
        **kwargs: Any,
    ) -> None:
        root = EXTERNAL_ROOT / "CyGNet"
        with prepend_sys_path(root), contextlib.redirect_stdout(io.StringIO()):
            module = import_from_file("attention_external_cygnet_link_prediction", root / "link_prediction.py")
            model = module.link_prediction(
                i_dim=int(num_entities),
                h_dim=int(hidden_dim),
                num_rels=int(num_relations),
                num_times=int(num_times),
                use_cuda=bool(kwargs.pop("use_cuda", False)),
            )
        super().__init__(model, official_name="CyGNet.link_prediction", use_attention=use_attention, attention_replacements=0)


class OfficialXERTE(OfficialModelAdapter):
    def __init__(self, *, use_attention: bool = False, **kwargs: Any) -> None:
        attention_kwargs = _attention_kwargs(kwargs)
        if "num_entities" in kwargs and "num_entity" not in kwargs:
            kwargs["num_entity"] = kwargs.pop("num_entities")
        if "num_relations" in kwargs and "num_rel" not in kwargs:
            kwargs["num_rel"] = kwargs.pop("num_relations")
        if "expansion_layers" in kwargs and "DP_steps" not in kwargs:
            kwargs["DP_steps"] = kwargs.pop("expansion_layers")
        hidden_dim = kwargs.pop("hidden_dim", None)
        kwargs.pop("time_dim", None)
        if hidden_dim is not None and "emb_dim" not in kwargs:
            steps = int(kwargs.get("DP_steps", 3))
            kwargs["emb_dim"] = [int(hidden_dim)] * (steps + 1)
        root = EXTERNAL_ROOT / "xERTE" / "tKGR"
        with prepend_sys_path(root):
            module = import_from_file("attention_external_xerte_model", root / "model.py")
            model_cls = getattr(module, "xERTE")
            model = model_cls(**kwargs)
        replacements = 0
        if use_attention:
            replacements = replace_torch_multihead_attention(model, **attention_kwargs)
        super().__init__(model, official_name="xERTE.xERTE", use_attention=use_attention, attention_replacements=replacements)


class OfficialNSTPP(OfficialModelAdapter):
    def __init__(
        self,
        *,
        use_attention: bool = False,
        dim: int = 2,
        hidden_dims: list[int] | None = None,
        tpp_hidden_dims: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        attention_kwargs = _attention_kwargs(kwargs)
        root = EXTERNAL_ROOT / "neural_stpp"
        hidden_dim = kwargs.pop("hidden_dim", None)
        time_dim = kwargs.pop("time_dim", None)
        kwargs.pop("num_marks", None)
        kwargs.pop("dropout", None)
        kwargs.pop("num_layers", None)
        kwargs.pop("num_heads", None)
        kwargs.pop("device", None)
        kwargs.pop("gpu", None)
        hidden_dims = hidden_dims or ([int(hidden_dim)] * 3 if hidden_dim is not None else [64, 64, 64])
        tpp_hidden_dims = tpp_hidden_dims or ([max(8, int(time_dim or 8)), max(8, int(hidden_dim or 20))] if (time_dim is not None or hidden_dim is not None) else [8, 20])
        with prepend_sys_path(root), isolated_top_level_package("models", root / "models"):
            try:
                stpp_module = importlib.import_module("models.spatiotemporal")
                if use_attention:
                    attncnf_module = importlib.import_module("models.spatial.attncnf")
                    attncnf_module.MultiheadAttention = lambda embed_dim, num_heads: AttentionedNeuralSTPPMultiheadAttention(
                        embed_dim,
                        num_heads,
                        **attention_kwargs,
                    )
            except ModuleNotFoundError as exc:
                if exc.name == "torchdiffeq":
                    raise ImportError(
                        "Official NSTPP requires torchdiffeq. Install it with "
                        "`/opt/conda/bin/pip install torchdiffeq`, then rebuild the model."
                    ) from exc
                raise
            model = stpp_module.SelfAttentiveCNFSpatiotemporalModel(
                dim=dim,
                hidden_dims=hidden_dims,
                tpp_hidden_dims=tpp_hidden_dims,
                **kwargs,
            )
        replacements = 0
        if use_attention:
            replacements = sum(1 for module in model.modules() if isinstance(module, AttentionedNeuralSTPPMultiheadAttention))
        super().__init__(model, official_name="neural_stpp.SelfAttentiveCNFSpatiotemporalModel", use_attention=use_attention, attention_replacements=replacements)


class OfficialDeepSTPP(OfficialModelAdapter):
    def __init__(self, *, config: Any | None = None, device: str = "cpu", use_attention: bool = False, **kwargs: Any) -> None:
        attention_kwargs = _attention_kwargs(kwargs)
        root = EXTERNAL_ROOT / "deepstpp"
        if config is None:
            hidden_dim = int(kwargs.pop("hidden_dim", kwargs.pop("emb_dim", 128)))
            kwargs.pop("num_marks", None)
            kwargs.pop("time_dim", None)
            config = SimpleNamespace(
                emb_dim=hidden_dim,
                hid_dim=int(kwargs.pop("hid_dim", 4 * hidden_dim)),
                dropout=float(kwargs.pop("dropout", 0.1)),
                seq_len=int(kwargs.pop("seq_len", 32)),
                num_head=int(kwargs.pop("num_heads", kwargs.pop("num_head", 4))),
                nlayers=int(kwargs.pop("num_layers", kwargs.pop("nlayers", 2))),
                z_dim=int(kwargs.pop("z_dim", hidden_dim)),
                num_points=int(kwargs.pop("num_points", 32)),
                decoder_n_layer=int(kwargs.pop("decoder_n_layer", 2)),
                opt=kwargs.pop("opt", "adam"),
                lr=float(kwargs.pop("lr", 1.0e-3)),
                momentum=float(kwargs.pop("momentum", 0.9)),
                sample=bool(kwargs.pop("sample", True)),
                beta=float(kwargs.pop("beta", 1.0)),
                constrain_b=kwargs.pop("constrain_b", "softplus"),
                b_max=float(kwargs.pop("b_max", 10.0)),
                s_min=float(kwargs.pop("s_min", 1.0e-5)),
            )
        with prepend_sys_path(root / "src"):
            module = import_from_file("attention_external_deepstpp_model", root / "src" / "model.py")
            model = module.DeepSTPP(config, device)
        replacements = replace_torch_multihead_attention(model, **attention_kwargs) if use_attention else 0
        super().__init__(model, official_name="deepstpp.DeepSTPP", use_attention=use_attention, attention_replacements=replacements)


class OfficialFiD(OfficialModelAdapter):
    def __init__(self, *, config: Any | None = None, use_attention: bool = False, **kwargs: Any) -> None:
        fid_src = EXTERNAL_ROOT / "FiD" / "src"
        if config is None:
            try:
                from transformers import T5Config
            except Exception as exc:  # pragma: no cover - depends on optional external deps
                raise ImportError(
                    "FiD official implementation requires transformers dependencies. "
                    "Install them with `/opt/conda/bin/pip install transformers` or pass a ready T5Config."
                ) from exc
            config = T5Config(
                vocab_size=int(kwargs.pop("vocab_size", 32128)),
                d_model=int(kwargs.pop("hidden_dim", 256)),
                    d_ff=int(kwargs.pop("d_ff", 1024)),
                    num_layers=int(kwargs.pop("num_layers", 2)),
                    num_decoder_layers=int(kwargs.pop("num_decoder_layers", 2)),
                num_heads=int(kwargs.pop("num_heads", 4)),
            )
        with prepend_sys_path(fid_src):
            module = import_from_file("attention_external_fid_model", fid_src / "model.py")
            model = module.FiDT5(config)
            # The upstream FiD wrapper targets older transformers attention
            # signatures.  Unwrap the encoder for compatibility with the
            # currently installed transformers while keeping the FiDT5 model.
            if hasattr(model, "unwrap_encoder"):
                model.unwrap_encoder()
        replacements = replace_huggingface_attention(model, **_attention_kwargs(kwargs)) if use_attention else 0
        super().__init__(model, official_name="FiD.FiDT5", use_attention=use_attention, attention_replacements=replacements)

    def forward(self, question_ids: torch.Tensor, passage_ids: torch.Tensor, passage_mask: torch.Tensor | None = None) -> ModelResult:
        batch_size, passages, _ = passage_ids.shape
        question = question_ids.long().unsqueeze(1).expand(-1, passages, -1)
        input_ids = torch.cat([question, passage_ids.long()], dim=-1)
        attention_mask = input_ids.ne(0).long()
        if passage_mask is not None:
            attention_mask = attention_mask * passage_mask.to(attention_mask.device, dtype=attention_mask.dtype).unsqueeze(-1)
        decoder_start = getattr(self.model.config, "decoder_start_token_id", None)
        if decoder_start is None:
            decoder_start = getattr(self.model.config, "pad_token_id", 0)
        decoder_input_ids = torch.full((batch_size, 1), int(decoder_start or 0), dtype=torch.long, device=input_ids.device)
        outputs = self.model(
            input_ids=input_ids.reshape(batch_size, -1),
            attention_mask=attention_mask.reshape(batch_size, -1),
            decoder_input_ids=decoder_input_ids,
        )
        return ModelResult(logits=outputs.logits[:, 0, :], aux={})


class OfficialLED(OfficialModelAdapter):
    def __init__(self, *, config: Any | None = None, use_attention: bool = False, **kwargs: Any) -> None:
        try:
            from transformers import LEDConfig, LEDForConditionalGeneration
        except Exception as exc:  # pragma: no cover - depends on optional external deps
            raise ImportError(
                "LED official implementation requires transformers dependencies. "
                "Install them with `/opt/conda/bin/pip install transformers` or pass a ready LEDConfig."
            ) from exc
        if config is None:
            num_heads = int(kwargs.pop("num_heads", 4))
            config = LEDConfig(
                vocab_size=int(kwargs.pop("vocab_size", 50265)),
                d_model=int(kwargs.pop("hidden_dim", 256)),
                encoder_layers=int(kwargs.pop("encoder_layers", 2)),
                decoder_layers=int(kwargs.pop("decoder_layers", 2)),
                encoder_attention_heads=num_heads,
                decoder_attention_heads=num_heads,
                encoder_ffn_dim=int(kwargs.pop("encoder_ffn_dim", 1024)),
                decoder_ffn_dim=int(kwargs.pop("decoder_ffn_dim", 1024)),
            )
        model = LEDForConditionalGeneration(config)
        replacements = replace_huggingface_attention(model, **_attention_kwargs(kwargs)) if use_attention else 0
        super().__init__(model, official_name="transformers.LEDForConditionalGeneration", use_attention=use_attention, attention_replacements=replacements)

    def forward(self, question_ids: torch.Tensor, passage_ids: torch.Tensor, passage_mask: torch.Tensor | None = None) -> ModelResult:
        batch_size, passages, tokens = passage_ids.shape
        question = question_ids.long().unsqueeze(1).expand(-1, passages, -1)
        per_passage = torch.cat([question, passage_ids.long()], dim=-1)
        input_ids = per_passage.reshape(batch_size, passages * per_passage.shape[-1])
        attention_mask = input_ids.ne(0).long()
        if passage_mask is not None:
            expanded = passage_mask.to(input_ids.device, dtype=attention_mask.dtype).unsqueeze(-1).expand(-1, -1, per_passage.shape[-1])
            attention_mask = attention_mask * expanded.reshape(batch_size, -1)
        decoder_start = getattr(self.model.config, "decoder_start_token_id", None)
        if decoder_start is None:
            decoder_start = getattr(self.model.config, "pad_token_id", 0)
        decoder_input_ids = torch.full((batch_size, 1), int(decoder_start or 0), dtype=torch.long, device=input_ids.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
        return ModelResult(logits=outputs.logits[:, 0, :], aux={})


def build_official_model(name: str, *, use_attention: bool = False, **kwargs: Any) -> OfficialModelAdapter:
    if name in {"SAHP", "THP", "AttNHP"}:
        return OfficialEasyTPP(name, use_attention=use_attention, **kwargs)
    if name in {"TGAT", "DyGFormer", "GraphMixer"}:
        return OfficialDyGLib(name, use_attention=use_attention, **kwargs)
    if name == "RE-NET":
        return OfficialRENET(use_attention=use_attention, **kwargs)
    if name == "CyGNet":
        return OfficialCyGNet(use_attention=use_attention, **kwargs)
    if name == "xERTE":
        return OfficialXERTE(use_attention=use_attention, **kwargs)
    if name == "NSTPP":
        return OfficialNSTPP(use_attention=use_attention, **kwargs)
    if name == "DeepSTPP":
        return OfficialDeepSTPP(use_attention=use_attention, **kwargs)
    if name == "Transformer-STPP":
        attention_kwargs = _attention_kwargs(kwargs)
        model = __import__("models.stpp", fromlist=["TransformerSTPP"]).TransformerSTPP(use_attention=use_attention, **kwargs)
        replacements = replace_torch_multihead_attention(model, **attention_kwargs) if use_attention else 0
        return OfficialModelAdapter(model, official_name="local.TransformerSTPP", use_attention=use_attention, attention_replacements=replacements)
    if name == "FiD":
        return OfficialFiD(use_attention=use_attention, **kwargs)
    if name == "LED":
        return OfficialLED(use_attention=use_attention, **kwargs)
    raise ValueError(f"No official adapter registered for {name!r}")


__all__ = [
    "OfficialCyGNet",
    "OfficialDyGLib",
    "OfficialEasyTPP",
    "OfficialFiD",
    "OfficialDeepSTPP",
    "OfficialLED",
    "OfficialModelAdapter",
    "OfficialNSTPP",
    "OfficialRENET",
    "OfficialXERTE",
    "build_dyglib_inputs",
    "build_official_model",
]
