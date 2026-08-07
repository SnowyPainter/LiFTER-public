from __future__ import annotations

import hashlib
import io
import json
import math
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.factory import build_model
from models.lifter import (
    DEFAULT_FACT_CONTEXT_LEN,
    LIFTER_CONTEXT_TOKEN_DIM,
    lifter_edge_feature_dim,
)
from models.attention import AttentionedAttention
from models.official import build_dyglib_inputs
from models.attention_block import EdgeConditionedAttentionBlock
from models.attention_patches import (
    AttentionedDyGLibMultiHeadAttention,
    AttentionedEasyTPPMultiHeadAttention,
    AttentionedNeuralSTPPMultiheadAttention,
    AttentionedTorchMultiheadAttention,
)


@dataclass(frozen=True)
class RunSpec:
    domain: str
    dataset: str
    dataset_path: Path
    model: str
    variant: str
    seed: int
    implementation: str
    model_config: dict[str, Any]
    use_attention: bool


def hash_token(value: Any, vocab_size: int) -> int:
    text = str(value)
    digest = hashlib.blake2b(text.encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % max(1, vocab_size)


def label_encode(values: pd.Series, limit: int | None = None) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(values.astype(str), sort=True)
    if limit is not None:
        codes = np.asarray([int(code % limit) for code in codes], dtype=np.int64)
        return codes, int(min(limit, len(uniques)))
    return codes.astype(np.int64), int(len(uniques))


def chronological_sample(frame: pd.DataFrame, max_examples: int) -> pd.DataFrame:
    if "timestamp" in frame.columns:
        frame = frame.sort_values("timestamp")
    if len(frame) <= max_examples:
        return frame.reset_index(drop=True)
    return frame.iloc[-max_examples:].reset_index(drop=True)


def read_csv_tail(path: Path, max_examples: int) -> pd.DataFrame:
    """Read the final rows of a line-oriented processed CSV without loading it all."""

    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    with path.open("rb") as handle:
        header = handle.readline()
        handle.seek(0, 2)
        position = handle.tell()
        blocks: list[bytes] = []
        newline_count = 0
        block_size = 1024 * 1024
        while position > len(header) and newline_count <= max_examples:
            read_size = min(block_size, position - len(header))
            position -= read_size
            handle.seek(position)
            block = handle.read(read_size)
            blocks.append(block)
            newline_count += block.count(b"\n")
    body = b"".join(reversed(blocks)).splitlines()[-max_examples:]
    return pd.read_csv(io.BytesIO(header + b"\n".join(body) + b"\n"))


def split_indices(n: int, eval_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    split = max(1, int(n * (1.0 - eval_ratio)))
    split = min(split, max(1, n - 1))
    return np.arange(split), np.arange(split, n)


def batch_iter(indices: np.ndarray, batch_size: int, *, shuffle: bool) -> Iterable[np.ndarray]:
    indices = indices.copy()
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def progress_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("training", {}).get("progress", True))


def epoch_iter(spec: RunSpec, config: dict[str, Any]):
    epochs = int(config["training"]["epochs"])
    return tqdm(
        range(epochs),
        desc=f"{spec.domain}/{spec.dataset} {spec.model} {spec.variant} train",
        unit="epoch",
        leave=True,
        disable=not progress_enabled(config),
    )


def progress_batches(
    indices: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    desc: str,
    config: dict[str, Any],
):
    return tqdm(
        batch_iter(indices, batch_size, shuffle=shuffle),
        total=int(np.ceil(len(indices) / max(1, batch_size))),
        desc=desc,
        unit="batch",
        leave=False,
        disable=not progress_enabled(config),
    )


def resolve_auto_model_config(domain: str, frame: Any, config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    if domain == "ctdg":
        max_node = int(max(frame["src"].max(), frame["dst"].max())) + 1
        if resolved.get("num_nodes") == "auto":
            resolved["num_nodes"] = max_node
    elif domain == "mtpp":
        if resolved.get("num_event_types") == "auto":
            resolved["num_event_types"] = int(frame["mark_id"].max()) + 1
    elif domain == "stpp":
        if resolved.get("num_marks") == "auto":
            resolved["num_marks"] = int(frame["mark_id"].max()) + 1
    elif domain == "tkg":
        if resolved.get("num_entities") == "auto":
            resolved["num_entities"] = int(max(frame["head_id"].max(), frame["tail_id"].max())) + 1
        if resolved.get("num_relations") == "auto":
            resolved["num_relations"] = int(frame["relation_id"].max()) + 1
    return resolved


def build_benchmark_model(spec: RunSpec, frame: Any, config: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model_config = resolve_auto_model_config(spec.domain, frame, spec.model_config)
    if spec.domain == "ctdg" and spec.model == "LiFTER":
        fact_context_len = int(
            model_config.get("fact_context_len", DEFAULT_FACT_CONTEXT_LEN)
        )
        raw_feature_dim = len(
            [column for column in frame.columns if str(column).startswith("feat_")]
        )
        # A grounded Link fact carries its original attributes and a canonical
        # causal neighbourhood. Entity identifiers remain symbolic arguments.
        model_config["raw_feature_dim"] = raw_feature_dim
        model_config["edge_feat_dim"] = lifter_edge_feature_dim(
            fact_context_len, raw_feature_dim
        )
    if spec.domain == "ctdg" and model_config.get("aggregation_operator") in {
        "multiscale_kernel",
        "multiscale_pool",
    }:
        model_config.setdefault("recurrence_history_len", int(config["training"]["history_len"]))
    attention = config.get("attention", {})
    model_config.setdefault("dropout", config.get("training", {}).get("dropout", 0.1))
    model_config.setdefault("gate_init", float(attention.get("gate_init", 0.95)))
    model_config.setdefault("gate_leak", float(attention.get("gate_leak", 0.05)))
    if spec.implementation == "official":
        if device.type == "cuda":
            model_config.setdefault("device", f"cuda:{0 if device.index is None else device.index}")
            model_config.setdefault("gpu", 0 if device.index is None else device.index)
        else:
            model_config.setdefault("device", "cpu")
            model_config.setdefault("gpu", -1)
        if spec.domain == "ctdg":
            model_config.update(
                build_dyglib_inputs(
                    frame,
                    node_feat_dim=int(model_config.get("hidden_dim", 128)),
                    edge_feat_dim=int(model_config.get("edge_feat_dim", 1)),
                    seed=spec.seed,
                )
            )
    model = build_model(
        spec.model,
        implementation=spec.implementation,
        use_attention=spec.use_attention,
        **model_config,
    )
    return model.to(device)


def audit_attention_application(model: torch.nn.Module, spec: RunSpec) -> dict[str, Any]:
    active_wrappers = (
        AttentionedDyGLibMultiHeadAttention,
        AttentionedEasyTPPMultiHeadAttention,
        AttentionedNeuralSTPPMultiheadAttention,
        AttentionedTorchMultiheadAttention,
    )
    active_blocks = 0
    for module in model.modules():
        if isinstance(module, AttentionedAttention) and module.use_attention:
            active_blocks += 1
        elif isinstance(module, EdgeConditionedAttentionBlock):
            active_blocks += 1
        elif bool(getattr(module, "_attention_audit_active", False)):
            active_blocks += 1
        elif isinstance(module, active_wrappers):
            active_blocks += 1
        elif hasattr(module, "_attention_block"):
            active_blocks += 1

    replacements = int(getattr(model, "attention_replacements", 0))
    effective = getattr(model, "official_name", None)
    if effective is None and hasattr(model, "model"):
        effective = getattr(getattr(model, "model"), "official_name", None)
    effective = effective or model.__class__.__name__

    attention_expected = bool(getattr(model, "attention_expected", spec.use_attention))
    if spec.use_attention and attention_expected and active_blocks == 0:
        raise RuntimeError(
            f"use_attention=True for {spec.model}, but no active AttentionBlock-backed attention was found. "
            "This run would produce a fake attention result."
        )

    return {
        "effective_implementation": effective,
        "attention_active_blocks": float(active_blocks),
        "attention_replacements": float(replacements),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
    }


def binary_auc(labels: list[float], scores: list[float]) -> float:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if not all(math.isfinite(float(score)) for score in scores):
        raise ValueError("AUC received a non-finite prediction score")
    pairs = sorted((float(score), float(label) > 0.5) for score, label in zip(scores, labels))
    positive_count = sum(int(label) for _, label in pairs)
    negative_count = len(pairs) - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    wins = 0.0
    negatives_before = 0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        group_positives = 0
        group_negatives = 0
        while index < len(pairs) and pairs[index][0] == score:
            if pairs[index][1]:
                group_positives += 1
            else:
                group_negatives += 1
            index += 1
        wins += group_positives * negatives_before
        wins += 0.5 * group_positives * group_negatives
        negatives_before += group_negatives
    return wins / (positive_count * negative_count)


def binary_average_precision(labels: list[float], scores: list[float]) -> float:
    """Average precision for binary labels, with equal scores handled as one threshold."""

    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if not labels:
        return float("nan")
    pairs = sorted(
        ((float(score), 1 if float(label) > 0.5 else 0) for label, score in zip(labels, scores)),
        key=lambda pair: pair[0],
        reverse=True,
    )
    total_positives = sum(label for _, label in pairs)
    if total_positives == 0:
        return float("nan")

    true_positives = 0
    false_positives = 0
    previous_recall = 0.0
    average_precision = 0.0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        group_positives = 0
        group_size = 0
        while index < len(pairs) and pairs[index][0] == score:
            group_positives += pairs[index][1]
            group_size += 1
            index += 1
        true_positives += group_positives
        false_positives += group_size - group_positives
        recall = true_positives / total_positives
        precision = true_positives / (true_positives + false_positives)
        average_precision += (recall - previous_recall) * precision
        previous_recall = recall
    return average_precision


def build_ctdg_node_histories(
    frame: pd.DataFrame,
    history_len: int,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build source-node temporal histories available strictly before each event."""

    nodes = np.zeros((len(frame), history_len), dtype=np.int64)
    times = np.zeros((len(frame), history_len), dtype=np.float32)
    masks = np.zeros((len(frame), history_len), dtype=bool)
    memories: list[deque[tuple[int, float]]] = [deque(maxlen=history_len) for _ in range(num_nodes)]

    src_values = frame["src"].to_numpy(dtype=np.int64)
    dst_values = frame["dst"].to_numpy(dtype=np.int64)
    ts_values = frame["timestamp"].to_numpy(dtype=np.float32)
    for idx, (src, dst, ts) in enumerate(zip(src_values, dst_values, ts_values)):
        history = list(memories[int(src)])
        if history:
            width = min(history_len, len(history))
            selected = history[-width:]
            start = history_len - width
            nodes[idx, start:] = [item[0] for item in selected]
            times[idx, start:] = [item[1] for item in selected]
            masks[idx, start:] = True
        memories[int(src)].append((int(dst), float(ts)))
        if src != dst:
            memories[int(dst)].append((int(src), float(ts)))
    return nodes, times, masks


class CTDGHistoryIndex:
    def __init__(self, frame: pd.DataFrame, history_len: int, num_nodes: int) -> None:
        self.history_len = int(history_len)
        self.node_rows: list[np.ndarray] = []
        self.node_neighbors: list[np.ndarray] = []
        self.node_times: list[np.ndarray] = []
        buckets: list[list[tuple[int, int, float]]] = [[] for _ in range(num_nodes)]
        src_values = frame["src"].to_numpy(dtype=np.int64)
        dst_values = frame["dst"].to_numpy(dtype=np.int64)
        ts_values = frame["timestamp"].to_numpy(dtype=np.float32)
        for row, (src, dst, ts) in enumerate(zip(src_values, dst_values, ts_values)):
            buckets[int(src)].append((row, int(dst), float(ts)))
            if src != dst:
                buckets[int(dst)].append((row, int(src), float(ts)))
        for bucket in buckets:
            bucket.sort(key=lambda item: item[0])
            self.node_rows.append(np.asarray([item[0] for item in bucket], dtype=np.int64))
            self.node_neighbors.append(np.asarray([item[1] for item in bucket], dtype=np.int64))
            self.node_times.append(np.asarray([item[2] for item in bucket], dtype=np.float32))

    def gather(self, node_ids: np.ndarray, row_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        nodes = np.zeros((len(node_ids), self.history_len), dtype=np.int64)
        times = np.zeros((len(node_ids), self.history_len), dtype=np.float32)
        masks = np.zeros((len(node_ids), self.history_len), dtype=bool)
        for idx, (node_id, row_id) in enumerate(zip(node_ids, row_ids)):
            if node_id < 0 or node_id >= len(self.node_rows):
                continue
            rows = self.node_rows[int(node_id)]
            end = int(np.searchsorted(rows, int(row_id), side="left"))
            if end <= 0:
                continue
            start = max(0, end - self.history_len)
            width = end - start
            offset = self.history_len - width
            nodes[idx, offset:] = self.node_neighbors[int(node_id)][start:end]
            times[idx, offset:] = self.node_times[int(node_id)][start:end]
            masks[idx, offset:] = True
        return nodes, times, masks


def build_lifter_causal_fact_context(
    frame: pd.DataFrame,
    fact_context_len: int = DEFAULT_FACT_CONTEXT_LEN,
) -> np.ndarray:
    """Canonical raw fact neighborhoods with strict temporal causality.

    For each ``f_i=Link(u_i,v_i,t_i)``, a token for an earlier incident fact
    ``f_j=Link(a_j,b_j,t_j)`` is

    ``[a_j=u_i, b_j=u_i, a_j=v_i, b_j=v_i, log(1+t_i-t_j), valid]``.

    These are grammar-level equality and time observations, not named pattern
    statistics.  The tensor is computed once per event and shared unchanged
    when that fact is later retrieved from either endpoint.
    """

    fact_context_len = int(fact_context_len)
    if fact_context_len < 1:
        raise ValueError("fact_context_len must be positive")
    num_nodes = int(max(frame["src"].max(), frame["dst"].max())) + 1
    context = np.zeros(
        (len(frame), fact_context_len, LIFTER_CONTEXT_TOKEN_DIM),
        dtype=np.float32,
    )
    src_values = frame["src"].to_numpy(dtype=np.int64)
    dst_values = frame["dst"].to_numpy(dtype=np.int64)
    ts_values = frame["timestamp"].to_numpy(dtype=np.float64)
    # Only the latest ``fact_context_len`` rows of either endpoint can enter
    # the latest ``fact_context_len`` rows of their union.  Bounding these
    # queues keeps construction O(N * L log L), instead of repeatedly copying
    # a high-degree node's complete history.
    incident_rows = [deque(maxlen=fact_context_len) for _ in range(num_nodes)]

    start = 0
    while start < len(frame):
        timestamp = float(ts_values[start])
        end = start + 1
        while end < len(frame) and float(ts_values[end]) == timestamp:
            end += 1

        # The incident index is frozen across this equal-time group, so only
        # facts satisfying t_j < t_i can appear in a token set.
        for row in range(start, end):
            source = int(src_values[row])
            destination = int(dst_values[row])
            prior_rows = set(incident_rows[source])
            prior_rows.update(incident_rows[destination])
            selected = sorted(
                prior_rows, key=lambda index: (float(ts_values[index]), index)
            )[-fact_context_len:]
            offset = fact_context_len - len(selected)
            for token_index, prior_row in enumerate(selected, start=offset):
                prior_source = int(src_values[prior_row])
                prior_destination = int(dst_values[prior_row])
                context[row, token_index] = np.asarray(
                    [
                        prior_source == source,
                        prior_destination == source,
                        prior_source == destination,
                        prior_destination == destination,
                        np.log1p(max(0.0, timestamp - float(ts_values[prior_row]))),
                        1.0,
                    ],
                    dtype=np.float32,
                )

        for row in range(start, end):
            source = int(src_values[row])
            destination = int(dst_values[row])
            incident_rows[source].append(row)
            if source != destination:
                incident_rows[destination].append(row)
        start = end
    return context


class LiFTERHistoryIndex:
    """Grounded fact index preserving Link arguments and original direction."""

    def __init__(
        self,
        frame: pd.DataFrame,
        history_len: int,
        num_nodes: int,
        fact_context: np.ndarray,
        raw_features: np.ndarray | None = None,
    ) -> None:
        self.history_len = int(history_len)
        if fact_context.ndim != 3 or fact_context.shape[-1] != LIFTER_CONTEXT_TOKEN_DIM:
            raise ValueError("invalid LiFTER fact-context tensor")
        self.fact_context_len = int(fact_context.shape[1])
        if raw_features is None:
            raw_features = np.empty((len(frame), 0), dtype=np.float32)
        raw_features = np.asarray(raw_features, dtype=np.float32)
        if raw_features.ndim != 2 or raw_features.shape[0] != len(frame):
            raise ValueError("invalid LiFTER raw fact-feature tensor")
        self.raw_feature_dim = int(raw_features.shape[1])
        self.edge_feature_dim = lifter_edge_feature_dim(
            self.fact_context_len, self.raw_feature_dim
        )
        self.node_rows: list[np.ndarray] = []
        self.node_neighbors: list[np.ndarray] = []
        self.node_times: list[np.ndarray] = []
        self.node_directions: list[np.ndarray] = []
        self.node_raw_features: list[np.ndarray] = []
        self.node_context: list[np.ndarray] = []
        buckets: list[list[tuple[int, int, float, float]]] = [
            [] for _ in range(num_nodes)
        ]
        src_values = frame["src"].to_numpy(dtype=np.int64)
        dst_values = frame["dst"].to_numpy(dtype=np.int64)
        ts_values = frame["timestamp"].to_numpy(dtype=np.float32)
        self.fact_sources = src_values
        self.fact_destinations = dst_values
        self.fact_times = ts_values
        self.fact_raw_features = raw_features
        self.fact_context = fact_context
        outgoing: list[list[int]] = [[] for _ in range(num_nodes)]
        incoming: list[list[int]] = [[] for _ in range(num_nodes)]
        for row, (source, destination, timestamp) in enumerate(
            zip(src_values, dst_values, ts_values)
        ):
            outgoing[int(source)].append(row)
            incoming[int(destination)].append(row)
            # +1: centre is Link arg1; -1: centre is Link arg2.
            buckets[int(source)].append((row, int(destination), float(timestamp), 1.0))
            if source != destination:
                buckets[int(destination)].append((row, int(source), float(timestamp), -1.0))
        self.outgoing_rows = [np.asarray(rows, dtype=np.int64) for rows in outgoing]
        self.incoming_rows = [np.asarray(rows, dtype=np.int64) for rows in incoming]
        for bucket in buckets:
            bucket.sort(key=lambda item: item[0])
            rows = np.asarray([item[0] for item in bucket], dtype=np.int64)
            self.node_rows.append(rows)
            self.node_neighbors.append(
                np.asarray([item[1] for item in bucket], dtype=np.int64)
            )
            self.node_times.append(
                np.asarray([item[2] for item in bucket], dtype=np.float32)
            )
            self.node_directions.append(
                np.asarray([item[3] for item in bucket], dtype=np.float32)
            )
            self.node_raw_features.append(
                raw_features[rows]
                if len(rows)
                else np.empty((0, self.raw_feature_dim), dtype=np.float32)
            )
            self.node_context.append(
                fact_context[rows]
                if len(rows)
                else np.empty(
                    (0, self.fact_context_len, LIFTER_CONTEXT_TOKEN_DIM),
                    dtype=np.float32,
                )
            )

    def gather(
        self,
        node_ids: np.ndarray,
        row_ids: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        nodes = np.zeros((len(node_ids), self.history_len), dtype=np.int64)
        times = np.zeros((len(node_ids), self.history_len), dtype=np.float32)
        features = np.zeros(
            (len(node_ids), self.history_len, self.edge_feature_dim),
            dtype=np.float32,
        )
        masks = np.zeros((len(node_ids), self.history_len), dtype=bool)
        for idx, (node_id, row_id) in enumerate(zip(node_ids, row_ids)):
            if node_id < 0 or node_id >= len(self.node_rows):
                continue
            rows = self.node_rows[int(node_id)]
            end = int(np.searchsorted(rows, int(row_id), side="left"))
            if end <= 0:
                continue
            start = max(0, end - self.history_len)
            width = end - start
            offset = self.history_len - width
            nodes[idx, offset:] = self.node_neighbors[int(node_id)][start:end]
            times[idx, offset:] = self.node_times[int(node_id)][start:end]
            features[idx, offset:, 0] = self.node_directions[int(node_id)][start:end]
            raw_end = 1 + self.raw_feature_dim
            features[idx, offset:, 1:raw_end] = self.node_raw_features[int(node_id)][
                start:end
            ]
            features[idx, offset:, raw_end:] = self.node_context[int(node_id)][
                start:end
            ].reshape(width, -1)
            masks[idx, offset:] = True
        return nodes, times, features, masks

    def gather_three_hop_paths(
        self,
        source_ids: np.ndarray,
        destination_ids: np.ndarray,
        row_ids: np.ndarray,
        max_paths: int,
        adjacent: bool = False,
        bidirectional_adjacent: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Gather causal X-A, B-A, B-Y paths for a Link(X,Y,Tq) query."""

        max_paths = int(max_paths)
        sources = np.zeros((len(row_ids), max_paths, 3), dtype=np.int64)
        destinations = np.zeros_like(sources)
        times = np.zeros((len(row_ids), max_paths, 3), dtype=np.float32)
        features = np.zeros(
            (len(row_ids), max_paths, 3, self.edge_feature_dim), dtype=np.float32
        )
        masks = np.zeros((len(row_ids), max_paths), dtype=bool)
        endpoint_width = min(self.history_len, max(8, max_paths))
        for batch_row, (query_source, query_destination, row_id) in enumerate(
            zip(source_ids, destination_ids, row_ids)
        ):
            if not (0 <= query_source < len(self.outgoing_rows)) or not (
                0 <= query_destination < len(self.incoming_rows)
            ):
                continue
            left = self.outgoing_rows[int(query_source)]
            left = left[left < int(row_id)][-(1 if adjacent else endpoint_width):]
            right = self.incoming_rows[int(query_destination)]
            right = right[right < int(row_id)][-endpoint_width:]
            left_by_destination: dict[int, int] = {}
            for fact_row in left:
                intermediate_destination = int(self.fact_destinations[fact_row])
                if intermediate_destination != int(query_destination):
                    left_by_destination[intermediate_destination] = int(fact_row)
            candidates: list[tuple[float, int, int, int]] = []
            for right_row in right:
                middle_source = int(self.fact_sources[right_row])
                if middle_source == int(query_source):
                    continue
                middle_rows = self.outgoing_rows[middle_source]
                if adjacent:
                    causal_middle = middle_rows[middle_rows < int(row_id)]
                    position = int(
                        np.searchsorted(causal_middle, int(right_row), side="left")
                    )
                    selected_positions = [position - 1]
                    if bidirectional_adjacent:
                        # Skip the right fact itself and take the next distinct
                        # interaction by the same source when it exists.
                        next_position = position
                        while (
                            next_position < len(causal_middle)
                            and int(causal_middle[next_position]) == int(right_row)
                        ):
                            next_position += 1
                        selected_positions.append(next_position)
                    middle_rows = np.asarray(
                        [
                            causal_middle[selected]
                            for selected in selected_positions
                            if 0 <= selected < len(causal_middle)
                        ],
                        dtype=np.int64,
                    )
                else:
                    middle_rows = middle_rows[middle_rows < int(row_id)][-self.history_len :]
                for middle_row in middle_rows:
                    left_row = left_by_destination.get(
                        int(self.fact_destinations[middle_row])
                    )
                    if left_row is None:
                        continue
                    latest = max(
                        float(self.fact_times[left_row]),
                        float(self.fact_times[middle_row]),
                        float(self.fact_times[right_row]),
                    )
                    candidates.append(
                        (latest, left_row, int(middle_row), int(right_row))
                    )
            candidates.sort(key=lambda item: item[0])
            for path_index, (_, left_row, middle_row, right_row) in enumerate(
                candidates[-max_paths:]
            ):
                for slot, fact_row in enumerate((left_row, middle_row, right_row)):
                    sources[batch_row, path_index, slot] = self.fact_sources[fact_row]
                    destinations[batch_row, path_index, slot] = self.fact_destinations[fact_row]
                    times[batch_row, path_index, slot] = self.fact_times[fact_row]
                    features[batch_row, path_index, slot, 0] = 1.0
                    raw_end = 1 + self.raw_feature_dim
                    features[batch_row, path_index, slot, 1:raw_end] = self.fact_raw_features[fact_row]
                    features[batch_row, path_index, slot, raw_end:] = self.fact_context[fact_row].reshape(-1)
                masks[batch_row, path_index] = True
        return sources, destinations, times, features, masks


def sample_historical_destinations(
    source_values: np.ndarray,
    positive_destinations: np.ndarray,
    history_nodes: np.ndarray,
    history_mask: np.ndarray,
    destination_pool: np.ndarray,
    strategy: str = "uniform",
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a prior destination per source, with random fallback.

    ``uniform`` is the historical-negative benchmark: it avoids turning the
    task into a deterministic test against the immediately preceding event.
    ``most_recent`` is retained as a deliberately adversarial recency stress
    test and is reported separately.
    """

    if strategy not in {"uniform", "most_recent"}:
        raise ValueError("historical strategy must be uniform or most_recent")

    sampled = np.empty_like(positive_destinations, dtype=np.int64)
    historical = np.zeros(len(sampled), dtype=bool)
    valid_destinations = set(np.asarray(destination_pool, dtype=np.int64).tolist())
    for index, (source, positive) in enumerate(
        zip(source_values, positive_destinations)
    ):
        candidates = history_nodes[index][history_mask[index]]
        candidates = np.asarray(
            [
                int(value) for value in candidates
                if int(value) in valid_destinations
                and int(value) != int(positive)
                and int(value) != int(source)
            ],
            dtype=np.int64,
        )
        chosen: int | None = None
        if len(candidates):
            if strategy == "most_recent":
                chosen = int(candidates[-1])
            else:
                chosen = int(np.random.choice(np.unique(candidates)))
            historical[index] = True
        if chosen is None:
            choices = destination_pool[
                (destination_pool != int(positive)) & (destination_pool != int(source))
            ]
            if len(choices) == 0:
                raise ValueError("no valid negative destination")
            chosen = int(np.random.choice(choices))
        sampled[index] = chosen
    return sampled, historical


def build_grounded_explanation_universe(
    inputs: tuple[torch.Tensor, ...],
    occurrence_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Merge endpoint and path occurrences that denote the same Link fact.

    Returns ``scores``, ``valid``, endpoint-occurrence ids, and path-slot ids.
    The latter two maps let explanation perturbations alter every executor
    view of a selected grounded fact consistently.
    """
    batch, src_width = inputs[3].shape[:2]
    dst_width = inputs[7].shape[1]
    path_count = inputs[11].shape[1] if len(inputs) > 15 else 0
    endpoint_ids = torch.full(
        (batch, src_width + dst_width), -1,
        dtype=torch.long, device=occurrence_scores.device,
    )
    path_ids = torch.full(
        (batch, path_count, 3), -1,
        dtype=torch.long, device=occurrence_scores.device,
    )
    row_scores: list[torch.Tensor] = []
    for row in range(batch):
        lookup: dict[tuple[int, int, float], int] = {}
        values: list[torch.Tensor] = []

        def register(key: tuple[int, int, float], score: torch.Tensor) -> int:
            fact_id = lookup.get(key)
            if fact_id is None:
                fact_id = len(values)
                lookup[key] = fact_id
                values.append(score)
            else:
                values[fact_id] = values[fact_id] + score
            return fact_id

        occurrence = 0
        for bank, (centre, nodes, times, features, valid) in enumerate((
            (inputs[0], inputs[3], inputs[4], inputs[5], inputs[6]),
            (inputs[1], inputs[7], inputs[8], inputs[9], inputs[10]),
        )):
            width = nodes.shape[1]
            for index in range(width):
                if bool(valid[row, index]):
                    neighbour = int(nodes[row, index])
                    centre_id = int(centre[row])
                    outgoing = float(features[row, index, 0]) >= 0
                    source, destination = (
                        (centre_id, neighbour)
                        if outgoing else (neighbour, centre_id)
                    )
                    fact_id = register(
                        (source, destination, float(times[row, index])),
                        occurrence_scores[row, occurrence + index],
                    )
                    endpoint_ids[row, occurrence + index] = fact_id
            occurrence += width
        if path_count:
            path_offset = src_width + dst_width
            for path in range(path_count):
                if not bool(inputs[15][row, path]):
                    continue
                for slot in range(3):
                    fact_id = register(
                        (
                            int(inputs[11][row, path, slot]),
                            int(inputs[12][row, path, slot]),
                            float(inputs[13][row, path, slot]),
                        ),
                        occurrence_scores[row, path_offset + path * 3 + slot],
                    )
                    path_ids[row, path, slot] = fact_id
        row_scores.append(
            torch.stack(values) if values else occurrence_scores.new_zeros(0)
        )
    width = max((len(values) for values in row_scores), default=0)
    scores = occurrence_scores.new_zeros((batch, width))
    valid = torch.zeros(
        (batch, width), dtype=torch.bool, device=occurrence_scores.device
    )
    for row, values in enumerate(row_scores):
        scores[row, : len(values)] = values
        valid[row, : len(values)] = True
    return scores, valid, endpoint_ids, path_ids


@torch.no_grad()
def counterfactual_universe_responsibility(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    base_scores: torch.Tensor,
    endpoint_ids: torch.Tensor,
    path_ids: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Score each unique grounded fact by deleting all of its occurrences."""
    fact_ids = torch.nonzero(valid[0], as_tuple=False).flatten()
    if fact_ids.numel() == 0:
        return base_scores
    count = int(fact_ids.numel())
    expanded = tuple(
        value.repeat(count + 1, *([1] * (value.ndim - 1))) for value in inputs
    )
    modified = list(expanded)
    endpoint_map = endpoint_ids[0]
    src_width = inputs[3].shape[1]
    for offset, fact_id in enumerate(fact_ids.tolist(), start=1):
        occurrences = endpoint_map == fact_id
        modified[6][offset] &= ~occurrences[:src_width]
        modified[10][offset] &= ~occurrences[src_width:]
        if len(modified) > 15:
            affected_paths = (path_ids[0] == fact_id).any(-1)
            modified[15][offset] &= ~affected_paths
    probabilities = model(*tuple(modified)).logits.sigmoid()
    delta = (probabilities[0] - probabilities[1:]).abs()
    counterfactual = torch.zeros_like(base_scores)
    counterfactual[0, fact_ids] = delta
    base_scale = base_scores.amax(1, keepdim=True).clamp_min(1e-12)
    delta_scale = counterfactual.amax(1, keepdim=True).clamp_min(1e-12)
    # Singleton deletion is the responsibility definition.  The exact trace
    # is used only as a deterministic tie-break when two deletions have the
    # same effect; there is no fitted or dataset-specific mixing weight.
    return counterfactual / delta_scale + 1e-6 * base_scores / base_scale


class LengthThreeCandidateIndex:
    """Mine candidate destinations with the same X-A<-B-Y proof support.

    The index is model-independent and uses only facts strictly preceding the
    query row.  It supports a hard evaluation where both the observed
    destination and its negative have at least one connected length-three
    grounding, preventing proof existence itself from revealing the label.
    """

    def __init__(self, frame: pd.DataFrame, endpoint_width: int = 128) -> None:
        self.src = frame["src"].to_numpy(dtype=np.int64)
        self.dst = frame["dst"].to_numpy(dtype=np.int64)
        self.times = frame["timestamp"].to_numpy(dtype=np.float64)
        node_count = int(max(self.src.max(), self.dst.max())) + 1
        outgoing: list[list[int]] = [[] for _ in range(node_count)]
        incoming: list[list[int]] = [[] for _ in range(node_count)]
        for row, (source, destination) in enumerate(zip(self.src, self.dst)):
            outgoing[int(source)].append(row)
            incoming[int(destination)].append(row)
        self.outgoing = [np.asarray(rows, dtype=np.int64) for rows in outgoing]
        self.incoming = [np.asarray(rows, dtype=np.int64) for rows in incoming]
        self.endpoint_width = int(endpoint_width)

    def _prior_outgoing(self, node: int, row: int) -> np.ndarray:
        if node < 0 or node >= len(self.outgoing):
            return np.empty(0, dtype=np.int64)
        rows = self.outgoing[node]
        end = int(np.searchsorted(rows, row, side="left"))
        return rows[max(0, end - self.endpoint_width):end]

    def support(self, source: int, row: int) -> dict[int, tuple[int, float]]:
        left = self._prior_outgoing(source, row)
        if not len(left):
            return {}
        shared_destination = int(self.dst[left[-1]])
        left_time = float(self.times[left[-1]])
        counts: dict[int, tuple[int, float]] = {}
        # A valid adjacent proof is X->A and consecutive B->A, B->Y facts.
        incoming = self.incoming[shared_destination]
        incoming = incoming[incoming < row][-self.endpoint_width:]
        middle_sources = np.unique(self.src[incoming])
        for middle_source in middle_sources:
            if middle_source == source:
                continue
            rows = self.outgoing[int(middle_source)]
            end = int(np.searchsorted(rows, row, side="left"))
            recent = rows[max(0, end - self.endpoint_width):end]
            for prior_row, right_row in zip(recent[:-1], recent[1:]):
                if int(self.dst[prior_row]) != shared_destination:
                    continue
                if not (
                    float(self.times[prior_row])
                    < float(self.times[right_row])
                    < left_time
                ):
                    continue
                candidate = int(self.dst[right_row])
                if candidate == shared_destination:
                    continue
                count, latest = counts.get(candidate, (0, -float("inf")))
                counts[candidate] = (
                    count + 1,
                    max(latest, float(self.times[right_row])),
                )
        return counts

    def _recurs(self, source: int, destination: int, row: int) -> bool:
        return bool(np.any(self.dst[self._prior_outgoing(source, row)] == destination))

    def _degree(self, destination: int, row: int) -> int:
        if destination < 0 or destination >= len(self.incoming):
            return 0
        return int(np.searchsorted(self.incoming[destination], row, side="left"))

    def sample(self, sources: np.ndarray, positives: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        sampled = positives.astype(np.int64).copy()
        selected = np.zeros(len(rows), dtype=bool)
        positive_counts = np.zeros(len(rows), dtype=np.int64)
        negative_counts = np.zeros(len(rows), dtype=np.int64)
        positive_ages = np.zeros(len(rows), dtype=np.float64)
        negative_ages = np.zeros(len(rows), dtype=np.float64)
        positive_degrees = np.zeros(len(rows), dtype=np.int64)
        negative_degrees = np.zeros(len(rows), dtype=np.int64)
        for index, (source, positive, row) in enumerate(zip(sources, positives, rows)):
            support = self.support(int(source), int(row))
            positive_count, positive_latest = support.get(int(positive), (0, 0.0))
            positive_counts[index] = positive_count
            if positive_count <= 0:
                continue
            query_time = float(self.times[int(row)])
            positive_age = max(0.0, query_time - positive_latest)
            positive_degree = self._degree(int(positive), int(row))
            positive_recurrence = self._recurs(int(source), int(positive), int(row))
            positive_ages[index] = positive_age
            positive_degrees[index] = positive_degree
            candidates = [
                (
                    abs(np.log1p(count) - np.log1p(positive_count))
                    + abs(np.log1p(max(0.0, query_time - latest)) - np.log1p(positive_age))
                    + abs(np.log1p(self._degree(destination, int(row))) - np.log1p(positive_degree)),
                    destination,
                    count,
                    max(0.0, query_time - latest),
                    self._degree(destination, int(row)),
                )
                for destination, (count, latest) in support.items()
                if destination not in (int(positive), int(source))
                and count > 0
                and self._recurs(int(source), destination, int(row)) == positive_recurrence
                and abs(np.log1p(count) - np.log1p(positive_count)) <= np.log(2.0)
                and abs(
                    np.log1p(max(0.0, query_time - latest)) - np.log1p(positive_age)
                ) <= np.log(2.0)
                and abs(
                    np.log1p(self._degree(destination, int(row))) - np.log1p(positive_degree)
                ) <= np.log(2.0)
            ]
            if not candidates:
                continue
            _, destination, count, age, degree = min(candidates)
            sampled[index] = destination
            negative_counts[index] = count
            negative_ages[index] = age
            negative_degrees[index] = degree
            selected[index] = True
        return sampled, selected, {
            "positive_count": positive_counts,
            "negative_count": negative_counts,
            "positive_age": positive_ages,
            "negative_age": negative_ages,
            "positive_degree": positive_degrees,
            "negative_degree": negative_degrees,
        }


def lifter_mechanism_logits(model: torch.nn.Module, result: Any) -> dict[str, torch.Tensor]:
    """Re-score a LiFTER result by masking executable rule families."""

    contributions = result.aux["rule_scores"]
    device = contributions.device
    lengths = torch.tensor([rule.length for rule in model.program], device=device)
    left_roles = model.left_roles.to(device)
    direct = (left_roles == 0) | (left_roles == 1)
    renewal = model.renewal_rules.to(device)
    recurrence_only = ((lengths == 1) & direct) | renewal
    no_direct = ~direct
    length_three = lengths == 3
    direct_unary = (lengths == 1) & direct
    prior = result.aux["prior_rule_contribution"]
    source_context = (lengths == 1) & ((left_roles == 2) | (left_roles == 3))
    candidate_context = (lengths == 1) & ((left_roles == 4) | (left_roles == 5))
    covered = direct_unary | source_context | candidate_context | renewal
    components = {
        "direct_pair": contributions[:, direct_unary].sum(-1),
        "source_context": contributions[:, source_context].sum(-1),
        "candidate_context": contributions[:, candidate_context].sum(-1),
        "pair_renewal": contributions[:, renewal].sum(-1),
        "positioned_recurrence": result.aux[
            "positioned_recurrence_rule_contribution"
        ],
        "ordered_transition_1": result.aux["transition_rule_contribution"],
        "ordered_transition_2": result.aux["transition2_rule_contribution"],
        "other_program_rules": contributions[:, ~covered].sum(-1),
    }
    reconstructed = prior + torch.stack(tuple(components.values()), dim=-1).sum(-1)
    if not torch.allclose(reconstructed, result.logits, atol=1e-5, rtol=1e-5):
        error = float((reconstructed - result.logits).abs().max().detach().cpu())
        raise RuntimeError(
            f"LiFTER mechanism decomposition does not reconstruct logits: {error}"
        )
    output = {
        "full": result.logits,
        "prior_only": prior,
        "recurrence_only": prior + contributions[:, recurrence_only].sum(-1),
        "no_direct_recurrence": prior + contributions[:, no_direct].sum(-1),
        "length_three_only": prior + contributions[:, length_three].sum(-1),
        "renewal_plus_length_three": prior
        + contributions[:, renewal | length_three].sum(-1),
        "direct_plus_length_three": prior
        + contributions[:, direct_unary | length_three].sum(-1),
        "recurrence_plus_length_three": prior
        + contributions[:, recurrence_only | length_three].sum(-1),
    }
    for name, component in components.items():
        output[f"only_{name}"] = prior + component
        output[f"without_{name}"] = result.logits - component
    return output


def lifter_component_fact_indices(
    model: torch.nn.Module, result: Any, component: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the concrete fact carrying the largest absolute component execution.

    Bank 0 is the source history and bank 1 is the candidate history.  The
    returned fact is an executor grounding, not an attribution surrogate.
    """

    contributions = result.aux["rule_scores"]
    device = contributions.device
    lengths = torch.tensor([rule.length for rule in model.program], device=device)
    roles = model.left_roles.to(device)
    renewal = model.renewal_rules.to(device)
    if component == "positioned_recurrence":
        values = result.aux["positioned_recurrence_fact_contributions"].abs()
        index = values.argmax(-1)
        valid = values.gather(1, index[:, None]).squeeze(1) > 0
        return torch.zeros_like(index), index, valid
    if component == "ordered_transition_1":
        index = result.aux["transition_fact_index"].long()
        return (
            torch.zeros_like(index), index,
            result.aux["transition_grounding_valid"].bool()
            & (result.aux["transition_rule_contribution"].abs() > 0),
        )
    if component == "ordered_transition_2":
        # The most recent of the two ordered states is the final body fact.
        index = result.aux["transition2_second_fact_index"].long()
        return (
            torch.zeros_like(index), index,
            result.aux["transition2_grounding_valid"].bool()
            & (result.aux["transition2_rule_contribution"].abs() > 0),
        )
    masks = {
        "direct_pair": (lengths == 1) & ((roles == 0) | (roles == 1)),
        "source_context": (lengths == 1) & ((roles == 2) | (roles == 3)),
        "candidate_context": (lengths == 1) & ((roles == 4) | (roles == 5)),
        "pair_renewal": renewal,
    }
    if component not in masks:
        raise ValueError(f"unsupported intervention component: {component}")
    rule_mask = masks[component]
    masked = contributions.abs().masked_fill(~rule_mask[None], -1.0)
    rule_id = masked.argmax(-1)
    rows = torch.arange(len(rule_id), device=device)
    valid = masked[rows, rule_id] > 0
    if component == "pair_renewal":
        index = result.aux["fact2_index_by_rule"][rows, rule_id].long()
        bank = torch.zeros_like(index)
    else:
        index = result.aux["fact1_index_by_rule"][rows, rule_id].long()
        bank = (roles[rule_id] >= 4).long()
    valid &= index >= 0
    return bank, index, valid


def delete_lifter_grounded_facts(
    batch: tuple[torch.Tensor, ...],
    bank: torch.Tensor,
    index: torch.Tensor,
    valid: torch.Tensor,
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], torch.Tensor]:
    """Delete a selected fact and a recency-matched random fact, including duplicates."""

    top = list(batch)
    random_batch = list(batch)
    top_src_mask = batch[6].bool().clone()
    top_dst_mask = batch[10].bool().clone()
    random_src_mask = top_src_mask.clone()
    random_dst_mask = top_dst_mask.clone()
    rng = np.random.default_rng(seed)
    eligible = valid.clone()

    def canonical(sample: int, selected_bank: int, selected_index: int):
        anchor_position = 0 if selected_bank == 0 else 1
        node_position = 3 if selected_bank == 0 else 7
        time_position = 4 if selected_bank == 0 else 8
        feature_position = 5 if selected_bank == 0 else 9
        anchor = batch[anchor_position][sample]
        node = batch[node_position][sample, selected_index]
        outgoing = batch[feature_position][sample, selected_index, 0] >= 0
        source = torch.where(outgoing, anchor, node)
        destination = torch.where(outgoing, node, anchor)
        return source, destination, batch[time_position][sample, selected_index]

    def remove_matching(
        sample: int, fact: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        source_mask: torch.Tensor, destination_mask: torch.Tensor,
    ) -> None:
        fact_source, fact_destination, fact_time = fact
        for selected_bank, mask in ((0, source_mask), (1, destination_mask)):
            anchor_position = 0 if selected_bank == 0 else 1
            node_position = 3 if selected_bank == 0 else 7
            time_position = 4 if selected_bank == 0 else 8
            feature_position = 5 if selected_bank == 0 else 9
            anchor = batch[anchor_position][sample]
            nodes = batch[node_position][sample]
            outgoing = batch[feature_position][sample, :, 0] >= 0
            sources = torch.where(outgoing, anchor, nodes)
            destinations = torch.where(outgoing, nodes, anchor)
            matches = (
                mask[sample]
                & (sources == fact_source)
                & (destinations == fact_destination)
                & torch.isclose(batch[time_position][sample], fact_time)
            )
            mask[sample] &= ~matches

    for sample in range(len(index)):
        selected_bank = int(bank[sample])
        selected_index = int(index[sample])
        selected_mask = batch[6 if selected_bank == 0 else 10][sample].bool()
        if (
            not bool(eligible[sample]) or selected_index < 0
            or selected_index >= len(selected_mask) or not bool(selected_mask[selected_index])
        ):
            eligible[sample] = False
            continue
        valid_indices = torch.nonzero(selected_mask, as_tuple=False).flatten().cpu().numpy()
        selected_rank = int(np.flatnonzero(valid_indices == selected_index)[0])
        quartile = min(3, int(4 * selected_rank / max(1, len(valid_indices))))
        alternatives = [
            int(candidate) for rank, candidate in enumerate(valid_indices)
            if candidate != selected_index
            and min(3, int(4 * rank / max(1, len(valid_indices)))) == quartile
        ]
        if not alternatives:
            alternatives = [int(candidate) for candidate in valid_indices if candidate != selected_index]
        if not alternatives:
            eligible[sample] = False
            continue
        random_index = alternatives[int(rng.integers(len(alternatives)))]
        remove_matching(
            sample, canonical(sample, selected_bank, selected_index),
            top_src_mask, top_dst_mask,
        )
        remove_matching(
            sample, canonical(sample, selected_bank, random_index),
            random_src_mask, random_dst_mask,
        )
    top[6], top[10] = top_src_mask, top_dst_mask
    random_batch[6], random_batch[10] = random_src_mask, random_dst_mask
    return tuple(top), tuple(random_batch), eligible


def delete_lifter_proofs(
    result: Any,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Delete the largest or a uniformly sampled nonzero rule contribution."""

    contributions = result.aux["rule_scores"]
    present = result.aux["rule_grounding_valid"].bool() & (contributions != 0)
    top_delta = torch.zeros(len(contributions), device=contributions.device)
    random_delta = torch.zeros_like(top_delta)
    for row in range(len(contributions)):
        ids = torch.nonzero(present[row], as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        top_id = ids[contributions[row, ids].abs().argmax()]
        random_id = ids[int(rng.integers(0, ids.numel()))]
        top_delta[row] = contributions[row, top_id]
        random_delta[row] = contributions[row, random_id]
    return (
        result.logits - top_delta,
        result.logits - random_delta,
        top_delta.abs(),
        random_delta.abs(),
    )


def run_ctdg(spec: RunSpec, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    training = config["training"]
    frame = read_csv_tail(spec.dataset_path, int(training["max_examples"]))
    frame["src"] = pd.to_numeric(frame["src"], errors="coerce").fillna(0).astype("int64")
    frame["dst"] = pd.to_numeric(frame["dst"], errors="coerce").fillna(0).astype("int64")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0.0).astype("float32")
    if spec.model == "LiFTER":
        frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    train_idx, eval_idx = split_indices(len(frame), float(training["eval_ratio"]))
    query_filter = training.get("query_filter")
    if query_filter:
        column = str(query_filter.get("column", ""))
        value = query_filter.get("value")
        if column not in frame.columns:
            raise ValueError(f"query_filter column does not exist: {column!r}")
        eligible = frame[column].astype(str).to_numpy() == str(value)
        train_idx = train_idx[eligible[train_idx]]
        eval_idx = eval_idx[eligible[eval_idx]]
        if len(train_idx) == 0 or len(eval_idx) == 0:
            raise ValueError(
                f"query_filter {column}={value!r} produced an empty train or eval split"
            )
    # Multi-fidelity pilots must share the full benchmark's chronological
    # boundary and evaluation queries. Restricting indices after the split
    # creates a nested training window instead of re-splitting a different
    # tail sample, which otherwise changes the task itself.
    train_max_examples = int(training.get("train_max_examples", 0))
    if train_max_examples > 0:
        train_sampling = str(
            training.get("train_sampling_strategy", "recent")
        )
        if train_sampling == "recent":
            train_idx = train_idx[-train_max_examples:]
        elif train_sampling == "uniform_horizon":
            if train_max_examples < len(train_idx):
                positions = np.linspace(
                    0,
                    len(train_idx) - 1,
                    train_max_examples,
                    dtype=np.int64,
                )
                train_idx = train_idx[positions]
        else:
            raise ValueError(
                "train_sampling_strategy must be recent or uniform_horizon"
            )
    eval_max_examples = int(training.get("eval_max_examples", 0))
    if eval_max_examples > 0:
        if eval_max_examples < len(eval_idx):
            # Cover the complete test horizon instead of evaluating only its
            # easiest/earliest prefix.  This makes a small pilot an unbiased
            # temporal proxy for the full benchmark while remaining fixed and
            # deterministic across models.
            positions = np.linspace(
                0, len(eval_idx) - 1, eval_max_examples, dtype=np.int64
            )
            eval_idx = eval_idx[positions]
    model = build_benchmark_model(spec, frame, config, device)
    # Parameter initialization may consume a model-dependent number of random
    # draws. Decouple that from minibatch/dropout randomness so architecture
    # ablations with the same seed follow the same stochastic training path.
    torch.manual_seed(spec.seed + 65537)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(spec.seed + 65537)
    load_checkpoint = config.get("paths", {}).get("load_checkpoint")
    if load_checkpoint:
        saved = torch.load(load_checkpoint, map_location=device, weights_only=False)
        state = saved.get("model", saved)
        model.load_state_dict(state)
    model_audit = audit_attention_application(model, spec)
    base_learning_rate = float(training["learning_rate"])
    weight_decay = float(training.get("weight_decay", 0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=base_learning_rate, weight_decay=weight_decay)
    history_len = int(training["history_len"])
    num_nodes = int(max(frame["src"].max(), frame["dst"].max())) + 1
    # Uncapped causal activity statistics for query-regime diagnostics.  These
    # are computed before each event and therefore do not inherit the model's
    # finite history-window saturation.
    causal_source_activity = np.zeros(len(frame), dtype=np.int64)
    causal_destination_popularity = np.zeros(len(frame), dtype=np.int64)
    causal_node_counts = np.zeros(num_nodes, dtype=np.int64)
    frame_sources = frame["src"].to_numpy(dtype=np.int64)
    frame_destinations = frame["dst"].to_numpy(dtype=np.int64)
    for causal_row, (causal_source, causal_destination) in enumerate(
        zip(frame_sources, frame_destinations)
    ):
        causal_source_activity[causal_row] = causal_node_counts[causal_source]
        causal_destination_popularity[causal_row] = causal_node_counts[causal_destination]
        causal_node_counts[causal_source] += 1
        causal_node_counts[causal_destination] += 1
    dst_pool = frame["dst"].drop_duplicates().to_numpy(dtype=np.int64)
    proof_matched_enabled = bool(
        config.get("proof_matched_evaluation", {}).get("enabled", False)
    )
    proof_candidate_index = (
        LengthThreeCandidateIndex(
            frame,
            int(config.get("proof_matched_evaluation", {}).get("endpoint_width", training["history_len"])),
        )
        if proof_matched_enabled
        else None
    )
    is_lifter = bool(getattr(model, "is_neuro_symbolic", False))
    lifter_diagnostics_enabled = is_lifter and bool(
        config.get("lifter_diagnostics", {}).get("enabled", True)
    )
    source_hist_features: np.ndarray | None = None
    if is_lifter:
        raw_columns = [
            column for column in frame.columns if str(column).startswith("feat_")
        ]
        raw_features = (
            frame[raw_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        fact_context = build_lifter_causal_fact_context(
            frame, int(model.fact_context_len)
        )
        history_index: CTDGHistoryIndex | LiFTERHistoryIndex = LiFTERHistoryIndex(
            frame, history_len, num_nodes, fact_context, raw_features
        )
        all_rows = np.arange(len(frame), dtype=np.int64)
        (
            source_hist_nodes,
            source_hist_times,
            source_hist_features,
            source_hist_mask,
        ) = history_index.gather(
            frame["src"].to_numpy(dtype=np.int64), all_rows
        )
    else:
        source_hist_nodes, source_hist_times, source_hist_mask = build_ctdg_node_histories(
            frame, history_len, num_nodes
        )
        history_index = CTDGHistoryIndex(frame, history_len, num_nodes)

    def make_batch(
        rows: np.ndarray,
        negative: bool | str,
        destination_override: np.ndarray | None = None,
    ):
        src_values = frame.iloc[rows]["src"].to_numpy(dtype=np.int64)
        src = torch.tensor(src_values, device=device)
        dst_values = frame.iloc[rows]["dst"].to_numpy().copy()
        historical_selection = np.zeros(len(rows), dtype=bool)
        if destination_override is not None:
            dst_values = np.asarray(destination_override, dtype=np.int64)
            if dst_values.shape != src_values.shape:
                raise ValueError("destination_override must match rows")
        elif negative == "historical":
            dst_values, historical_selection = sample_historical_destinations(
                src_values,
                dst_values.astype(np.int64),
                source_hist_nodes[rows],
                source_hist_mask[rows],
                dst_pool,
                strategy=str(training.get("historical_negative_strategy", "uniform")),
            )
        elif negative == "historical_recent":
            dst_values, historical_selection = sample_historical_destinations(
                src_values,
                dst_values.astype(np.int64),
                source_hist_nodes[rows],
                source_hist_mask[rows],
                dst_pool,
                strategy="most_recent",
            )
        elif negative == "proof_matched":
            if proof_candidate_index is None:
                raise ValueError("proof-matched evaluation is not enabled")
            dst_values, historical_selection, _ = proof_candidate_index.sample(
                src_values,
                dst_values.astype(np.int64),
                rows.astype(np.int64),
            )
        elif negative == "in_batch":
            pos_values = frame.iloc[rows]["dst"].to_numpy(dtype=np.int64)
            if len(pos_values) > 1:
                shift = int(np.random.randint(1, len(pos_values)))
                dst_values = np.roll(pos_values, shift)
            else:
                dst_values = np.random.choice(dst_pool, size=len(rows), replace=True)
            bad = (dst_values == src_values) | (dst_values == pos_values)
            while bad.any():
                dst_values[bad] = np.random.choice(dst_pool, size=int(bad.sum()), replace=True)
                bad = (dst_values == src_values) | (dst_values == pos_values)
        elif negative:
            pos_values = frame.iloc[rows]["dst"].to_numpy()
            dst_values = np.random.choice(dst_pool, size=len(rows), replace=True)
            bad = (dst_values == src_values) | (dst_values == pos_values)
            while bad.any():
                dst_values[bad] = np.random.choice(dst_pool, size=int(bad.sum()), replace=True)
                bad = (dst_values == src_values) | (dst_values == pos_values)
        dst = torch.tensor(dst_values, device=device)
        timestamp_values = frame.iloc[rows]["timestamp"].to_numpy(dtype=np.float32)
        ts = torch.tensor(timestamp_values, dtype=torch.float32, device=device)
        hist_nodes_t = torch.tensor(source_hist_nodes[rows], device=device)
        hist_times_t = torch.tensor(source_hist_times[rows], dtype=torch.float32, device=device)
        edge_feats = (
            torch.tensor(
                source_hist_features[rows], dtype=torch.float32, device=device
            )
            if source_hist_features is not None
            else torch.zeros(
                (len(rows), history_len, 1), dtype=torch.float32, device=device
            )
        )
        masks_t = torch.tensor(source_hist_mask[rows], dtype=torch.bool, device=device)
        gathered_destination = history_index.gather(
            dst_values.astype(np.int64), rows.astype(np.int64)
        )
        if is_lifter:
            (
                dst_hist_nodes,
                dst_hist_times,
                dst_hist_features,
                dst_hist_masks,
            ) = gathered_destination
        else:
            dst_hist_nodes, dst_hist_times, dst_hist_masks = gathered_destination
            dst_hist_features = None
        dst_hist_nodes_t = torch.tensor(dst_hist_nodes, device=device)
        dst_hist_times_t = torch.tensor(dst_hist_times, dtype=torch.float32, device=device)
        dst_edge_feats = (
            torch.tensor(dst_hist_features, dtype=torch.float32, device=device)
            if dst_hist_features is not None
            else torch.zeros(
                (len(rows), history_len, 1), dtype=torch.float32, device=device
            )
        )
        dst_masks_t = torch.tensor(dst_hist_masks, dtype=torch.bool, device=device)
        batch = (
            src,
            dst,
            ts,
            hist_nodes_t,
            hist_times_t,
            edge_feats,
            masks_t,
            dst_hist_nodes_t,
            dst_hist_times_t,
            dst_edge_feats,
            dst_masks_t,
        )
        if not is_lifter or int(getattr(model, "max_rule_length", 2)) < 3:
            return batch, historical_selection
        path_sources, path_destinations, path_times, path_features, path_masks = (
            history_index.gather_three_hop_paths(
                src_values,
                dst_values.astype(np.int64),
                rows.astype(np.int64),
                int(model.max_three_hop_paths),
                bool(getattr(model, "adjacent_three_hop_paths", False)),
                bool(getattr(model, "bidirectional_adjacent_paths", False)),
            )
        )
        corruption = str(
            config.get("evaluation_intervention", {}).get(
                "three_hop_corruption", "none"
            )
        )
        if not model.training and corruption != "none":
            valid_paths = np.argwhere(path_masks)
            if len(valid_paths) > 1 or (
                corruption == "break_binding" and len(valid_paths) > 0
            ):
                source_copy = path_sources.copy()
                destination_copy = path_destinations.copy()
                time_copy = path_times.copy()
                feature_copy = path_features.copy()
                rolled_once = np.roll(valid_paths, 1, axis=0)
                if corruption == "break_binding":
                    # Change only one concrete argument. Predicate features,
                    # timestamps, path count, and every other fact value stay
                    # fixed; the executor must reject the broken equality.
                    for target in valid_paths:
                        tr, tp = map(int, target)
                        path_destinations[tr, tp, 1] = (
                            source_copy[tr, tp, 1] + 1 + int(path_destinations.max())
                        )
                elif corruption == "middle_binding_shuffle":
                    for target, donor in zip(valid_paths, rolled_once):
                        tr, tp = map(int, target)
                        dr, dp = map(int, donor)
                        path_sources[tr, tp, 1] = source_copy[dr, dp, 1]
                        path_destinations[tr, tp, 1] = destination_copy[dr, dp, 1]
                        path_features[tr, tp, 1] = feature_copy[dr, dp, 1]
                elif corruption == "time_shuffle":
                    for target, donor in zip(valid_paths, rolled_once):
                        tr, tp = map(int, target)
                        dr, dp = map(int, donor)
                        path_times[tr, tp] = time_copy[dr, dp]
                elif corruption == "disconnected_shuffle":
                    rolled_twice = np.roll(valid_paths, 2, axis=0)
                    for target, middle_donor, right_donor in zip(
                        valid_paths, rolled_once, rolled_twice
                    ):
                        tr, tp = map(int, target)
                        mr, mp = map(int, middle_donor)
                        rr, rp = map(int, right_donor)
                        for slot, donor_row, donor_path in (
                            (1, mr, mp), (2, rr, rp)
                        ):
                            path_sources[tr, tp, slot] = source_copy[donor_row, donor_path, slot]
                            path_destinations[tr, tp, slot] = destination_copy[donor_row, donor_path, slot]
                            path_features[tr, tp, slot] = feature_copy[donor_row, donor_path, slot]
                else:
                    raise ValueError(
                        "three_hop_corruption must be none, break_binding, middle_binding_shuffle, "
                        "time_shuffle, or disconnected_shuffle"
                    )
        return batch + (
            torch.tensor(path_sources, device=device),
            torch.tensor(path_destinations, device=device),
            torch.tensor(path_times, dtype=torch.float32, device=device),
            torch.tensor(path_features, dtype=torch.float32, device=device),
            torch.tensor(path_masks, dtype=torch.bool, device=device),
        ), historical_selection

    last_train = 0.0
    last_symbolic_metrics: dict[str, float] = {}
    for epoch in epoch_iter(spec, config):
        model.train()
        if hasattr(model, "set_symbolic_progress"):
            epoch_count = int(training["epochs"])
            progress = (
                0.0 if epoch_count <= 1 else float(epoch) / float(epoch_count - 1)
            )
            model.set_symbolic_progress(progress)
        losses = []
        symbolic_epoch: dict[str, list[float]] = {}
        epoch_batch_size = int(training["batch_size"])
        if (
            str(training.get("objective", "binary_cross_entropy"))
            == "in_batch_risk_set"
            and epoch < int(training.get("risk_set_warmup_epochs", 0))
        ):
            epoch_batch_size = int(
                training.get("risk_set_warmup_batch_size", epoch_batch_size)
            )
        if epoch_batch_size < 1:
            raise ValueError("training batch size must be positive")
        for rows in progress_batches(
            train_idx,
            epoch_batch_size,
            shuffle=True,
            desc=f"epoch {epoch + 1}/{int(training['epochs'])} batches",
            config=config,
        ):
            objective = str(training.get("objective", "binary_cross_entropy"))
            risk_set_warmup_epochs = int(
                training.get("risk_set_warmup_epochs", 0)
            )
            if risk_set_warmup_epochs < 0:
                raise ValueError("risk_set_warmup_epochs must be non-negative")
            if objective == "in_batch_risk_set" and epoch < risk_set_warmup_epochs:
                objective = "binary_cross_entropy"
            pos, _ = make_batch(rows, negative=False)
            if objective == "in_batch_risk_set":
                positive_destinations = frame.iloc[rows]["dst"].to_numpy(dtype=np.int64)
                uniform_count = int(training.get("risk_set_uniform_candidates", 0))
                if uniform_count < 0:
                    raise ValueError("risk_set_uniform_candidates must be non-negative")
                uniform_destinations = (
                    np.random.choice(
                        dst_pool,
                        size=uniform_count,
                        replace=uniform_count > len(dst_pool),
                    ).astype(np.int64)
                    if uniform_count
                    else np.empty(0, dtype=np.int64)
                )
                candidate_destinations, target = np.unique(
                    np.concatenate((positive_destinations, uniform_destinations)),
                    return_inverse=True,
                )
                target = target[:len(positive_destinations)]
                candidate_count = len(candidate_destinations)
                expanded_rows = np.repeat(rows, candidate_count)
                expanded_destinations = np.tile(candidate_destinations, len(rows))
                candidates, _ = make_batch(
                    expanded_rows,
                    negative=False,
                    destination_override=expanded_destinations,
                )
                candidate_result = model(*candidates)
                candidate_logits = candidate_result.logits.reshape(
                    len(rows), candidate_count
                )
                loss = F.cross_entropy(
                    candidate_logits,
                    torch.tensor(target, dtype=torch.long, device=device),
                )
                conditional_task_loss = loss
                pointwise_task_loss = None
                risk_set_bce_weight = float(
                    training.get("risk_set_bce_weight", 0.0)
                )
                if risk_set_bce_weight < 0:
                    raise ValueError("risk_set_bce_weight must be non-negative")
                if risk_set_bce_weight:
                    candidate_labels = torch.zeros_like(candidate_logits)
                    candidate_labels.scatter_(
                        1,
                        torch.tensor(target, dtype=torch.long, device=device)[:, None],
                        1.0,
                    )
                    balanced_bce = F.binary_cross_entropy_with_logits(
                        candidate_logits,
                        candidate_labels,
                        pos_weight=torch.tensor(
                            max(1, candidate_count - 1),
                            dtype=candidate_logits.dtype,
                            device=device,
                        ),
                    )
                    loss = loss + risk_set_bce_weight * balanced_bce
                pos_result = model(*pos)
                random_bce_weight = float(
                    training.get("risk_set_random_bce_weight", 0.0)
                )
                if random_bce_weight < 0:
                    raise ValueError("risk_set_random_bce_weight must be non-negative")
                random_result = None
                if random_bce_weight:
                    random_batch, _ = make_batch(rows, negative=True)
                    random_result = model(*random_batch)
                    pointwise_logits = torch.cat(
                        (pos_result.logits, random_result.logits)
                    )
                    pointwise_labels = torch.cat(
                        (
                            torch.ones_like(pos_result.logits),
                            torch.zeros_like(random_result.logits),
                        )
                    )
                    random_loss_type = str(
                        training.get("risk_set_random_loss", "bce")
                    )
                    if random_loss_type == "bce":
                        pointwise_task_loss = F.binary_cross_entropy_with_logits(
                            pointwise_logits, pointwise_labels
                        )
                    elif random_loss_type in {"margin", "soft_margin"}:
                        margin = float(training.get("risk_set_random_margin", 1.0))
                        if margin <= 0:
                            raise ValueError("risk_set_random_margin must be positive")
                        margin_error = margin - (
                            pos_result.logits - random_result.logits
                        )
                        pointwise_task_loss = (
                            torch.relu(margin_error).mean()
                            if random_loss_type == "margin"
                            else F.softplus(margin_error).mean()
                        )
                    else:
                        raise ValueError(
                            "risk_set_random_loss must be bce, margin, or soft_margin"
                        )
                    loss = loss + random_bce_weight * pointwise_task_loss
                neg_result = candidate_result
                pos_logits = pos_result.logits
                neg_logits = candidate_result.logits
                logits = neg_logits
                labels = torch.zeros_like(neg_logits)
                regularizer_results = (
                    (pos_result, candidate_result, random_result)
                    if random_result is not None
                    else (pos_result, candidate_result)
                )
            else:
                negative_count = int(training.get("negatives_per_positive", 1))
                if negative_count < 1:
                    raise ValueError("negatives_per_positive must be positive")
                negative_mode: bool | str = (
                    "in_batch"
                    if str(training.get("negative_sampling", "uniform")) == "in_batch"
                    else True
                )
                negative_batches = [
                    make_batch(rows, negative=negative_mode)[0]
                    for _ in range(negative_count)
                ]
                neg = tuple(
                    torch.cat([batch[position] for batch in negative_batches], dim=0)
                    for position in range(len(negative_batches[0]))
                )
                if is_lifter and negative_count == 1:
                    pos_result, neg_result = model.score_candidate_batches(pos, neg)
                else:
                    pos_result = model(*pos)
                    neg_result = model(*neg)
                pos_logits = pos_result.logits
                neg_logits = neg_result.logits
                if objective == "sampled_competing_risk":
                    candidate_logits = torch.cat(
                        (pos_logits[:, None], neg_logits.reshape(negative_count, -1).T),
                        dim=1,
                    )
                    loss = F.cross_entropy(
                        candidate_logits,
                        torch.zeros(len(rows), dtype=torch.long, device=device),
                    )
                    logits = torch.cat([pos_logits, neg_logits])
                    labels = torch.cat(
                        [torch.ones_like(pos_logits), torch.zeros_like(neg_logits)]
                    )
                elif objective == "binary_cross_entropy":
                    logits = torch.cat([pos_logits, neg_logits])
                    labels = torch.cat(
                        [torch.ones_like(pos_logits), torch.zeros_like(neg_logits)]
                    )
                    loss = F.binary_cross_entropy_with_logits(logits, labels)
                else:
                    raise ValueError(
                        "training objective must be binary_cross_entropy, "
                        "sampled_competing_risk, or in_batch_risk_set"
                    )
                regularizer_results = (pos_result, neg_result)
            auxiliary_logits = [
                result.aux.get("auxiliary_logits")
                for result in regularizer_results
            ]
            if objective != "in_batch_risk_set" and all(
                value is not None for value in auxiliary_logits
            ):
                positive_aux, negative_aux = auxiliary_logits
                aux_logits = torch.cat((positive_aux, negative_aux), dim=0)
                aux_labels = labels[:, None].expand_as(aux_logits)
                loss = loss + F.binary_cross_entropy_with_logits(
                    aux_logits, aux_labels
                )
            regularizers = [
                result.aux["regularization_loss"]
                for result in regularizer_results
                if "regularization_loss" in result.aux
            ]
            if regularizers:
                regularization_term = torch.stack(regularizers).mean()
                loss = loss + regularization_term
            else:
                regularization_term = None
            for key in (
                "predicate_entropy",
                "predicate_diversity_kl",
                "rule_entropy",
                "rule_usage_entropy",
                "rule_diversity_kl",
            ):
                values = [
                    result.aux[key]
                    for result in regularizer_results
                    if key in result.aux
                ]
                if values:
                    symbolic_epoch.setdefault(key, []).append(
                        float(torch.stack(values).mean().detach().cpu())
                    )
            opt.zero_grad()
            gradient_strategy = str(training.get("gradient_strategy", "sum"))
            if gradient_strategy not in {"sum", "pcgrad", "primary_pcgrad"}:
                raise ValueError(
                    "gradient_strategy must be sum, pcgrad, or primary_pcgrad"
                )
            use_pcgrad = (
                gradient_strategy in {"pcgrad", "primary_pcgrad"}
                and objective == "in_batch_risk_set"
                and pointwise_task_loss is not None
            )
            if use_pcgrad:
                parameters = [
                    parameter for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                first = torch.autograd.grad(
                    conditional_task_loss,
                    parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                second = torch.autograd.grad(
                    pointwise_task_loss,
                    parameters,
                    retain_graph=regularization_term is not None,
                    allow_unused=True,
                )
                regularization_grads = (
                    torch.autograd.grad(
                        regularization_term,
                        parameters,
                        allow_unused=True,
                    )
                    if regularization_term is not None
                    else [None] * len(parameters)
                )
                dot = sum(
                    (left * right).sum()
                    for left, right in zip(first, second)
                    if left is not None and right is not None
                )
                first_norm = sum(
                    value.square().sum() for value in first if value is not None
                ).clamp_min(1e-12)
                second_norm = sum(
                    value.square().sum() for value in second if value is not None
                ).clamp_min(1e-12)
                conflict = dot < 0
                for parameter, left, right, regularizer_grad in zip(
                    parameters, first, second, regularization_grads
                ):
                    combined = None
                    if left is not None:
                        combined = left
                        if gradient_strategy == "pcgrad":
                            combined = combined - torch.where(
                                conflict, dot / second_norm, dot.new_zeros(())
                            ) * (
                                right if right is not None else torch.zeros_like(left)
                            )
                    if right is not None:
                        projected = right - torch.where(
                            conflict, dot / first_norm, dot.new_zeros(())
                        ) * (left if left is not None else torch.zeros_like(right))
                        combined = projected if combined is None else combined + projected
                    if regularizer_grad is not None:
                        combined = (
                            regularizer_grad
                            if combined is None
                            else combined + regularizer_grad
                        )
                    parameter.grad = combined
            else:
                loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        last_train = float(np.mean(losses)) if losses else 0.0
        last_symbolic_metrics = {
            key: float(np.mean(values)) for key, values in symbolic_epoch.items()
        }
        if progress_enabled(config):
            tqdm.write(f"{spec.model}/{spec.variant} epoch {epoch + 1}: train_loss={last_train:.6f}")

    model.eval()
    # Evaluation candidates must not depend on how many random numbers a
    # training objective happened to consume. Reset the sampler so every model
    # and architecture variant with the same seed sees identical random and
    # historical alternatives, in the same query order.
    np.random.seed(spec.seed + 104729)
    labels_all, scores_all, losses = [], [], []
    soft_scores_all: list[float] = []
    projection_logit_differences: list[float] = []
    positive_grounding_valid: list[float] = []
    negative_grounding_valid: list[float] = []
    historical_labels_all: list[float] = []
    historical_scores_all: list[float] = []
    historical_losses: list[float] = []
    historical_selection_all: list[float] = []
    historical_pairwise_all: list[float] = []
    recent_historical_labels_all: list[float] = []
    recent_historical_scores_all: list[float] = []
    recent_historical_selection_all: list[float] = []
    recent_historical_pairwise_all: list[float] = []
    proof_matched_labels_all: list[float] = []
    proof_matched_scores_all: list[float] = []
    proof_matched_selection_all: list[float] = []
    proof_matched_pairwise_all: list[float] = []
    grounded_deletion_top_deltas: list[float] = []
    grounded_deletion_random_deltas: list[float] = []
    mechanism_scores: dict[str, list[float]] = {}
    historical_mechanism_scores: dict[str, list[float]] = {}
    historical_mechanism_margins: dict[str, list[float]] = {}
    top_deleted_scores: list[float] = []
    random_deleted_scores: list[float] = []
    top_proof_logit_deltas: list[float] = []
    random_proof_logit_deltas: list[float] = []
    historical_family_margins: dict[str, list[float]] = {}
    historical_logit_margins: list[float] = []
    historical_positive_recurrence: list[float] = []
    deep_mechanism_config = config.get("mechanism_decomposition", {})
    deep_mechanism_enabled = is_lifter and bool(
        deep_mechanism_config.get("enabled", False)
    )
    deep_component_names = (
        "direct_pair",
        "source_context",
        "candidate_context",
        "pair_renewal",
        "positioned_recurrence",
        "ordered_transition_1",
        "ordered_transition_2",
    )
    deep_positive_components: dict[str, list[float]] = {
        name: [] for name in deep_component_names
    }
    deep_historical_components: dict[str, list[float]] = {
        name: [] for name in deep_component_names
    }
    deep_positive_prior: list[float] = []
    deep_historical_prior: list[float] = []
    deep_rows: list[int] = []
    deep_direct_recurrence: list[bool] = []
    deep_source_activity: list[int] = []
    deep_destination_popularity: list[int] = []
    deep_test_position: list[float] = []
    fact_intervention_config = deep_mechanism_config.get("fact_intervention", {})
    fact_intervention_enabled = deep_mechanism_enabled and bool(
        fact_intervention_config.get("enabled", False)
    )
    fact_intervention_component = str(
        fact_intervention_config.get("component", "pair_renewal")
    )
    fact_intervention_limit = int(fact_intervention_config.get("max_queries", 512))
    fact_intervention_targets = np.zeros(len(eval_idx), dtype=bool)
    if fact_intervention_enabled and len(eval_idx) and fact_intervention_limit > 0:
        target_count = min(len(eval_idx), fact_intervention_limit)
        fact_intervention_targets[
            np.linspace(0, len(eval_idx) - 1, target_count, dtype=np.int64)
        ] = True
    fact_intervention_cursor = 0
    fact_intervention_original_positive: list[float] = []
    fact_intervention_original_negative: list[float] = []
    fact_intervention_top_positive: list[float] = []
    fact_intervention_top_negative: list[float] = []
    fact_intervention_random_positive: list[float] = []
    fact_intervention_random_negative: list[float] = []
    fact_intervention_support_top_margin_drop: list[float] = []
    fact_intervention_support_random_margin_drop: list[float] = []
    fact_intervention_oppose_top_margin_gain: list[float] = []
    fact_intervention_oppose_random_margin_gain: list[float] = []
    query_diagnostic_config = config.get("query_diagnostics", {})
    query_diagnostics_enabled = bool(query_diagnostic_config.get("enabled", False))
    query_diagnostic_rows: list[int] = []
    query_diagnostic_positive_destinations: list[int] = []
    query_diagnostic_historical_destinations: list[int] = []
    query_diagnostic_positive_logits: list[float] = []
    query_diagnostic_historical_logits: list[float] = []
    query_diagnostic_historical_selected: list[bool] = []
    predicate_diagnostics_enabled = is_lifter and bool(
        config.get("predicate_diagnostics", {}).get("enabled", False)
    )
    predicate_component_mass: np.ndarray | None = None
    predicate_query_active_counts: list[float] = []
    predicate_top_fraction: dict[int, list[float]] = {1: [], 3: [], 5: []}
    predicate_reconstruction_errors: list[float] = []
    predicate_role_profiles: dict[str, np.ndarray] = {}
    predicate_group_counts: dict[str, int] = {}
    predicate_rule_roles: np.ndarray | None = None
    if predicate_diagnostics_enabled:
        predicate_count = int(model.predicate_count)
        grounding_capacity = int(model.max_grounding_facts)
        predicate_component_mass = np.zeros(
            model.rule_count
            + grounding_capacity * predicate_count
            + predicate_count
            + predicate_count * predicate_count,
            dtype=np.float64,
        )
        predicate_rule_roles = np.asarray(
            [6 if rule.renewal else int(rule.left_role) for rule in model.program],
            dtype=np.int64,
        )
        predicate_role_profiles = {
            group: np.zeros((predicate_count, 10), dtype=np.float64)
            for group in (
                "all", "seen_entities", "unseen_entity", "early_interval",
                "later_interval", "low_source_activity", "high_source_activity",
                "low_destination_popularity", "high_destination_popularity",
            )
        }
        predicate_group_counts = {group: 0 for group in predicate_role_profiles}
        train_frame = frame.iloc[train_idx]
        train_sources = set(train_frame["src"].astype(int))
        train_destinations = set(train_frame["dst"].astype(int))
        source_activity = train_frame["src"].value_counts().to_dict()
        destination_popularity = train_frame["dst"].value_counts().to_dict()
        eval_sources = frame.iloc[eval_idx]["src"].map(source_activity).fillna(0).to_numpy()
        eval_destinations = (
            frame.iloc[eval_idx]["dst"].map(destination_popularity).fillna(0).to_numpy()
        )
        source_activity_threshold = float(np.median(eval_sources))
        destination_popularity_threshold = float(np.median(eval_destinations))
        eval_midpoint = int(eval_idx[len(eval_idx) // 2])
    explanation_config = config.get("explanation_evaluation", {})
    explanation_sparsities = [
        float(value) for value in explanation_config.get(
            "explanation_ratios", [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        )
    ]
    explanation_timeout_ms = 1000.0 * float(
        explanation_config.get("timeout_seconds", 60)
    )
    explanation_limit = int(explanation_config.get("max_queries", 500))
    explanation_seen = 0
    intrinsic_sufficiency: list[list[float]] = []
    intrinsic_necessity: list[list[float]] = []
    intrinsic_counts: list[list[float]] = []
    intrinsic_times_ms: list[float] = []
    deletion_rng = np.random.default_rng(spec.seed + 104729)
    explanation_example: dict[str, Any] | None = None
    contrastive_explanation: dict[str, Any] | None = None
    contrastive_best_margin = -float("inf")
    with torch.no_grad():
        for rows in progress_batches(eval_idx, int(training["batch_size"]), shuffle=False, desc="eval batches", config=config):
            pos, _ = make_batch(rows, negative=False)
            neg, _ = make_batch(rows, negative=True)
            historical_neg, historical_selected = make_batch(rows, negative="historical")
            recent_historical_neg, recent_historical_selected = make_batch(
                rows, negative="historical_recent"
            )
            proof_matched_neg = None
            proof_matched_selected = None
            if proof_matched_enabled:
                proof_matched_neg, proof_matched_selected = make_batch(
                    rows, negative="proof_matched"
                )
            if is_lifter:
                candidate_batches = [pos, neg, historical_neg, recent_historical_neg]
                if proof_matched_neg is not None:
                    candidate_batches.append(proof_matched_neg)
                candidate_results = model.score_candidate_batches(*candidate_batches)
                (
                    pos_result,
                    neg_result,
                    historical_neg_result,
                    recent_historical_neg_result,
                ) = candidate_results[:4]
                proof_matched_result = (
                    candidate_results[4] if proof_matched_neg is not None else None
                )
            else:
                pos_result = model(*pos)
                neg_result = model(*neg)
                historical_neg_result = model(*historical_neg)
                recent_historical_neg_result = model(*recent_historical_neg)
                proof_matched_result = (
                    model(*proof_matched_neg) if proof_matched_neg is not None else None
                )
            pos_logits = pos_result.logits
            neg_logits = neg_result.logits
            if predicate_diagnostics_enabled:
                rule_abs = pos_result.aux["rule_scores"].abs().cpu().numpy()
                positioned_abs = pos_result.aux[
                    "positioned_recurrence_fact_contributions"
                ].abs().cpu().numpy()
                src_predicates = pos_result.aux["predicate_ids_src"].cpu().numpy()
                q1_abs = pos_result.aux["transition_rule_contribution"].abs().cpu().numpy()
                q1_predicates = pos_result.aux["transition_previous_predicate"].cpu().numpy()
                q2_abs = pos_result.aux["transition2_rule_contribution"].abs().cpu().numpy()
                q2_first = pos_result.aux["transition2_first_predicate"].cpu().numpy()
                q2_second = pos_result.aux["transition2_second_predicate"].cpu().numpy()
                predicate_count = int(model.predicate_count)
                grounding_capacity = int(model.max_grounding_facts)
                rule_offset = int(model.rule_count)
                assert predicate_component_mass is not None
                predicate_component_mass[:rule_offset] += rule_abs.sum(0)
                for sample in range(len(rows)):
                    for fact_index, contribution in enumerate(positioned_abs[sample]):
                        if contribution == 0:
                            continue
                        position = positioned_abs.shape[1] - fact_index - 1
                        predicate = int(src_predicates[sample, fact_index])
                        predicate_component_mass[
                            rule_offset + position * predicate_count + predicate
                        ] += contribution
                    if q1_abs[sample] > 0 and q1_predicates[sample] >= 0:
                        predicate_component_mass[
                            rule_offset + grounding_capacity * predicate_count
                            + int(q1_predicates[sample])
                        ] += q1_abs[sample]
                    if q2_abs[sample] > 0 and q2_first[sample] >= 0 and q2_second[sample] >= 0:
                        predicate_component_mass[
                            rule_offset + grounding_capacity * predicate_count
                            + predicate_count
                            + int(q2_first[sample]) * predicate_count
                            + int(q2_second[sample])
                        ] += q2_abs[sample]
                execution_abs = torch.cat(
                    (
                        pos_result.aux["rule_scores"].abs(),
                        pos_result.aux["positioned_recurrence_fact_contributions"].abs(),
                        pos_result.aux["transition_rule_contribution"].abs()[:, None],
                        pos_result.aux["transition2_rule_contribution"].abs()[:, None],
                    ),
                    dim=1,
                )
                totals = execution_abs.sum(1)
                predicate_query_active_counts.extend(
                    (execution_abs > 1e-8).sum(1).float().cpu().tolist()
                )
                for top_k in predicate_top_fraction:
                    selected = execution_abs.topk(
                        min(top_k, execution_abs.shape[1]), dim=1
                    ).values.sum(1)
                    fraction = torch.where(totals > 0, selected / totals, torch.ones_like(totals))
                    predicate_top_fraction[top_k].extend(fraction.cpu().tolist())
                reconstructed = (
                    pos_result.aux["prior_rule_contribution"]
                    + pos_result.aux["rule_scores"].sum(-1)
                    + pos_result.aux["positioned_recurrence_rule_contribution"]
                    + pos_result.aux["transition_rule_contribution"]
                    + pos_result.aux["transition2_rule_contribution"]
                )
                predicate_reconstruction_errors.extend(
                    (pos_logits - reconstructed).abs().cpu().tolist()
                )
                query_frame = frame.iloc[rows]
                src_values = query_frame["src"].astype(int).to_numpy()
                dst_values = query_frame["dst"].astype(int).to_numpy()
                src_activity_values = np.asarray(
                    [source_activity.get(value, 0) for value in src_values]
                )
                dst_popularity_values = np.asarray(
                    [destination_popularity.get(value, 0) for value in dst_values]
                )
                group_masks = {
                    "all": np.ones(len(rows), dtype=bool),
                    "seen_entities": np.asarray([
                        source in train_sources and destination in train_destinations
                        for source, destination in zip(src_values, dst_values)
                    ]),
                    "unseen_entity": np.asarray([
                        source not in train_sources or destination not in train_destinations
                        for source, destination in zip(src_values, dst_values)
                    ]),
                    "early_interval": rows < eval_midpoint,
                    "later_interval": rows >= eval_midpoint,
                    "low_source_activity": src_activity_values <= source_activity_threshold,
                    "high_source_activity": src_activity_values > source_activity_threshold,
                    "low_destination_popularity": dst_popularity_values <= destination_popularity_threshold,
                    "high_destination_popularity": dst_popularity_values > destination_popularity_threshold,
                }
                assert predicate_rule_roles is not None
                for group, mask in group_masks.items():
                    if not bool(mask.any()):
                        continue
                    predicate_group_counts[group] += int(mask.sum())
                    profile = predicate_role_profiles[group]
                    rule_mass = rule_abs[mask].sum(0)
                    for rule_id, mass in enumerate(rule_mass):
                        if mass == 0:
                            continue
                        rule = model.program[rule_id]
                        profile[int(rule.left_predicate), predicate_rule_roles[rule_id]] += mass
                        if rule.renewal and rule.right_predicate >= 0:
                            profile[int(rule.left_predicate), 6] -= 0.5 * mass
                            profile[int(rule.right_predicate), 6] += 0.5 * mass
                    selected_positions = positioned_abs[mask]
                    selected_predicates = src_predicates[mask]
                    for predicate in range(predicate_count):
                        profile[predicate, 7] += selected_positions[
                            selected_predicates == predicate
                        ].sum()
                    for predicate in range(predicate_count):
                        profile[predicate, 8] += q1_abs[mask & (q1_predicates == predicate)].sum()
                        profile[predicate, 9] += q2_abs[
                            mask & ((q2_first == predicate) | (q2_second == predicate))
                        ].sum()
            if (
                bool(config.get("contrastive_explanation_evaluation", {}).get("enabled", False))
                and is_lifter
                and hasattr(model, "explain")
            ):
                top_lengths = pos_result.aux.get("top_rule_length")
                if top_lengths is not None:
                    margins = pos_logits - historical_neg_result.logits
                    eligible = (top_lengths == 3) & torch.tensor(
                        historical_selected, dtype=torch.bool, device=device
                    )
                    if bool(eligible.any()):
                        candidate_ids = torch.nonzero(eligible, as_tuple=False).flatten()
                        local = int(candidate_ids[margins[candidate_ids].argmax()])
                        margin = float(margins[local].detach().cpu())
                        if margin > contrastive_best_margin:
                            positive_one = tuple(value[local:local + 1] for value in pos)
                            negative_one = tuple(value[local:local + 1] for value in historical_neg)
                            contrastive_explanation = {
                                "row": int(rows[local]),
                                "positive": model.explain(*positive_one)[0],
                                "historical_negative": model.explain(*negative_one)[0],
                                "logit_margin": margin,
                            }
                            contrastive_best_margin = margin
            if (
                bool(config.get("grounded_deletion_evaluation", {}).get("enabled", False))
                and is_lifter
                and len(pos) >= 16
                and "top_path_index" in pos_result.aux
            ):
                top_indices = pos_result.aux["top_path_index"].detach()
                top_lengths = pos_result.aux["top_rule_length"].detach()
                path_sources, path_destinations, path_times = pos[11], pos[12], pos[13]
                original_mask = pos[15].bool()
                top_mask = original_mask.clone()
                random_mask = original_mask.clone()
                rng = np.random.default_rng(spec.seed + int(rows[0]))
                eligible = torch.zeros(len(rows), dtype=torch.bool, device=device)
                for sample in range(len(rows)):
                    path_index = int(top_indices[sample])
                    if int(top_lengths[sample]) != 3 or path_index < 0 or not bool(original_mask[sample, path_index]):
                        continue
                    eligible[sample] = True
                    # Delete the concrete middle fact from every grounding in
                    # which it occurs, then re-execute the complete program.
                    fact_source = path_sources[sample, path_index, 1]
                    fact_destination = path_destinations[sample, path_index, 1]
                    fact_time = path_times[sample, path_index, 1]
                    contains = (
                        (path_sources[sample] == fact_source)
                        & (path_destinations[sample] == fact_destination)
                        & torch.isclose(path_times[sample], fact_time)
                    ).any(-1)
                    top_mask[sample] &= ~contains
                    alternatives = torch.nonzero(original_mask[sample], as_tuple=False).flatten()
                    random_path = int(alternatives[int(rng.integers(len(alternatives)))])
                    random_source = path_sources[sample, random_path, 1]
                    random_destination = path_destinations[sample, random_path, 1]
                    random_time = path_times[sample, random_path, 1]
                    random_contains = (
                        (path_sources[sample] == random_source)
                        & (path_destinations[sample] == random_destination)
                        & torch.isclose(path_times[sample], random_time)
                    ).any(-1)
                    random_mask[sample] &= ~random_contains
                if bool(eligible.any()):
                    top_inputs = list(pos); top_inputs[15] = top_mask
                    random_inputs = list(pos); random_inputs[15] = random_mask
                    top_logits = model(*tuple(top_inputs)).logits
                    random_logits = model(*tuple(random_inputs)).logits
                    grounded_deletion_top_deltas.extend(
                        (pos_logits[eligible] - top_logits[eligible]).abs().detach().cpu().tolist()
                    )
                    grounded_deletion_random_deltas.extend(
                        (pos_logits[eligible] - random_logits[eligible]).abs().detach().cpu().tolist()
                    )
            if proof_matched_result is not None and proof_matched_selected is not None:
                selected_tensor = torch.tensor(
                    proof_matched_selected, dtype=torch.bool, device=device
                )
                selected_positive = pos_logits[selected_tensor].detach().cpu().tolist()
                selected_negative = proof_matched_result.logits[selected_tensor].detach().cpu().tolist()
                proof_matched_labels_all.extend(
                    [1.0] * len(selected_positive) + [0.0] * len(selected_negative)
                )
                proof_matched_scores_all.extend(selected_positive + selected_negative)
                proof_matched_pairwise_all.extend(
                    (np.asarray(selected_positive) > np.asarray(selected_negative)).astype(float).tolist()
                )
                proof_matched_selection_all.extend(
                    proof_matched_selected.astype(float).tolist()
                )
            if query_diagnostics_enabled:
                query_diagnostic_rows.extend(rows.astype(np.int64).tolist())
                query_diagnostic_positive_destinations.extend(
                    pos[1].detach().cpu().to(torch.int64).tolist()
                )
                query_diagnostic_historical_destinations.extend(
                    historical_neg[1].detach().cpu().to(torch.int64).tolist()
                )
                query_diagnostic_positive_logits.extend(
                    pos_logits.detach().cpu().tolist()
                )
                query_diagnostic_historical_logits.extend(
                    historical_neg_result.logits.detach().cpu().tolist()
                )
                query_diagnostic_historical_selected.extend(
                    historical_selected.astype(bool).tolist()
                )
            event_importance = pos_result.aux.get("event_importance")
            if event_importance is not None and explanation_seen < explanation_limit:
                take = min(len(rows), explanation_limit - explanation_seen)
                positive = tuple(value[:take] for value in pos)
                # LiFTER executes only its bounded suffix from each endpoint
                # history.  Align the perturbation mask and importance vector
                # to that exact executed database; otherwise valid indices
                # from the unexecuted prefix can exceed the trace width.
                grounding_width = int(
                    getattr(model, "max_grounding_facts", positive[3].shape[1])
                )
                if grounding_width < positive[3].shape[1]:
                    aligned = list(positive)
                    for position in range(3, 11):
                        aligned[position] = aligned[position][:, -grounding_width:]
                    positive = tuple(aligned)
                scores = event_importance[:take]
                use_counterfactual = bool(
                    getattr(model, "counterfactual_explanation", False)
                )
                timing_indices: list[int] = []
                for sample in range(take):
                    timing_indices.append(len(intrinsic_times_ms))
                    if use_counterfactual:
                        intrinsic_times_ms.append(0.0)
                        continue
                    one = tuple(value[sample:sample + 1] for value in positive)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    timed_result = model(*one)
                    timed_importance = timed_result.aux["event_importance"]
                    _ = timed_importance
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    intrinsic_times_ms.append(
                        1000.0 * (time.perf_counter() - started)
                    )
                (
                    scores,
                    valid_events,
                    endpoint_universe_ids,
                    path_universe_ids,
                ) = build_grounded_explanation_universe(
                    positive, scores
                )
                if use_counterfactual:
                    counterfactual_rows = []
                    for sample in range(take):
                        one = tuple(
                            value[sample:sample + 1] for value in positive
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        counterfactual_rows.append(
                            counterfactual_universe_responsibility(
                                model,
                                one,
                                scores[sample:sample + 1],
                                endpoint_universe_ids[sample:sample + 1],
                                path_universe_ids[sample:sample + 1],
                                valid_events[sample:sample + 1],
                            )
                        )
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        intrinsic_times_ms[timing_indices[sample]] += (
                            1000.0 * (time.perf_counter() - started)
                        )
                    scores = torch.cat(counterfactual_rows, dim=0)
                src_width = positive[3].shape[1]
                original_probability = pos_logits[:take].sigmoid()
                sufficiency_row, necessity_row, count_row = [], [], []
                original_decision = original_probability >= 0.5
                for sparsity in explanation_sparsities:
                    selected = torch.zeros_like(valid_events)
                    for sample in range(take):
                        ids = torch.nonzero(valid_events[sample], as_tuple=False).flatten()
                        if ids.numel() == 0:
                            continue
                        keep = max(1, int(math.ceil(float(sparsity) * ids.numel())))
                        chosen = ids[scores[sample, ids].topk(keep).indices]
                        selected[sample, chosen] = True
                    def masked_probability(mask: torch.Tensor) -> torch.Tensor:
                        masked = list(positive)
                        endpoint_present = (
                            mask.gather(
                                1, endpoint_universe_ids.clamp_min(0)
                            )
                            & (endpoint_universe_ids >= 0)
                        )
                        masked[6] = positive[6].bool() & endpoint_present[:, :src_width]
                        masked[10] = (
                            positive[10].bool() & endpoint_present[:, src_width:]
                        )
                        if len(masked) > 15:
                            path_slots_present = (
                                mask.gather(
                                    1, path_universe_ids.flatten(1).clamp_min(0)
                                ).reshape_as(path_universe_ids)
                                & (path_universe_ids >= 0)
                            )
                            masked[15] = (
                                positive[15].bool()
                                & path_slots_present.all(-1)
                            )
                        return model(*tuple(masked)).logits.sigmoid()
                    sufficient_decision = masked_probability(selected) >= 0.5
                    sufficiency_row.append(
                        (sufficient_decision == original_decision).float().cpu().tolist()
                    )
                    necessity_row.append(
                        torch.abs(original_probability - masked_probability(valid_events & ~selected)).cpu().tolist()
                    )
                    count_row.append(selected.sum(1).float().cpu().tolist())
                intrinsic_sufficiency.extend(np.asarray(sufficiency_row).T.tolist())
                intrinsic_necessity.extend(np.asarray(necessity_row).T.tolist())
                intrinsic_counts.extend(np.asarray(count_row).T.tolist())
                explanation_seen += take
            if lifter_diagnostics_enabled:
                positive_grounding_valid.extend(
                    pos_result.aux["top_grounding_valid"].float().cpu().tolist()
                )
                negative_grounding_valid.extend(
                    neg_result.aux["top_grounding_valid"].float().cpu().tolist()
                )
                with model.execution_mode("soft"):
                    soft_pos_logits = model(*pos).logits
                    soft_neg_logits = model(*neg).logits
                soft_logits = torch.cat([soft_pos_logits, soft_neg_logits])
                soft_scores_all.extend(torch.sigmoid(soft_logits).cpu().tolist())
                projection_logit_differences.extend(
                    torch.abs(torch.cat([pos_logits, neg_logits]) - soft_logits)
                    .cpu()
                    .tolist()
                )
                if explanation_example is None and hasattr(model, "explain"):
                    valid_examples = torch.nonzero(
                        pos_result.aux["top_grounding_valid"], as_tuple=False
                    ).flatten()
                    if valid_examples.numel():
                        sample = int(valid_examples[0])
                        one_positive = tuple(
                            value[sample:sample + 1] for value in pos
                        )
                        explanation_example = model.explain(*one_positive)[0]
                pos_mechanisms = lifter_mechanism_logits(model, pos_result)
                neg_mechanisms = lifter_mechanism_logits(model, neg_result)
                historical_neg_mechanisms = lifter_mechanism_logits(
                    model, historical_neg_result
                )
                lengths = torch.tensor(
                    [rule.length for rule in model.program], device=device
                )
                roles = model.left_roles.to(device)
                direct = (roles == 0) | (roles == 1)
                renewal = model.renewal_rules.to(device)
                family_masks = {
                    "direct_unary": (lengths == 1) & direct,
                    "source_context_unary": (lengths == 1)
                    & ((roles == 2) | (roles == 3)),
                    "destination_context_unary": (lengths == 1)
                    & ((roles == 4) | (roles == 5)),
                    "renewal": renewal,
                    "direct_conjunction": (lengths == 2) & direct & ~renewal,
                    "context_conjunction": (lengths == 2) & ~direct & ~renewal,
                    "length_three": lengths == 3,
                }
                for family, family_mask in family_masks.items():
                    margin = (
                        pos_result.aux["rule_scores"][:, family_mask].sum(-1)
                        - historical_neg_result.aux["rule_scores"][:, family_mask].sum(-1)
                    )
                    historical_family_margins.setdefault(family, []).extend(
                        margin.cpu().tolist()
                    )
                historical_logit_margins.extend(
                    (pos_logits - historical_neg_result.logits).cpu().tolist()
                )
                positive_values = frame.iloc[rows]["dst"].to_numpy(dtype=np.int64)
                positive_recurrence = (
                    source_hist_mask[rows]
                    & (source_hist_nodes[rows] == positive_values[:, None])
                    & (source_hist_features[rows, :, 0] >= 0)
                ).any(axis=1)
                historical_positive_recurrence.extend(
                    positive_recurrence.astype(np.float32).tolist()
                )
                if deep_mechanism_enabled:
                    pos_prior = pos_mechanisms["prior_only"]
                    historical_prior = historical_neg_mechanisms["prior_only"]
                    deep_positive_prior.extend(pos_prior.cpu().tolist())
                    deep_historical_prior.extend(historical_prior.cpu().tolist())
                    for component_name in deep_component_names:
                        deep_positive_components[component_name].extend(
                            (
                                pos_mechanisms[f"only_{component_name}"]
                                - pos_prior
                            ).cpu().tolist()
                        )
                        deep_historical_components[component_name].extend(
                            (
                                historical_neg_mechanisms[f"only_{component_name}"]
                                - historical_prior
                            ).cpu().tolist()
                        )
                    deep_rows.extend(rows.astype(np.int64).tolist())
                    deep_direct_recurrence.extend(positive_recurrence.tolist())
                    deep_source_activity.extend(
                        causal_source_activity[rows].tolist()
                    )
                    deep_destination_popularity.extend(
                        causal_destination_popularity[rows].tolist()
                    )
                    eval_start = int(eval_idx[0]) if len(eval_idx) else 0
                    eval_span = max(1, int(eval_idx[-1]) - eval_start)
                    deep_test_position.extend(
                        ((rows.astype(np.float64) - eval_start) / eval_span).tolist()
                    )
                for name in pos_mechanisms:
                    mechanism_scores.setdefault(name, []).extend(
                        torch.sigmoid(
                            torch.cat([pos_mechanisms[name], neg_mechanisms[name]])
                        ).cpu().tolist()
                    )
                    historical_mechanism_scores.setdefault(name, []).extend(
                        torch.sigmoid(
                            torch.cat(
                                [pos_mechanisms[name], historical_neg_mechanisms[name]]
                            )
                        ).cpu().tolist()
                    )
                    historical_mechanism_margins.setdefault(name, []).extend(
                        (
                            pos_mechanisms[name]
                            - historical_neg_mechanisms[name]
                        ).cpu().tolist()
                    )
                if fact_intervention_enabled:
                    batch_positions = np.arange(
                        fact_intervention_cursor,
                        fact_intervention_cursor + len(rows),
                    )
                    selected_local = np.flatnonzero(
                        fact_intervention_targets[batch_positions]
                    )
                    if len(selected_local):
                        selected_tensor = torch.tensor(
                            selected_local, dtype=torch.long, device=device
                        )

                        def select_batch(values: tuple[torch.Tensor, ...]):
                            return tuple(value.index_select(0, selected_tensor) for value in values)

                        intervention_pos = select_batch(pos)
                        intervention_historical = select_batch(historical_neg)
                        intervention_pos_result = model(*intervention_pos)
                        intervention_historical_result = model(*intervention_historical)
                        pos_bank, pos_index, pos_valid = lifter_component_fact_indices(
                            model, intervention_pos_result, fact_intervention_component
                        )
                        hist_bank, hist_index, hist_valid = lifter_component_fact_indices(
                            model, intervention_historical_result, fact_intervention_component
                        )
                        # Executor trace indices address the internally truncated
                        # H-slot bank. Translate them back to the full input bank.
                        pos_index = pos_index + (
                            intervention_pos[6].shape[1]
                            - intervention_pos_result.aux["predicate_ids_src"].shape[1]
                        )
                        hist_index = hist_index + (
                            intervention_historical[6].shape[1]
                            - intervention_historical_result.aux["predicate_ids_src"].shape[1]
                        )
                        pos_top_batch, pos_random_batch, pos_eligible = delete_lifter_grounded_facts(
                            intervention_pos, pos_bank, pos_index, pos_valid,
                            spec.seed * 1_000_003 + int(rows[0]),
                        )
                        hist_top_batch, hist_random_batch, hist_eligible = delete_lifter_grounded_facts(
                            intervention_historical, hist_bank, hist_index, hist_valid,
                            spec.seed * 1_000_033 + int(rows[0]),
                        )
                        eligible = pos_eligible | hist_eligible
                        if bool(eligible.any()):
                            pos_top_logits = model(*pos_top_batch).logits
                            pos_random_logits = model(*pos_random_batch).logits
                            hist_top_logits = model(*hist_top_batch).logits
                            hist_random_logits = model(*hist_random_batch).logits
                            fact_intervention_original_positive.extend(
                                intervention_pos_result.logits[eligible].cpu().tolist()
                            )
                            fact_intervention_original_negative.extend(
                                intervention_historical_result.logits[eligible].cpu().tolist()
                            )
                            fact_intervention_top_positive.extend(
                                pos_top_logits[eligible].cpu().tolist()
                            )
                            fact_intervention_top_negative.extend(
                                hist_top_logits[eligible].cpu().tolist()
                            )
                            fact_intervention_random_positive.extend(
                                pos_random_logits[eligible].cpu().tolist()
                            )
                            fact_intervention_random_negative.extend(
                                hist_random_logits[eligible].cpu().tolist()
                            )
                            pos_component = (
                                lifter_mechanism_logits(model, intervention_pos_result)[f"only_{fact_intervention_component}"]
                                - intervention_pos_result.aux["prior_rule_contribution"]
                            )
                            hist_component = (
                                lifter_mechanism_logits(model, intervention_historical_result)[f"only_{fact_intervention_component}"]
                                - intervention_historical_result.aux["prior_rule_contribution"]
                            )
                            original_margin = intervention_pos_result.logits - intervention_historical_result.logits
                            for local in torch.nonzero(eligible, as_tuple=False).flatten().tolist():
                                margin_parts = (float(pos_component[local]), -float(hist_component[local]))
                                support_side = int(margin_parts[1] > margin_parts[0])
                                if margin_parts[support_side] > 0:
                                    top_margin = (
                                        pos_top_logits[local] - intervention_historical_result.logits[local]
                                        if support_side == 0 else
                                        intervention_pos_result.logits[local] - hist_top_logits[local]
                                    )
                                    random_margin = (
                                        pos_random_logits[local] - intervention_historical_result.logits[local]
                                        if support_side == 0 else
                                        intervention_pos_result.logits[local] - hist_random_logits[local]
                                    )
                                    fact_intervention_support_top_margin_drop.append(float(original_margin[local] - top_margin))
                                    fact_intervention_support_random_margin_drop.append(float(original_margin[local] - random_margin))
                                oppose_side = int(margin_parts[1] < margin_parts[0])
                                if margin_parts[oppose_side] < 0:
                                    top_margin = (
                                        pos_top_logits[local] - intervention_historical_result.logits[local]
                                        if oppose_side == 0 else
                                        intervention_pos_result.logits[local] - hist_top_logits[local]
                                    )
                                    random_margin = (
                                        pos_random_logits[local] - intervention_historical_result.logits[local]
                                        if oppose_side == 0 else
                                        intervention_pos_result.logits[local] - hist_random_logits[local]
                                    )
                                    fact_intervention_oppose_top_margin_gain.append(float(top_margin - original_margin[local]))
                                    fact_intervention_oppose_random_margin_gain.append(float(random_margin - original_margin[local]))
                pos_top, pos_random, pos_top_delta, pos_random_delta = delete_lifter_proofs(
                    pos_result, deletion_rng
                )
                neg_top, neg_random, neg_top_delta, neg_random_delta = delete_lifter_proofs(
                    neg_result, deletion_rng
                )
                top_deleted_scores.extend(
                    torch.sigmoid(torch.cat([pos_top, neg_top])).cpu().tolist()
                )
                random_deleted_scores.extend(
                    torch.sigmoid(torch.cat([pos_random, neg_random])).cpu().tolist()
                )
                top_proof_logit_deltas.extend(
                    torch.cat([pos_top_delta, neg_top_delta]).cpu().tolist()
                )
                random_proof_logit_deltas.extend(
                    torch.cat([pos_random_delta, neg_random_delta]).cpu().tolist()
                )
            logits = torch.cat([pos_logits, neg_logits])
            labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
            losses.append(float(F.binary_cross_entropy_with_logits(logits, labels).cpu()))
            labels_all.extend(labels.cpu().tolist())
            scores_all.extend(torch.sigmoid(logits).cpu().tolist())
            historical_logits = torch.cat([pos_logits, historical_neg_result.logits])
            historical_labels = torch.cat(
                [torch.ones_like(pos_logits), torch.zeros_like(historical_neg_result.logits)]
            )
            historical_losses.append(
                float(F.binary_cross_entropy_with_logits(historical_logits, historical_labels).cpu())
            )
            historical_labels_all.extend(historical_labels.cpu().tolist())
            historical_scores_all.extend(torch.sigmoid(historical_logits).cpu().tolist())
            historical_selection_all.extend(historical_selected.astype(np.float32).tolist())
            historical_pairwise_all.extend(
                (pos_logits > historical_neg_result.logits).float().cpu().tolist()
            )
            recent_historical_logits = torch.cat(
                [pos_logits, recent_historical_neg_result.logits]
            )
            recent_historical_labels = torch.cat(
                [
                    torch.ones_like(pos_logits),
                    torch.zeros_like(recent_historical_neg_result.logits),
                ]
            )
            recent_historical_labels_all.extend(
                recent_historical_labels.cpu().tolist()
            )
            recent_historical_scores_all.extend(
                torch.sigmoid(recent_historical_logits).cpu().tolist()
            )
            recent_historical_selection_all.extend(
                recent_historical_selected.astype(np.float32).tolist()
            )
            recent_historical_pairwise_all.extend(
                (pos_logits > recent_historical_neg_result.logits).float().cpu().tolist()
            )
            fact_intervention_cursor += len(rows)
    preds = [1.0 if score >= 0.5 else 0.0 for score in scores_all]
    acc = float(np.mean([pred == label for pred, label in zip(preds, labels_all)])) if labels_all else 0.0
    auc = binary_auc(labels_all, scores_all)
    average_precision = binary_average_precision(labels_all, scores_all)
    historical_auc = binary_auc(historical_labels_all, historical_scores_all)
    historical_average_precision = binary_average_precision(
        historical_labels_all, historical_scores_all
    )
    recent_historical_auc = binary_auc(
        recent_historical_labels_all, recent_historical_scores_all
    )
    recent_historical_average_precision = binary_average_precision(
        recent_historical_labels_all, recent_historical_scores_all
    )
    operator_metrics: dict[str, Any] = {
        "aggregation_operator": str(getattr(model, "aggregation_operator", "model_native"))
    }
    if config.get("evaluation_intervention"):
        operator_metrics["evaluation_intervention"] = dict(
            config["evaluation_intervention"]
        )
    kernel_aggregation = getattr(model, "kernel_aggregation", None)
    if kernel_aggregation is not None:
        operator_metrics["initial_recurrence_scales"] = (
            kernel_aggregation.initial_scale_values.detach().cpu().tolist()
        )
        operator_metrics["learned_recurrence_scales"] = (
            torch.exp(kernel_aggregation.log_scales).detach().cpu().tolist()
        )
        operator_metrics["recurrence_scales_trainable"] = bool(
            kernel_aggregation.log_scales.requires_grad
        )
        operator_metrics["recurrence_scale_init"] = str(kernel_aggregation.scale_init)
        operator_metrics["recurrence_evidence_mode"] = str(
            kernel_aggregation.evidence_mode
        )
        operator_metrics["kernel_output_scale"] = float(
            kernel_aggregation.output_scale.detach().cpu()
        )
    target_pool = getattr(model, "target_agnostic_pool", None)
    if target_pool is not None:
        operator_metrics["target_agnostic_scales"] = (
            target_pool.scale_values.detach().cpu().tolist()
        )
        operator_metrics["target_agnostic_include_activity"] = bool(
            target_pool.include_activity
        )
        operator_metrics["target_agnostic_output_scale"] = float(
            target_pool.output_scale.detach().cpu()
        )
    if intrinsic_sufficiency:
        sufficiency_curve = np.mean(np.asarray(intrinsic_sufficiency), axis=0)
        necessity_curve = np.mean(np.asarray(intrinsic_necessity), axis=0)
        count_curve = np.mean(np.asarray(intrinsic_counts), axis=0)
        ratios = np.asarray(explanation_sparsities)
        width = float(ratios[-1] - ratios[0])
        operator_metrics["intrinsic_explanation"] = {
            "explanation_ratio": explanation_sparsities,
            "sufficiency_agreement": sufficiency_curve.tolist(),
            "necessity_probability_delta": necessity_curve.tolist(),
            "avg_selected_events": count_curve.tolist(),
            "sufficiency_auc": float(np.trapz(sufficiency_curve, ratios) / width),
            "necessity_auc": float(np.trapz(necessity_curve, ratios) / width),
            "time_mean_ms": float(np.mean(intrinsic_times_ms)),
            "time_p95_ms": float(np.percentile(intrinsic_times_ms, 95)),
            "timeout_rate": float(np.mean(np.asarray(intrinsic_times_ms) > explanation_timeout_ms)),
            "queries": explanation_seen,
        }
    if predicate_diagnostics_enabled:
        assert predicate_component_mass is not None
        total_mass = float(predicate_component_mass.sum())
        normalized_mass = (
            predicate_component_mass / total_mass
            if total_mass > 0
            else predicate_component_mass
        )
        positive_mass = normalized_mass[normalized_mass > 0]
        effective_rule_count = float(
            np.exp(-(positive_mass * np.log(positive_mass)).sum())
        ) if len(positive_mass) else 0.0
        cumulative = np.cumsum(np.sort(normalized_mass)[::-1])
        rules_for_90pct = int(np.searchsorted(cumulative, 0.9) + 1) if total_mass else 0

        def role_stability(left: str, right: str) -> dict[str, float]:
            left_profile = predicate_role_profiles[left]
            right_profile = predicate_role_profiles[right]
            similarities, weights = [], []
            for predicate in range(int(model.predicate_count)):
                left_row = left_profile[predicate]
                right_row = right_profile[predicate]
                left_norm = float(np.linalg.norm(left_row))
                right_norm = float(np.linalg.norm(right_row))
                if left_norm == 0 or right_norm == 0:
                    continue
                similarities.append(
                    float(np.dot(left_row, right_row) / (left_norm * right_norm))
                )
                weights.append(min(float(left_row.sum()), float(right_row.sum())))
            return {
                "weighted_cosine": (
                    float(np.average(similarities, weights=weights))
                    if similarities and sum(weights) > 0 else float("nan")
                ),
                "comparable_predicates": len(similarities),
                "left_queries": predicate_group_counts[left],
                "right_queries": predicate_group_counts[right],
            }

        operator_metrics["predicate_diagnostics"] = {
            "assignment_mode": str(model.predicate_assignment_mode),
            "execution_mode": str(model.predicate_execution_mode),
            "predicate_count": int(model.predicate_count),
            "mean_active_proofs_per_query": float(np.mean(predicate_query_active_counts)),
            "top_proof_absolute_logit_fraction": {
                str(key): float(np.mean(values))
                for key, values in predicate_top_fraction.items()
            },
            "max_logit_reconstruction_error": float(
                np.max(predicate_reconstruction_errors)
            ),
            "mean_logit_reconstruction_error": float(
                np.mean(predicate_reconstruction_errors)
            ),
            "effective_rule_count": effective_rule_count,
            "rules_for_90pct_absolute_contribution": rules_for_90pct,
            "component_absolute_mass": predicate_component_mass.tolist(),
            "predicate_role_profiles": {
                group: profile.tolist()
                for group, profile in predicate_role_profiles.items()
            },
            "role_reuse": {
                "seen_to_unseen_entity": role_stability("seen_entities", "unseen_entity"),
                "early_to_later_interval": role_stability("early_interval", "later_interval"),
                "source_activity_strata": role_stability(
                    "low_source_activity", "high_source_activity"
                ),
                "destination_popularity_strata": role_stability(
                    "low_destination_popularity", "high_destination_popularity"
                ),
            },
        }
    if lifter_diagnostics_enabled:
        soft_auc = binary_auc(labels_all, soft_scores_all)
        soft_ap = binary_average_precision(labels_all, soft_scores_all)
        mechanism_metrics = {}
        for name, condition_scores in mechanism_scores.items():
            mechanism_metrics[name] = {
                "random_negative_auc": binary_auc(labels_all, condition_scores),
                "random_negative_ap": binary_average_precision(labels_all, condition_scores),
                "historical_negative_auc": binary_auc(
                    historical_labels_all, historical_mechanism_scores[name]
                ),
                "historical_negative_ap": binary_average_precision(
                    historical_labels_all, historical_mechanism_scores[name]
                ),
                "mean_historical_logit_margin": float(
                    np.mean(historical_mechanism_margins[name])
                ),
            }
        top_deleted_ap = binary_average_precision(labels_all, top_deleted_scores)
        random_deleted_ap = binary_average_precision(labels_all, random_deleted_scores)
        operator_metrics.update(
            {
                "architecture": "neuro-symbolic-rule-only",
                "latent_predicate_count": int(model.predicate_count),
                "induced_rule_count": int(model.rule_count),
                "symbolic_regularization": last_symbolic_metrics,
                "soft_projection_auc": soft_auc,
                "soft_projection_ap": soft_ap,
                "hard_projection_auc_gap": float(auc - soft_auc),
                "hard_projection_ap_gap": float(average_precision - soft_ap),
                "hard_projection_mean_abs_logit_gap": float(
                    np.mean(projection_logit_differences)
                ),
                "positive_grounding_coverage": float(
                    np.mean(positive_grounding_valid)
                ),
                "negative_grounding_coverage": float(
                    np.mean(negative_grounding_valid)
                ),
                "induced_rules": model.export_symbolic_rules(),
                "explanation_example": explanation_example,
                "proof_deletion": {
                    "top_mean_abs_logit_delta": float(np.mean(top_proof_logit_deltas)),
                    "random_mean_abs_logit_delta": float(np.mean(random_proof_logit_deltas)),
                    "top_deleted_ap": top_deleted_ap,
                    "random_deleted_ap": random_deleted_ap,
                    "top_ap_drop": float(average_precision - top_deleted_ap),
                    "random_ap_drop": float(average_precision - random_deleted_ap),
                },
                "mechanism_decomposition": mechanism_metrics,
                "historical_failure_diagnosis": {
                    "mean_logit_margin": float(np.mean(historical_logit_margins)),
                    "pairwise_accuracy": float(
                        np.mean(np.asarray(historical_logit_margins) > 0)
                    ),
                    "positive_recurrence_rate": float(
                        np.mean(historical_positive_recurrence)
                    ),
                    "pairwise_accuracy_with_positive_recurrence": float(
                        np.mean(
                            np.asarray(historical_logit_margins)[
                                np.asarray(historical_positive_recurrence) > 0
                            ]
                            > 0
                        )
                    ),
                    "pairwise_accuracy_without_positive_recurrence": float(
                        np.mean(
                            np.asarray(historical_logit_margins)[
                                np.asarray(historical_positive_recurrence) == 0
                            ]
                            > 0
                        )
                    ),
                    "family_mean_logit_margins": {
                        family: float(np.mean(margins))
                        for family, margins in historical_family_margins.items()
                    },
                    "family_positive_margin_rates": {
                        family: float(np.mean(np.asarray(margins) > 0))
                        for family, margins in historical_family_margins.items()
                    },
                },
            }
        )
        if deep_mechanism_enabled:
            operator_metrics["mechanism_query_trace"] = {
                "component_names": list(deep_component_names),
                "rows": deep_rows,
                "positive_prior": deep_positive_prior,
                "historical_prior": deep_historical_prior,
                "positive_components": deep_positive_components,
                "historical_components": deep_historical_components,
                "direct_recurrence": deep_direct_recurrence,
                "source_activity": deep_source_activity,
                "destination_popularity": deep_destination_popularity,
                "test_position": deep_test_position,
            }
        if fact_intervention_enabled:
            original_logits = np.concatenate((
                np.asarray(fact_intervention_original_positive),
                np.asarray(fact_intervention_original_negative),
            ))
            top_logits = np.concatenate((
                np.asarray(fact_intervention_top_positive),
                np.asarray(fact_intervention_top_negative),
            ))
            random_logits = np.concatenate((
                np.asarray(fact_intervention_random_positive),
                np.asarray(fact_intervention_random_negative),
            ))
            intervention_labels = np.concatenate((
                np.ones(len(fact_intervention_original_positive)),
                np.zeros(len(fact_intervention_original_negative)),
            )).tolist()
            original_scores = torch.sigmoid(torch.tensor(original_logits)).tolist()
            top_scores = torch.sigmoid(torch.tensor(top_logits)).tolist()
            random_scores = torch.sigmoid(torch.tensor(random_logits)).tolist()
            original_intervention_auc = binary_auc(intervention_labels, original_scores)
            original_intervention_ap = binary_average_precision(intervention_labels, original_scores)
            top_intervention_auc = binary_auc(intervention_labels, top_scores)
            top_intervention_ap = binary_average_precision(intervention_labels, top_scores)
            random_intervention_auc = binary_auc(intervention_labels, random_scores)
            random_intervention_ap = binary_average_precision(intervention_labels, random_scores)
            top_delta = np.abs(original_logits - top_logits)
            random_delta = np.abs(original_logits - random_logits)
            operator_metrics["grounded_fact_intervention"] = {
                "component": fact_intervention_component,
                "queries": len(fact_intervention_original_positive),
                "matching": "same history bank and recency quartile",
                "original_auc": original_intervention_auc,
                "original_ap": original_intervention_ap,
                "top_fact_deleted_auc": top_intervention_auc,
                "top_fact_deleted_ap": top_intervention_ap,
                "matched_random_deleted_auc": random_intervention_auc,
                "matched_random_deleted_ap": random_intervention_ap,
                "top_auc_drop": original_intervention_auc - top_intervention_auc,
                "top_ap_drop": original_intervention_ap - top_intervention_ap,
                "matched_random_auc_drop": original_intervention_auc - random_intervention_auc,
                "matched_random_ap_drop": original_intervention_ap - random_intervention_ap,
                "top_mean_abs_logit_delta": float(np.mean(top_delta)) if len(top_delta) else float("nan"),
                "matched_random_mean_abs_logit_delta": float(np.mean(random_delta)) if len(random_delta) else float("nan"),
                "delta_ratio": (
                    float(np.mean(top_delta) / np.mean(random_delta))
                    if len(random_delta) and float(np.mean(random_delta)) > 0 else float("nan")
                ),
                "signed_margin_intervention": {
                    "supporting_queries": len(fact_intervention_support_top_margin_drop),
                    "supporting_top_mean_margin_drop": float(np.mean(fact_intervention_support_top_margin_drop)) if fact_intervention_support_top_margin_drop else float("nan"),
                    "supporting_random_mean_margin_drop": float(np.mean(fact_intervention_support_random_margin_drop)) if fact_intervention_support_random_margin_drop else float("nan"),
                    "supporting_direction_consistency": float(np.mean(np.asarray(fact_intervention_support_top_margin_drop) > 0)) if fact_intervention_support_top_margin_drop else float("nan"),
                    "opposing_queries": len(fact_intervention_oppose_top_margin_gain),
                    "opposing_top_mean_margin_gain": float(np.mean(fact_intervention_oppose_top_margin_gain)) if fact_intervention_oppose_top_margin_gain else float("nan"),
                    "opposing_random_mean_margin_gain": float(np.mean(fact_intervention_oppose_random_margin_gain)) if fact_intervention_oppose_random_margin_gain else float("nan"),
                    "opposing_direction_consistency": float(np.mean(np.asarray(fact_intervention_oppose_top_margin_gain) > 0)) if fact_intervention_oppose_top_margin_gain else float("nan"),
                },
            }
    if query_diagnostics_enabled:
        operator_metrics["query_diagnostics"] = {
            "rows": query_diagnostic_rows,
            "positive_destinations": query_diagnostic_positive_destinations,
            "historical_destinations": query_diagnostic_historical_destinations,
            "positive_logits": query_diagnostic_positive_logits,
            "historical_logits": query_diagnostic_historical_logits,
            "historical_selected": query_diagnostic_historical_selected,
        }
    checkpoint_root = config.get("paths", {}).get("checkpoints")
    if checkpoint_root:
        checkpoint_dir = Path(checkpoint_root)
        if not checkpoint_dir.is_absolute():
            checkpoint_dir = Path(__file__).resolve().parents[1] / checkpoint_dir
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{spec.dataset}_{spec.model}_seed{spec.seed}.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "model_name": spec.model,
                "model_config": resolve_auto_model_config(
                    spec.domain, frame, spec.model_config
                ),
                "seed": spec.seed,
                "examples": len(frame),
            },
            checkpoint_path,
        )
        operator_metrics["checkpoint"] = str(checkpoint_path)
    return {
        "train_loss": last_train,
        "eval_loss": float(np.mean(losses)),
        "primary_metric": auc,
        "accuracy": acc,
        "auc": auc,
        "average_precision": average_precision,
        "ap": average_precision,
        "historical_negative_eval_loss": float(np.mean(historical_losses)),
        "historical_negative_auc": historical_auc,
        "historical_negative_average_precision": historical_average_precision,
        "historical_negative_ap": historical_average_precision,
        "historical_negative_coverage": float(np.mean(historical_selection_all)),
        "historical_negative_pairwise_accuracy": float(
            np.mean(historical_pairwise_all)
        ),
        "historical_negative_strategy": str(
            training.get("historical_negative_strategy", "uniform")
        ),
        "most_recent_historical_negative_auc": recent_historical_auc,
        "most_recent_historical_negative_average_precision": (
            recent_historical_average_precision
        ),
        "most_recent_historical_negative_ap": recent_historical_average_precision,
        "most_recent_historical_negative_coverage": float(
            np.mean(recent_historical_selection_all)
        ),
        "most_recent_historical_negative_pairwise_accuracy": float(
            np.mean(recent_historical_pairwise_all)
        ),
        "proof_matched_auc": (
            binary_auc(proof_matched_labels_all, proof_matched_scores_all)
            if proof_matched_labels_all else float("nan")
        ),
        "proof_matched_ap": (
            binary_average_precision(proof_matched_labels_all, proof_matched_scores_all)
            if proof_matched_labels_all else float("nan")
        ),
        "proof_matched_pairwise_accuracy": (
            float(np.mean(proof_matched_pairwise_all))
            if proof_matched_pairwise_all else float("nan")
        ),
        "proof_matched_coverage": (
            float(np.mean(proof_matched_selection_all))
            if proof_matched_selection_all else 0.0
        ),
        "grounded_deletion": {
            "queries": len(grounded_deletion_top_deltas),
            "top_mean_abs_logit_delta": (
                float(np.mean(grounded_deletion_top_deltas))
                if grounded_deletion_top_deltas else float("nan")
            ),
            "random_mean_abs_logit_delta": (
                float(np.mean(grounded_deletion_random_deltas))
                if grounded_deletion_random_deltas else float("nan")
            ),
        },
        "contrastive_explanation": contrastive_explanation,
        "examples": float(len(frame)),
        **operator_metrics,
        **model_audit,
    }


def prepare_sequence_frame(path: Path, max_examples: int) -> pd.DataFrame:
    frame = chronological_sample(pd.read_csv(path), max_examples)
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0.0).astype("float32")
    if "sequence_id" not in frame.columns:
        frame["sequence_id"] = "sequence_0"
    frame["mark_id"], _ = label_encode(frame.get("mark", pd.Series(["event"] * len(frame))))
    return frame


def sequence_windows(frame: pd.DataFrame, history_len: int) -> list[tuple[np.ndarray, np.ndarray, int, float, int]]:
    windows = []
    for _, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values("timestamp")
        marks = group["mark_id"].to_numpy(dtype=np.int64)
        times = group["timestamp"].to_numpy(dtype=np.float32)
        for idx in range(1, len(group)):
            start = max(0, idx - history_len)
            hist_marks = marks[start:idx]
            hist_times = times[start:idx]
            pad = history_len - len(hist_marks)
            windows.append(
                (
                    np.pad(hist_marks, (pad, 0)),
                    np.pad(hist_times, (pad, 0)),
                    int(marks[idx]),
                    float(max(0.0, times[idx] - times[idx - 1])),
                    int(len(hist_marks)),
                )
            )
    return windows


def single_sequence_windows(frame: pd.DataFrame, history_len: int) -> list[tuple[np.ndarray, np.ndarray, int, float, int]]:
    ordered = frame.sort_values("timestamp")
    marks = ordered["mark_id"].to_numpy(dtype=np.int64)
    times = ordered["timestamp"].to_numpy(dtype=np.float32)
    windows = []
    for idx in range(1, len(ordered)):
        start = max(0, idx - history_len)
        hist_marks = marks[start:idx]
        hist_times = times[start:idx]
        pad = history_len - len(hist_marks)
        windows.append(
            (
                np.pad(hist_marks, (pad, 0)),
                np.pad(hist_times, (pad, 0)),
                int(marks[idx]),
                float(max(0.0, times[idx] - times[idx - 1])),
                int(len(hist_marks)),
            )
        )
    return windows


def stpp_attention_targets(
    coords: torch.Tensor,
    target_coords: torch.Tensor,
    mask: torch.Tensor,
    *,
    top_k: int = 3,
) -> torch.Tensor:
    valid = mask.to(torch.bool)
    if not bool(valid.any()):
        return coords.new_zeros(coords.shape[:2])
    distances = torch.linalg.norm(coords - target_coords.unsqueeze(1), dim=-1)
    max_distance = distances.masked_fill(~valid, 0.0).max(dim=1, keepdim=True).values.clamp_min(1.0)
    spatial_score = 1.0 - (distances / max_distance).clamp(0.0, 1.0)
    positions = torch.linspace(0.0, 1.0, steps=coords.shape[1], device=coords.device, dtype=coords.dtype).unsqueeze(0)
    support_score = 0.8 * spatial_score + 0.2 * positions
    support_score = support_score.masked_fill(~valid, -torch.inf)
    k = min(int(top_k), coords.shape[1])
    indices = torch.topk(support_score, k=k, dim=1).indices
    targets = coords.new_zeros(coords.shape[:2])
    targets.scatter_(1, indices, 1.0)
    return targets * valid.to(targets.dtype)


def stpp_last_query_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2:
        return logits
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand_as(mask)
    indices = torch.where(mask, positions, torch.full_like(positions, -1)).max(dim=1).values.clamp_min(0)
    return logits[torch.arange(logits.shape[0], device=logits.device), indices]


def tkg_attention_targets(
    head: torch.Tensor,
    relation: torch.Tensor,
    history_head: torch.Tensor,
    history_relation: torch.Tensor,
    history_tail: torch.Tensor,
    target_tail: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask.to(torch.bool)
    target_hit = (history_tail.long() == target_tail.long().unsqueeze(1)).to(torch.float32)
    query_head_touch = (
        (history_head.long() == head.long().unsqueeze(1))
        | (history_tail.long() == head.long().unsqueeze(1))
    ).to(torch.float32)
    relation_match = (history_relation.long() == relation.long().unsqueeze(1)).to(torch.float32)
    support = torch.maximum(target_hit, query_head_touch * relation_match)
    return support * valid.to(support.dtype)


def build_tkg_histories(
    frame: pd.DataFrame,
    history_len: int,
    *,
    strict_time: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precompute fixed-length TKG histories.

    By default this matches the original benchmark behavior: previous rows in
    the sorted stream are evidence, including earlier rows with the same
    timestamp.  ``strict_time=True`` is available for leakage audits.
    """

    n_rows = len(frame)
    history_len = int(history_len)
    heads = frame["head_id"].to_numpy(dtype=np.int64)
    relations = frame["relation_id"].to_numpy(dtype=np.int64)
    tails = frame["tail_id"].to_numpy(dtype=np.int64)
    times = frame["timestamp_id"].to_numpy(dtype=np.float32)
    hist_heads = np.zeros((n_rows, history_len), dtype=np.int64)
    hist_relations = np.zeros((n_rows, history_len), dtype=np.int64)
    hist_tails = np.zeros((n_rows, history_len), dtype=np.int64)
    hist_times = np.zeros((n_rows, history_len), dtype=np.float32)
    hist_masks = np.zeros((n_rows, history_len), dtype=bool)

    for row, query_time in enumerate(times):
        end = int(np.searchsorted(times, query_time, side="left")) if strict_time else row
        start = max(0, end - history_len)
        length = end - start
        if length <= 0:
            continue
        offset = history_len - length
        hist_heads[row, offset:] = heads[start:end]
        hist_relations[row, offset:] = relations[start:end]
        hist_tails[row, offset:] = tails[start:end]
        hist_times[row, offset:] = times[start:end]
        hist_masks[row, offset:] = True
    return hist_heads, hist_relations, hist_tails, hist_times, hist_masks


def build_strict_tkg_histories(frame: pd.DataFrame, history_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return build_tkg_histories(frame, history_len, strict_time=True)


def stpp_joint_nll(
    result: ModelResult,
    targets: torch.Tensor,
    target_coords: torch.Tensor,
    dtime: torch.Tensor,
    *,
    num_marks: int,
) -> torch.Tensor:
    mark_nll = F.cross_entropy(result.logits, targets) if num_marks > 1 else result.logits.sum() * 0.0
    loc = result.aux["next_location"]
    loc_log_scale = result.aux["location_log_scale"].clamp(-5.0, 5.0)
    loc_var = torch.exp(2.0 * loc_log_scale).view(1, -1)
    location_nll = (
        0.5 * ((target_coords - loc) ** 2 / loc_var).sum(dim=-1)
        + loc_log_scale.sum()
        + math.log(2.0 * math.pi)
    ).mean()

    log_target_time = torch.log1p(dtime.clamp_min(0.0))
    time_log_mean = result.aux["time_log_mean"]
    time_log_scale = result.aux["time_log_scale"].clamp(-5.0, 5.0)
    time_var = torch.exp(2.0 * time_log_scale)
    time_nll = (
        0.5 * ((log_target_time - time_log_mean) ** 2 / time_var)
        + time_log_scale
        + 0.5 * math.log(2.0 * math.pi)
        + torch.log1p(dtime.clamp_min(0.0))
    ).mean()
    return mark_nll + location_nll + time_nll


def run_mtpp(spec: RunSpec, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    training = config["training"]
    frame = prepare_sequence_frame(spec.dataset_path, int(training["max_examples"]))
    windows = sequence_windows(frame, int(training["history_len"]))
    if len(windows) < 2:
        windows = single_sequence_windows(frame, int(training["history_len"]))
    if len(windows) < 2:
        raise ValueError("Not enough MTPP windows")
    train_idx, eval_idx = split_indices(len(windows), float(training["eval_ratio"]))
    model = build_benchmark_model(spec, frame, config, device)
    model_audit = audit_attention_application(model, spec)
    opt = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training.get("weight_decay", 0.0)))

    def make_batch(rows: np.ndarray):
        batch = [windows[int(row)] for row in rows]
        marks = torch.tensor(np.stack([item[0] for item in batch]), device=device)
        times = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32, device=device)
        targets = torch.tensor([item[2] for item in batch], device=device)
        dtime = torch.tensor([item[3] for item in batch], dtype=torch.float32, device=device)
        mask = torch.tensor(np.stack([np.arange(marks.shape[1]) >= marks.shape[1] - item[4] for item in batch]), dtype=torch.bool, device=device)
        return marks, times, targets, dtime, mask

    last_train = 0.0
    for epoch in epoch_iter(spec, config):
        model.train()
        losses = []
        for rows in progress_batches(
            train_idx,
            int(training["batch_size"]),
            shuffle=True,
            desc=f"epoch {epoch + 1}/{int(training['epochs'])} batches",
            config=config,
        ):
            marks, times, targets, dtime, mask = make_batch(rows)
            result = model(marks, times, mask)
            loss = F.cross_entropy(result.logits, targets) + 0.01 * F.smooth_l1_loss(result.aux["next_time"], dtime)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        last_train = float(np.mean(losses)) if losses else 0.0
        if progress_enabled(config):
            tqdm.write(f"{spec.model}/{spec.variant} epoch {epoch + 1}: train_loss={last_train:.6f}")

    model.eval()
    losses, correct, total, maes, reciprocal_ranks = [], 0, 0, [], []
    with torch.no_grad():
        for rows in progress_batches(eval_idx, int(training["batch_size"]), shuffle=False, desc="eval batches", config=config):
            marks, times, targets, dtime, mask = make_batch(rows)
            result = model(marks, times, mask)
            loss = F.cross_entropy(result.logits, targets) + 0.01 * F.smooth_l1_loss(result.aux["next_time"], dtime)
            losses.append(float(loss.cpu()))
            correct += int((result.logits.argmax(dim=-1) == targets).sum().cpu())
            total += int(targets.numel())
            target_logits = result.logits.gather(1, targets.unsqueeze(1))
            ranks = (result.logits > target_logits).sum(dim=1).to(torch.float32) + 1.0
            reciprocal_ranks.extend((1.0 / ranks).cpu().tolist())
            maes.extend(torch.abs(result.aux["next_time"] - dtime).cpu().tolist())
    acc = correct / max(1, total)
    mark_mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan")
    return {
        "train_loss": last_train,
        "eval_loss": float(np.mean(losses)),
        "primary_metric": mark_mrr,
        "accuracy": acc,
        "mark_mrr": mark_mrr,
        "mae_time": float(np.mean(maes)),
        "examples": float(len(windows)),
        **model_audit,
    }


def run_stpp(spec: RunSpec, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    training = config["training"]
    frame = prepare_sequence_frame(spec.dataset_path, int(training["max_examples"]))
    frame["timestamp"] = pd.to_numeric(frame["timestamp"], errors="coerce").fillna(0.0).astype("float64")
    frame["lat"] = pd.to_numeric(frame["lat"], errors="coerce").fillna(0.0).astype("float32")
    frame["lon"] = pd.to_numeric(frame["lon"], errors="coerce").fillna(0.0).astype("float32")
    for column in ("lat", "lon"):
        std = float(frame[column].std() or 1.0)
        frame[column] = (frame[column] - float(frame[column].mean())) / std
    time_origin = float(frame["timestamp"].min())
    time_scale = float(frame["timestamp"].std() or 1.0)
    frame["time_value"] = ((frame["timestamp"] - time_origin) / max(time_scale, 1.0)).astype("float32")

    delta_values: list[float] = []
    for _, group in frame.groupby("sequence_id", sort=False):
        ts = group.sort_values("timestamp")["timestamp"].to_numpy(dtype=np.float64)
        if len(ts) > 1:
            delta_values.extend(np.log1p(np.maximum(0.0, np.diff(ts))).tolist())
    if not delta_values:
        ordered_ts = frame.sort_values("timestamp")["timestamp"].to_numpy(dtype=np.float64)
        if len(ordered_ts) > 1:
            delta_values.extend(np.log1p(np.maximum(0.0, np.diff(ordered_ts))).tolist())
    delta_scale = float(np.mean(delta_values) or 1.0)
    delta_scale = max(delta_scale, 1.0)

    windows = []
    history_len = int(training["history_len"])
    for _, group in frame.groupby("sequence_id", sort=False):
        group = group.sort_values("timestamp")
        marks = group["mark_id"].to_numpy(dtype=np.int64)
        raw_times = group["timestamp"].to_numpy(dtype=np.float64)
        times = group["time_value"].to_numpy(dtype=np.float32)
        coords = group[["lat", "lon"]].to_numpy(dtype=np.float32)
        for idx in range(1, len(group)):
            start = max(0, idx - history_len)
            pad = history_len - (idx - start)
            delta_target = np.log1p(max(0.0, raw_times[idx] - raw_times[idx - 1])) / delta_scale
            windows.append(
                (
                    np.pad(marks[start:idx], (pad, 0)),
                    np.pad(times[start:idx], (pad, 0)),
                    np.pad(coords[start:idx], ((pad, 0), (0, 0))),
                    int(marks[idx]),
                    coords[idx],
                    float(delta_target),
                    idx - start,
                )
            )
    if len(windows) < 2:
        ordered = frame.sort_values("timestamp")
        marks = ordered["mark_id"].to_numpy(dtype=np.int64)
        raw_times = ordered["timestamp"].to_numpy(dtype=np.float64)
        times = ordered["time_value"].to_numpy(dtype=np.float32)
        coords = ordered[["lat", "lon"]].to_numpy(dtype=np.float32)
        windows = []
        for idx in range(1, len(ordered)):
            start = max(0, idx - history_len)
            pad = history_len - (idx - start)
            delta_target = np.log1p(max(0.0, raw_times[idx] - raw_times[idx - 1])) / delta_scale
            windows.append(
                (
                    np.pad(marks[start:idx], (pad, 0)),
                    np.pad(times[start:idx], (pad, 0)),
                    np.pad(coords[start:idx], ((pad, 0), (0, 0))),
                    int(marks[idx]),
                    coords[idx],
                    float(delta_target),
                    idx - start,
                )
            )
    if len(windows) < 2:
        raise ValueError("Not enough STPP windows")
    train_idx, eval_idx = split_indices(len(windows), float(training["eval_ratio"]))
    model = build_benchmark_model(spec, frame, config, device)
    model_audit = audit_attention_application(model, spec)
    opt = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training.get("weight_decay", 0.0)))
    num_marks = int(frame["mark_id"].max()) + 1
    gate_loss_weight = float(spec.model_config.get("gate_loss_weight", config.get("stpp", {}).get("gate_loss_weight", 0.05)))

    def make_batch(rows: np.ndarray):
        batch = [windows[int(row)] for row in rows]
        marks = torch.tensor(np.stack([item[0] for item in batch]), device=device)
        times = torch.tensor(np.stack([item[1] for item in batch]), dtype=torch.float32, device=device)
        coords = torch.tensor(np.stack([item[2] for item in batch]), dtype=torch.float32, device=device)
        targets = torch.tensor([item[3] for item in batch], device=device)
        target_coords = torch.tensor(np.stack([item[4] for item in batch]), dtype=torch.float32, device=device)
        dtime = torch.tensor([item[5] for item in batch], dtype=torch.float32, device=device)
        mask = torch.tensor(np.stack([np.arange(marks.shape[1]) >= marks.shape[1] - item[6] for item in batch]), dtype=torch.bool, device=device)
        return times, coords, marks, targets, target_coords, dtime, mask

    last_train = 0.0
    for epoch in epoch_iter(spec, config):
        model.train()
        losses = []
        for rows in progress_batches(
            train_idx,
            int(training["batch_size"]),
            shuffle=True,
            desc=f"epoch {epoch + 1}/{int(training['epochs'])} batches",
            config=config,
        ):
            times, coords, marks, targets, target_coords, dtime, mask = make_batch(rows)
            result = model(times, coords, marks, mask)
            task_loss = stpp_joint_nll(result, targets, target_coords, dtime, num_marks=num_marks)
            loss = task_loss
            if spec.use_attention and gate_loss_weight > 0.0 and "raw_attention_logits" in result.aux:
                attention_targets = stpp_attention_targets(coords, target_coords, mask)
                attention_logits = stpp_last_query_logits(result.aux["raw_attention_logits"], mask)
                loss = loss + gate_loss_weight * masked_gate_bce(attention_logits, attention_targets, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(task_loss.detach().cpu()))
        last_train = float(np.mean(losses)) if losses else 0.0
        if progress_enabled(config):
            tqdm.write(f"{spec.model}/{spec.variant} epoch {epoch + 1}: train_loss={last_train:.6f}")

    model.eval()
    losses, correct, total, rmses, time_maes = [], 0, 0, [], []
    with torch.no_grad():
        for rows in progress_batches(eval_idx, int(training["batch_size"]), shuffle=False, desc="eval batches", config=config):
            times, coords, marks, targets, target_coords, dtime, mask = make_batch(rows)
            result = model(times, coords, marks, mask)
            task_loss = stpp_joint_nll(result, targets, target_coords, dtime, num_marks=num_marks)
            losses.append(float(task_loss.cpu()))
            if num_marks > 1:
                correct += int((result.logits.argmax(dim=-1) == targets).sum().cpu())
                total += int(targets.numel())
            rmses.extend(torch.sqrt(((result.aux["next_location"] - target_coords) ** 2).sum(dim=-1)).cpu().tolist())
            time_maes.extend(torch.abs(result.aux["next_time"] - dtime).cpu().tolist())
    acc = correct / max(1, total) if total else float("nan")
    rmse_location = float(np.mean(rmses))
    mae_time = float(np.mean(time_maes))
    joint_nll = float(np.mean(losses))
    primary = rmse_location
    return {
        "train_loss": last_train,
        "eval_loss": joint_nll,
        "primary_metric": primary,
        "joint_nll": joint_nll,
        "accuracy": acc,
        "mae_time": mae_time,
        "rmse_location": rmse_location,
        "examples": float(len(windows)),
        **model_audit,
    }


def run_tkg(spec: RunSpec, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    training = config["training"]
    raw_frame = pd.read_csv(spec.dataset_path)
    raw_frame["timestamp"] = pd.to_numeric(raw_frame["timestamp"], errors="coerce").fillna(0.0)
    if "split" in raw_frame.columns:
        raw_frame["split"] = raw_frame["split"].astype(str).str.lower()
    else:
        raw_frame["split"] = "train"
    entity_values = pd.concat([raw_frame["head"], raw_frame["tail"]], ignore_index=True)
    entity_map = {value: idx for idx, value in enumerate(sorted(entity_values.astype(str).unique()))}
    raw_frame["head_id"] = raw_frame["head"].astype(str).map(entity_map).astype("int64")
    raw_frame["tail_id"] = raw_frame["tail"].astype(str).map(entity_map).astype("int64")
    raw_frame["relation_id"], _ = label_encode(raw_frame["relation"])
    raw_frame["timestamp_id"], _ = label_encode(raw_frame["timestamp"])

    max_examples = int(training["max_examples"])
    eval_ratio = float(training["eval_ratio"])
    train_budget = max(1, int(max_examples * (1.0 - eval_ratio)))
    eval_budget = max(1, max_examples - train_budget)
    if "train" in set(raw_frame["split"]):
        train_frame = raw_frame[raw_frame["split"].eq("train")].sort_values("timestamp").tail(train_budget)
        eval_candidates = raw_frame[raw_frame["split"].isin(["test", "val", "valid"])].sort_values("timestamp")
        test_frame = eval_candidates[eval_candidates["split"].eq("test")]
        eval_frame = (test_frame if len(test_frame) else eval_candidates).head(eval_budget)
        frame = pd.concat([train_frame, eval_frame], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        train_idx = frame.index[frame["split"].eq("train")].to_numpy()
        eval_idx = frame.index[~frame["split"].eq("train")].to_numpy()
        if len(train_idx) == 0 or len(eval_idx) == 0:
            frame = chronological_sample(raw_frame, max_examples)
            train_idx, eval_idx = split_indices(len(frame), eval_ratio)
    else:
        frame = chronological_sample(raw_frame, max_examples)
        train_idx, eval_idx = split_indices(len(frame), eval_ratio)
    model = build_benchmark_model(spec, frame, config, device)
    model_audit = audit_attention_application(model, spec)
    lr_key = "attention_learning_rate" if spec.use_attention else "learning_rate"
    learning_rate = float(spec.model_config.get(lr_key, spec.model_config.get("learning_rate", training["learning_rate"])))
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(training.get("weight_decay", 0.0)))
    history_len = int(training["history_len"])
    gate_loss_weight = float(spec.model_config.get("gate_loss_weight", config.get("tkg", {}).get("gate_loss_weight", 0.05)))
    gradient_clip = float(spec.model_config.get("gradient_clip", config.get("tkg", {}).get("gradient_clip", 1.0)))
    head_values = frame["head_id"].to_numpy(dtype=np.int64)
    relation_values = frame["relation_id"].to_numpy(dtype=np.int64)
    time_values = frame["timestamp_id"].to_numpy(dtype=np.float32)
    tail_values = frame["tail_id"].to_numpy(dtype=np.int64)
    strict_time = bool(config.get("tkg", {}).get("strict_time_history", False))
    hist_heads, hist_relations, hist_tails, hist_times, hist_masks = build_tkg_histories(
        frame,
        history_len,
        strict_time=strict_time,
    )

    def make_batch(rows: np.ndarray):
        rows = np.asarray(rows, dtype=np.int64)
        return (
            torch.as_tensor(head_values[rows], device=device),
            torch.as_tensor(relation_values[rows], device=device),
            torch.as_tensor(time_values[rows], dtype=torch.float32, device=device),
            torch.as_tensor(hist_heads[rows], device=device),
            torch.as_tensor(hist_relations[rows], device=device),
            torch.as_tensor(hist_tails[rows], device=device),
            torch.as_tensor(hist_times[rows], dtype=torch.float32, device=device),
            torch.as_tensor(hist_masks[rows], dtype=torch.bool, device=device),
            torch.as_tensor(tail_values[rows], device=device),
        )

    last_train = 0.0
    for epoch in epoch_iter(spec, config):
        model.train()
        losses = []
        for rows in progress_batches(
            train_idx,
            int(training["batch_size"]),
            shuffle=True,
            desc=f"epoch {epoch + 1}/{int(training['epochs'])} batches",
            config=config,
        ):
            *inputs, target = make_batch(rows)
            result = model(*inputs)
            loss = F.cross_entropy(result.logits, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite TKG train loss for {spec.model}/{spec.dataset}/{spec.variant}")
            if "relation_logits" in result.aux:
                loss = loss + 0.05 * F.cross_entropy(result.aux["relation_logits"], inputs[1])
            if spec.use_attention and gate_loss_weight > 0.0 and "raw_attention_logits" in result.aux:
                attention_targets = tkg_attention_targets(inputs[0], inputs[1], inputs[3], inputs[4], inputs[5], target, inputs[7])
                loss = loss + gate_loss_weight * masked_gate_bce(result.aux["raw_attention_logits"], attention_targets, inputs[7])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite TKG total train loss for {spec.model}/{spec.dataset}/{spec.variant}")
            opt.zero_grad()
            loss.backward()
            if gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        last_train = float(np.mean(losses)) if losses else 0.0
        if progress_enabled(config):
            tqdm.write(f"{spec.model}/{spec.variant} epoch {epoch + 1}: train_loss={last_train:.6f}")

    model.eval()
    losses, correct, total, reciprocal_ranks = [], 0, 0, []
    with torch.no_grad():
        for rows in progress_batches(eval_idx, int(training["batch_size"]), shuffle=False, desc="eval batches", config=config):
            *inputs, target = make_batch(rows)
            result = model(*inputs)
            if not torch.isfinite(result.logits).all():
                raise FloatingPointError(f"non-finite TKG eval logits for {spec.model}/{spec.dataset}/{spec.variant}")
            eval_loss = F.cross_entropy(result.logits, target)
            if not torch.isfinite(eval_loss):
                raise FloatingPointError(f"non-finite TKG eval loss for {spec.model}/{spec.dataset}/{spec.variant}")
            losses.append(float(eval_loss.cpu()))
            correct += int((result.logits.argmax(dim=-1) == target).sum().cpu())
            total += int(target.numel())
            target_logits = result.logits.gather(1, target.unsqueeze(1))
            if not torch.isfinite(target_logits).all():
                raise FloatingPointError(f"non-finite TKG target logits for {spec.model}/{spec.dataset}/{spec.variant}")
            ranks = (result.logits > target_logits).sum(dim=1).to(torch.float32) + 1.0
            reciprocal_ranks.extend((1.0 / ranks).cpu().tolist())
    acc = correct / max(1, total)
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan")
    return {
        "train_loss": last_train,
        "eval_loss": float(np.mean(losses)),
        "primary_metric": mrr,
        "accuracy": acc,
        "mrr": mrr,
        "examples": float(len(frame)),
        **model_audit,
    }


def read_jsonl(path: Path, max_examples: int) -> list[dict[str, Any]]:
    examples = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                examples.append(json.loads(line))
            if len(examples) >= max_examples:
                break
    return examples


def tokenize_text(text: Any, vocab_size: int, max_len: int) -> np.ndarray:
    words = str(text).lower().split()[:max_len]
    ids = [hash_token(word, vocab_size - 1) + 1 for word in words]
    return np.pad(np.asarray(ids, dtype=np.int64), (0, max_len - len(ids)))[:max_len]


def rag_context_to_text(context: Any) -> str:
    if isinstance(context, str):
        return context
    if isinstance(context, dict):
        parts: list[str] = []
        title = context.get("title")
        if isinstance(title, list):
            parts.extend(str(item) for item in title)
        elif title is not None:
            parts.append(str(title))
        sentences = context.get("sentences")
        if isinstance(sentences, list):
            for item in sentences:
                if isinstance(item, list):
                    parts.extend(str(sentence) for sentence in item)
                else:
                    parts.append(str(item))
        elif sentences is not None:
            parts.append(str(sentences))
        if parts:
            return " ".join(parts)
    return str(context)


def rag_contexts_to_passages(contexts: Any, *, max_passages: int) -> list[dict[str, str]]:
    if not isinstance(contexts, list):
        contexts = [contexts] if contexts is not None else []
    passages: list[dict[str, str]] = []
    for context in contexts:
        if isinstance(context, dict):
            titles = context.get("title")
            sentences = context.get("sentences")
            if isinstance(titles, list) and isinstance(sentences, list) and len(titles) == len(sentences):
                for title, passage_sentences in zip(titles, sentences):
                    parts = [str(title)]
                    if isinstance(passage_sentences, list):
                        parts.extend(str(sentence) for sentence in passage_sentences)
                    elif passage_sentences is not None:
                        parts.append(str(passage_sentences))
                    text = " ".join(part for part in parts if part)
                    if text:
                        passages.append({"title": str(title), "text": text})
                    if len(passages) >= max_passages:
                        return passages
                continue
        text = rag_context_to_text(context)
        if text.strip():
            title = context.get("title", "") if isinstance(context, dict) else ""
            passages.append({"title": str(title), "text": text})
        if len(passages) >= max_passages:
            break
    return passages[:max_passages]


def rag_contexts_to_passage_texts(contexts: Any, *, max_passages: int) -> list[str]:
    return [passage["text"] for passage in rag_contexts_to_passages(contexts, max_passages=max_passages)]


def normalize_rag_text(text: Any) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text).lower())).strip()


def rag_support_titles(example: dict[str, Any]) -> set[str]:
    facts = example.get("supporting_facts") or example.get("supporting")
    titles: set[str] = set()
    if isinstance(facts, dict):
        raw_titles = facts.get("title") or facts.get("titles")
        if isinstance(raw_titles, list):
            titles.update(normalize_rag_text(title) for title in raw_titles)
        elif raw_titles is not None:
            titles.add(normalize_rag_text(raw_titles))
    elif isinstance(facts, list):
        for fact in facts:
            if isinstance(fact, (list, tuple)) and fact:
                titles.add(normalize_rag_text(fact[0]))
            elif isinstance(fact, dict) and fact.get("title") is not None:
                titles.add(normalize_rag_text(fact["title"]))
            elif isinstance(fact, str):
                titles.add(normalize_rag_text(fact))
    return {title for title in titles if title}


def rag_support_labels(example: dict[str, Any], passages: list[dict[str, str]], *, max_passages: int) -> np.ndarray:
    labels = np.zeros(max_passages, dtype=np.float32)
    support_titles = rag_support_titles(example)
    if support_titles:
        for idx, passage in enumerate(passages[:max_passages]):
            if normalize_rag_text(passage.get("title", "")) in support_titles:
                labels[idx] = 1.0
        if labels.any():
            return labels

    answers = example.get("answers") or [example.get("answer", "")]
    if not isinstance(answers, list):
        answers = [answers]
    normalized_answers = [normalize_rag_text(answer) for answer in answers if normalize_rag_text(answer)]
    for idx, passage in enumerate(passages[:max_passages]):
        normalized_text = normalize_rag_text(passage.get("text", ""))
        if normalized_text and any(answer and answer in normalized_text for answer in normalized_answers):
            labels[idx] = 1.0
    if labels.any() or not passages:
        return labels

    question_tokens = set(normalize_rag_text(example.get("question", "")).split())
    best_idx, best_score = 0, -1.0
    for idx, passage in enumerate(passages[:max_passages]):
        passage_tokens = set(normalize_rag_text(passage.get("text", "")).split())
        if not passage_tokens:
            continue
        score = len(question_tokens & passage_tokens) / max(1, len(question_tokens))
        if score > best_score:
            best_idx, best_score = idx, score
    if best_score > 0.0:
        labels[best_idx] = 1.0
    return labels


def masked_support_nll(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    usable_examples = mask.any(dim=1) & labels.bool().any(dim=1)
    if not bool(usable_examples.any()):
        return (logits * 0.0).sum()
    selected_logits = logits[usable_examples].masked_fill(~mask[usable_examples], torch.finfo(logits.dtype).min)
    selected_labels = labels[usable_examples] * mask[usable_examples].to(labels.dtype)
    target_dist = selected_labels / selected_labels.sum(dim=1, keepdim=True).clamp_min(1.0)
    log_probs = F.log_softmax(selected_logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1).mean()


def masked_gate_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    usable = mask & labels.bool().any(dim=1, keepdim=True)
    if not bool(usable.any()):
        return (logits * 0.0).sum()
    selected_labels = labels[usable]
    positives = selected_labels.sum().clamp_min(1.0)
    negatives = (1.0 - selected_labels).sum().clamp_min(1.0)
    pos_weight = (negatives / positives).detach()
    return F.binary_cross_entropy_with_logits(logits[usable], selected_labels, pos_weight=pos_weight)


def support_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    attention_mass: torch.Tensor | None = None,
) -> dict[str, float]:
    usable_examples = mask.any(dim=1) & labels.bool().any(dim=1)
    if not bool(usable_examples.any()):
        return {
            "support_precision": float("nan"),
            "support_recall": float("nan"),
            "support_f1": float("nan"),
            "support_recall_at_1": float("nan"),
            "support_recall_at_2": float("nan"),
            "support_recall_at_5": float("nan"),
            "support_mrr": float("nan"),
            "support_ap": float("nan"),
            "support_auc": float("nan"),
            "support_attention_mass": float("nan"),
            "distractor_attention_mass": float("nan"),
            "support_mass_ratio": float("nan"),
        }

    logits = logits.detach().float().cpu()
    labels = labels.detach().float().cpu()
    mask = mask.detach().bool().cpu()
    if attention_mass is None:
        attention_mass = logits.exp()
    attention_mass = attention_mass.detach().float().cpu()
    recall_hits = {1: 0.0, 2: 0.0, 5: 0.0}
    set_precisions = []
    set_recalls = []
    set_f1s = []
    reciprocal_ranks = []
    average_precisions = []
    auc_labels: list[float] = []
    auc_scores: list[float] = []
    support_masses = []
    distractor_masses = []
    mass_ratios = []
    for row_idx, (row_logits, row_labels, row_mask) in enumerate(zip(logits, labels, mask)):
        if not bool(row_mask.any()) or not bool(row_labels.bool().any()):
            continue
        scores = row_logits.masked_fill(~row_mask, -float("inf"))
        order = torch.argsort(scores, descending=True)
        gold_count = int((row_labels > 0.5).sum())
        pred_count = min(max(1, gold_count), int(row_mask.sum()))
        predicted = order[:pred_count]
        true_positives = float((row_labels[predicted] > 0.5).sum())
        precision = true_positives / max(1, pred_count)
        recall = true_positives / max(1, gold_count)
        set_precisions.append(precision)
        set_recalls.append(recall)
        set_f1s.append(0.0 if precision + recall <= 0.0 else 2.0 * precision * recall / (precision + recall))
        for k in recall_hits:
            topk = order[: min(k, int(row_mask.sum()))]
            recall_hits[k] += float(bool((row_labels[topk] > 0.5).any()))
        positive_ranks = torch.nonzero(row_labels[order] > 0.5, as_tuple=False)
        if positive_ranks.numel():
            reciprocal_ranks.append(1.0 / float(int(positive_ranks[0, 0]) + 1))
            precisions = []
            positives_seen = 0.0
            for rank, idx in enumerate(order.tolist(), start=1):
                if not bool(row_mask[idx]):
                    continue
                if row_labels[idx] > 0.5:
                    positives_seen += 1.0
                    precisions.append(positives_seen / float(rank))
            if precisions:
                average_precisions.append(float(np.mean(precisions)))
        row_mass = attention_mass[row_idx] if attention_mass.ndim == 2 else row_logits.exp()
        mass = row_mass * row_mask.to(row_mass.dtype)
        support_mass = float((mass * row_labels).sum())
        distractor_mass = float((mass * (1.0 - row_labels) * row_mask.to(row_labels.dtype)).sum())
        support_masses.append(support_mass)
        distractor_masses.append(distractor_mass)
        mass_ratios.append(support_mass / max(1.0e-8, support_mass + distractor_mass))
        for score, label, valid in zip(row_logits.tolist(), row_labels.tolist(), row_mask.tolist()):
            if valid:
                auc_scores.append(float(score))
                auc_labels.append(float(label))
    examples = int(usable_examples.sum().cpu())
    return {
        "support_precision": float(np.mean(set_precisions)) if set_precisions else float("nan"),
        "support_recall": float(np.mean(set_recalls)) if set_recalls else float("nan"),
        "support_f1": float(np.mean(set_f1s)) if set_f1s else float("nan"),
        "support_recall_at_1": recall_hits[1] / max(1, examples),
        "support_recall_at_2": recall_hits[2] / max(1, examples),
        "support_recall_at_5": recall_hits[5] / max(1, examples),
        "support_mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan"),
        "support_ap": float(np.mean(average_precisions)) if average_precisions else float("nan"),
        "support_auc": binary_auc(auc_labels, auc_scores),
        "support_attention_mass": float(np.mean(support_masses)) if support_masses else float("nan"),
        "distractor_attention_mass": float(np.mean(distractor_masses)) if distractor_masses else float("nan"),
        "support_mass_ratio": float(np.mean(mass_ratios)) if mass_ratios else float("nan"),
    }


def run_rag(spec: RunSpec, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    training = config["training"]
    examples = read_jsonl(spec.dataset_path, int(training["max_examples"]))
    if len(examples) < 2:
        raise ValueError("Not enough RAG examples")
    vocab_size = int(spec.model_config.get("vocab_size", 8192))
    train_idx, eval_idx = split_indices(len(examples), float(training["eval_ratio"]))
    frame_stub = {"vocab_size": vocab_size}
    model = build_benchmark_model(spec, frame_stub, config, device)
    model_audit = audit_attention_application(model, spec)
    opt = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training.get("weight_decay", 0.0)))

    max_passages = int(spec.model_config.get("max_passages", 4))
    question_len = int(spec.model_config.get("question_len", 32))
    passage_len = int(spec.model_config.get("passage_len", 64))
    batch_size = int(spec.model_config.get("batch_size", config.get("rag", {}).get("batch_size", training["batch_size"])))
    eval_batch_size = int(spec.model_config.get("eval_batch_size", config.get("rag", {}).get("eval_batch_size", batch_size)))
    support_loss_weight = float(spec.model_config.get("support_loss_weight", config.get("rag", {}).get("support_loss_weight", 1.0)))
    gate_loss_weight = float(spec.model_config.get("gate_loss_weight", config.get("rag", {}).get("gate_loss_weight", 0.0)))

    question_cache = np.zeros((len(examples), question_len), dtype=np.int64)
    passage_cache = np.zeros((len(examples), max_passages, passage_len), dtype=np.int64)
    mask_cache = np.zeros((len(examples), max_passages), dtype=bool)
    support_cache = np.zeros((len(examples), max_passages), dtype=np.float32)
    for idx, example in enumerate(examples):
        question_cache[idx] = tokenize_text(example.get("question", ""), vocab_size, question_len)
        passage_records = rag_contexts_to_passages(example.get("contexts"), max_passages=max_passages)
        for passage_idx, passage in enumerate(passage_records[:max_passages]):
            tokens = tokenize_text(passage["text"], vocab_size, passage_len)
            passage_cache[idx, passage_idx] = tokens
            mask_cache[idx, passage_idx] = bool(np.any(tokens))
        support_cache[idx] = rag_support_labels(example, passage_records, max_passages=max_passages)

    def make_batch(rows: np.ndarray):
        return (
            torch.as_tensor(question_cache[rows], device=device),
            torch.as_tensor(passage_cache[rows], device=device),
            torch.as_tensor(mask_cache[rows], dtype=torch.bool, device=device),
            torch.as_tensor(support_cache[rows], dtype=torch.float32, device=device),
        )

    last_train = 0.0
    for epoch in epoch_iter(spec, config):
        model.train()
        losses = []
        for rows in progress_batches(
            train_idx,
            batch_size,
            shuffle=True,
            desc=f"epoch {epoch + 1}/{int(training['epochs'])} batches",
            config=config,
        ):
            question, passages, mask, support_target = make_batch(rows)
            result = model(question, passages, mask)
            support_loss = masked_support_nll(result.aux["passage_logits"], support_target, mask)
            loss = support_loss_weight * support_loss
            if spec.use_attention and gate_loss_weight > 0.0:
                loss = loss + gate_loss_weight * masked_gate_bce(result.aux["raw_attention_logits"], support_target, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        last_train = float(np.mean(losses)) if losses else 0.0
        if progress_enabled(config):
            tqdm.write(f"{spec.model}/{spec.variant} epoch {epoch + 1}: train_loss={last_train:.6f}")

    model.eval()
    losses = []
    support_metric_rows: list[dict[str, float]] = []
    with torch.no_grad():
        for rows in progress_batches(eval_idx, eval_batch_size, shuffle=False, desc="eval batches", config=config):
            question, passages, mask, support_target = make_batch(rows)
            result = model(question, passages, mask)
            support_loss = masked_support_nll(result.aux["passage_logits"], support_target, mask)
            loss = support_loss_weight * support_loss
            if spec.use_attention and gate_loss_weight > 0.0:
                loss = loss + gate_loss_weight * masked_gate_bce(result.aux["raw_attention_logits"], support_target, mask)
            losses.append(float(loss.cpu()))
            support_metric_rows.append(
                support_metrics(
                    result.aux["passage_logits"],
                    support_target,
                    mask,
                    result.aux.get("attention_mass"),
                )
            )

    def finite_mean(values: list[float]) -> float:
        finite = [value for value in values if not np.isnan(value)]
        return float(np.mean(finite)) if finite else float("nan")

    support_precision = finite_mean([row["support_precision"] for row in support_metric_rows])
    support_recall = finite_mean([row["support_recall"] for row in support_metric_rows])
    support_f1 = finite_mean([row["support_f1"] for row in support_metric_rows])
    support_recall_at_1 = finite_mean([row["support_recall_at_1"] for row in support_metric_rows])
    support_recall_at_2 = finite_mean([row["support_recall_at_2"] for row in support_metric_rows])
    support_recall_at_5 = finite_mean([row["support_recall_at_5"] for row in support_metric_rows])
    support_mrr = finite_mean([row["support_mrr"] for row in support_metric_rows])
    support_ap = finite_mean([row["support_ap"] for row in support_metric_rows])
    support_auc = finite_mean([row["support_auc"] for row in support_metric_rows])
    support_mass = finite_mean([row["support_attention_mass"] for row in support_metric_rows])
    distractor_mass = finite_mean([row["distractor_attention_mass"] for row in support_metric_rows])
    support_mass_ratio = finite_mean([row["support_mass_ratio"] for row in support_metric_rows])
    primary = support_mrr
    return {
        "train_loss": last_train,
        "eval_loss": float(np.mean(losses)),
        "primary_metric": primary,
        "accuracy": float("nan"),
        "answer_accuracy": float("nan"),
        "support_precision": support_precision,
        "support_recall": support_recall,
        "support_f1": support_f1,
        "support_recall_at_1": support_recall_at_1,
        "support_recall_at_2": support_recall_at_2,
        "support_recall_at_5": support_recall_at_5,
        "support_mrr": support_mrr,
        "support_ap": support_ap,
        "support_auc": support_auc,
        "support_attention_mass": support_mass,
        "distractor_attention_mass": distractor_mass,
        "support_mass_ratio": support_mass_ratio,
        "examples": float(len(examples)),
        **model_audit,
    }


RUNNERS = {
    "ctdg": run_ctdg,
    "mtpp": run_mtpp,
    "stpp": run_stpp,
    "tkg": run_tkg,
    "rag": run_rag,
}


def evaluate_run(spec: RunSpec, config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Train/evaluate one already-selected benchmark run and return metrics."""

    if not spec.dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {spec.dataset_path}")
    return RUNNERS[spec.domain](spec, config, device)
