#!/opt/conda/bin/python
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.evaluate import LiFTERHistoryIndex, build_lifter_causal_fact_context
from models.lifter import LiFTER


def synchronize() -> None:
    if torch.cuda.is_available(): torch.cuda.synchronize()


def timed(call, warmup: int = 3, repeats: int = 10) -> float:
    for _ in range(warmup): call()
    synchronize(); started = time.perf_counter()
    for _ in range(repeats): call()
    synchronize(); return 1000 * (time.perf_counter() - started) / repeats


def make_model(h: int, k: int, sparse: bool, device: torch.device, rank: int = 32) -> LiFTER:
    return LiFTER(num_nodes=12000, edge_feat_dim=49, fact_context_len=8, hidden_dim=64,
        predicate_count=k, max_grounding_facts=h, max_rule_length=2,
        latent_transition_rule=True, latent_second_order_transition_rule=True,
        recurrence_guarded_transitions=True, predicate_conditioned_transitions=True,
        positioned_recurrence_rules=True, transition_dim=rank,
        sparse_renewal_execution=sparse, predicate_execution_mode="hard").to(device).eval()


def make_batch(batch: int, h: int, device: torch.device):
    generator = torch.Generator(device=device).manual_seed(12345 + h + batch)
    src = torch.randint(0, 4000, (batch,), generator=generator, device=device)
    dst = torch.randint(4000, 8000, (batch,), generator=generator, device=device)
    timestamp = torch.full((batch,), 10000.0, device=device)
    src_nodes = torch.randint(4000, 8000, (batch, h), generator=generator, device=device)
    dst_nodes = torch.randint(0, 4000, (batch, h), generator=generator, device=device)
    times = torch.arange(h, device=device).float()[None].expand(batch, -1) + 9000
    src_features = torch.randn(batch, h, 49, generator=generator, device=device); src_features[..., 0] = 1
    dst_features = torch.randn(batch, h, 49, generator=generator, device=device); dst_features[..., 0] = -1
    mask = torch.ones(batch, h, dtype=torch.bool, device=device)
    return (src, dst, timestamp, src_nodes, times, src_features, mask, dst_nodes, times.clone(), dst_features, mask.clone())


def executor_sweeps(device: torch.device) -> dict:
    rows = []
    for h in (10, 20, 40, 80):
        for k in (1, 2, 4, 8):
            batch_size = 32 if h >= 40 or k >= 4 else 128
            model, batch = make_model(h, k, True, device), make_batch(batch_size, h, device)
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            latency = timed(lambda: model(*batch), repeats=8)
            result = model(*batch)
            active = (result.aux["rule_grounding_valid"].sum(1)
                      + result.aux["transition_grounding_valid"].long()
                      + result.aux["transition2_grounding_valid"].long())
            peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
            rows.append({"H": h, "K": k, "batch": batch_size, "latency_ms": latency,
                         "queries_per_second": 1000 * batch_size / latency, "peak_gpu_mib": peak,
                         "rule_count": model.rule_count,
                         "mean_active_groundings": float(active.float().mean())})
    sparse_comparison = []
    for h in (10, 20, 40, 80):
        for sparse in (True, False):
            model, batch = make_model(h, 4, sparse, device), make_batch(32, h, device)
            latency = timed(lambda: model(*batch), repeats=8)
            sparse_comparison.append({"H": h, "K": 4, "executor": "sparse" if sparse else "naive", "latency_ms": latency})
    candidates = []
    model, base = make_model(10, 1, True, device), make_batch(128, 10, device)
    for count in (1, 2, 4, 8, 16):
        batches = []
        for candidate in range(count):
            item = list(base); item[1] = (base[1] + candidate) % model.num_nodes; batches.append(tuple(item))
        latency = timed(lambda: model.score_candidate_batches(*batches), repeats=8)
        candidates.append({"candidates": count, "queries": 128, "latency_ms": latency, "candidate_scores_per_second": 1000 * 128 * count / latency})
    batches = []
    for batch_size in (32, 128, 512, 2048):
        model, batch = make_model(10, 1, True, device), make_batch(batch_size, 10, device)
        latency = timed(lambda: model(*batch), repeats=8)
        batches.append({"batch": batch_size, "latency_ms": latency, "queries_per_second": 1000*batch_size/latency})
    ranks = []
    for rank in (8, 16, 32, 64):
        model, batch = make_model(10, 1, True, device, rank), make_batch(128, 10, device)
        latency = timed(lambda: model(*batch), repeats=8)
        ranks.append({"transition_rank": rank, "latency_ms": latency, "queries_per_second": 128000/latency})
    return {"executor": rows, "sparse_vs_naive": sparse_comparison, "candidate_scaling": candidates,
            "batch_scaling": batches, "transition_rank_scaling": ranks}


def training_and_explanation(device: torch.device) -> dict:
    model = make_model(10, 1, True, device).train(); batch = make_batch(512, 10, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    def step():
        result = model(*batch); loss = torch.nn.functional.softplus(-result.logits).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    step_ms = timed(step, warmup=2, repeats=10)
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    model.eval(); small = tuple(value[:32] for value in batch)
    prediction_ms = timed(lambda: model(*small), repeats=20)
    explanation_ms = timed(lambda: model.explain(*small), repeats=10)
    def delete_and_reexecute():
        edited = list(small); edited[6] = edited[6].clone(); edited[6][:, -1] = False
        return model(*edited)
    deletion_ms = timed(delete_and_reexecute, repeats=20)
    traces = model.explain(*small)
    sizes = [len(json.dumps(trace, separators=(",", ":"))) for trace in traces]
    throughput = 1000 * 512 / step_ms
    return {"training_events_per_second": throughput, "training_step_ms": step_ms,
            "estimated_32768_event_epoch_seconds": 32768 / throughput,
            "training_peak_gpu_mib": peak,
            "process_peak_cpu_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "prediction_ms_per_query": prediction_ms / 32,
            "prediction_plus_trace_ms_per_query": explanation_ms / 32,
            "fact_deletion_reexecution_ms_per_query": deletion_ms / 32,
            "trace_overhead_ratio": explanation_ms / prediction_ms,
            "mean_trace_bytes": float(np.mean(sizes)), "p95_trace_bytes": float(np.percentile(sizes, 95))}


def database_scaling() -> list[dict]:
    path = ROOT / "data/processed/ctdg/reddit/events.csv"
    frame = pd.read_csv(path, usecols=["src", "dst", "timestamp"])
    rows = []
    for n in (32768, 131072, 524288, len(frame)):
        subset = frame.iloc[:min(n, len(frame))].copy().reset_index(drop=True)
        subset["src"] = pd.to_numeric(subset.src).astype("int64"); subset["dst"] = pd.to_numeric(subset.dst).astype("int64"); subset["timestamp"] = pd.to_numeric(subset.timestamp).astype("float32")
        started = time.perf_counter(); context = build_lifter_causal_fact_context(subset, 8)
        index = LiFTERHistoryIndex(subset, 128, int(max(subset.src.max(), subset.dst.max())) + 1, context)
        elapsed = time.perf_counter() - started
        payload = context.nbytes + sum(array.nbytes for collection in (index.node_rows, index.node_neighbors, index.node_times, index.node_directions) for array in collection)
        rows.append({"events": len(subset), "nodes": len(index.node_rows), "index_seconds": elapsed,
                     "events_per_second": len(subset) / elapsed, "fact_database_mib": payload / 2**20,
                     "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
    return rows


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {"device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
              "database_scaling": database_scaling(), **executor_sweeps(device),
              "training_and_explanation": training_and_explanation(device)}
    output = HERE / "results" / "summary.json"; output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n"); print(f"saved {output}")


if __name__ == "__main__": main()
