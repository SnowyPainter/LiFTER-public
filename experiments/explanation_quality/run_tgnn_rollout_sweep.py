#!/opt/conda/bin/python
"""T-GNNExplainer rollout--fidelity sweep on the shared evidence universe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import run_experiment as base
from evaluation.evaluate import (
    CTDGHistoryIndex,
    RunSpec,
    build_benchmark_model,
    read_csv_tail,
    sample_historical_destinations,
    split_indices,
)
from models.explainability import TGNNExplainer


ROLLOUTS = (40, 100, 200)
SEEDS = (7, 17, 27)


def run(dataset: str, queries: int) -> dict:
    base.DATASET = dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = read_csv_tail(
        base.ROOT / f"data/processed/ctdg/{dataset}/events.csv", 32768
    )
    frame["src"] = pd.to_numeric(frame.src).astype("int64")
    frame["dst"] = pd.to_numeric(frame.dst).astype("int64")
    frame["timestamp"] = pd.to_numeric(frame.timestamp).astype("float32")
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    train, evaluation = split_indices(len(frame), 0.15)
    rows = evaluation[np.linspace(0, len(evaluation) - 1, queries, dtype=np.int64)]
    destinations = frame.dst.drop_duplicates().to_numpy(np.int64)
    node_count = int(max(frame.src.max(), frame.dst.max())) + 1

    sampling_index = CTDGHistoryIndex(frame, 128, node_count)
    history_nodes, _, history_mask = sampling_index.gather(
        frame.src.to_numpy(np.int64), np.arange(len(frame), dtype=np.int64)
    )
    positive_dst = frame.iloc[rows].dst.to_numpy(np.int64)
    negative_dst, _ = sample_historical_destinations(
        frame.iloc[rows].src.to_numpy(np.int64),
        positive_dst,
        history_nodes[rows],
        history_mask[rows],
        destinations,
        strategy="uniform",
    )
    shared_index = CTDGHistoryIndex(
        frame, base.HISTORY_PER_ENDPOINT, node_count
    )
    records = []
    for seed in SEEDS:
        base.seed_all(seed)
        config = {
            "num_nodes": "auto",
            "edge_feat_dim": 1,
            "hidden_dim": 64,
            "time_dim": 32,
            "dropout": 0.1,
        }
        checkpoint = base.train_checkpoint("TGN", seed, config, device)
        positive, _ = base.make_index_and_batches(
            frame,
            rows,
            positive_dst,
            base.HISTORY_PER_ENDPOINT,
            device=device,
            index=shared_index,
        )
        negative, _ = base.make_index_and_batches(
            frame,
            rows,
            negative_dst,
            base.HISTORY_PER_ENDPOINT,
            device=device,
            index=shared_index,
        )
        spec = RunSpec(
            "ctdg",
            dataset,
            base.ROOT / f"data/processed/ctdg/{dataset}/events.csv",
            "TGN",
            "base",
            seed,
            "reference",
            config,
            False,
        )
        predictor = build_benchmark_model(
            spec, frame, {"model_config": config, "attention": {}}, device
        )
        predictor.load_state_dict(
            torch.load(checkpoint, map_location=device, weights_only=False)["model"]
        )
        predictor.eval()

        explainer = TGNNExplainer(
            predictor,
            rollout=max(ROLLOUTS),
            min_events=1,
            exploration=2.0,
            sparsity_weight=0.05,
        ).to(device)
        optimizer = torch.optim.AdamW(explainer.navigator.parameters(), lr=2e-3)
        for start in range(0, min(1024, len(train)), 64):
            training_rows = train[start : start + 64]
            training_dst = frame.iloc[training_rows].dst.to_numpy(np.int64)
            batch, _ = base.make_index_and_batches(
                frame,
                training_rows,
                training_dst,
                base.HISTORY_PER_ENDPOINT,
                device=device,
                index=shared_index,
            )
            loss = explainer.navigator_training_loss(base.dictionary(batch))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        explainer.eval()

        for rollout in ROLLOUTS:
            explainer.rollout = rollout

            def importance(batch):
                return explainer.explain(base.dictionary(batch), top_k=1).event_scores

            metrics = base.evaluate_explanations(
                predictor, positive, negative, importance, device,
                compute_stability=False,
            )
            records.append({"dataset": dataset, "seed": seed, "rollout": rollout, **metrics})
            print(
                f"{dataset} seed={seed} rollout={rollout}: "
                f"ACC-AUC={metrics['acc_auc']:.4f} "
                f"AUFSC={metrics['deletion_aufsc']:.4f} "
                f"time={metrics['explanation_ms_per_query']:.2f} ms/query",
                flush=True,
            )

    summary = {}
    for rollout in ROLLOUTS:
        selected = [row for row in records if row["rollout"] == rollout]
        summary[str(rollout)] = {
            key: {
                "mean": float(np.mean([row[key] for row in selected])),
                "sample_std": float(np.std([row[key] for row in selected], ddof=1)),
            }
            for key in ("acc_auc", "deletion_aufsc", "explanation_ms_per_query")
        }
    return {
        "dataset": dataset,
        "queries_per_seed": queries,
        "seeds": list(SEEDS),
        "history_per_endpoint": base.HISTORY_PER_ENDPOINT,
        "rollouts": list(ROLLOUTS),
        "records": records,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("wikipedia", "reddit", "mooc", "lastfm"),
        default=("wikipedia", "reddit", "mooc", "lastfm"),
    )
    parser.add_argument("--queries", type=int, default=256)
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent / "results" / "tgnn_rollout_sweep.json"
    results = []
    for dataset in args.datasets:
        results.append(run(dataset, args.queries))
        destination.write_text(json.dumps({"datasets": results}, indent=2) + "\n")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
