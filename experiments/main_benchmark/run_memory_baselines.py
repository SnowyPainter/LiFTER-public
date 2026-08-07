#!/opt/conda/bin/python
"""Evaluate parameter-free edge-memory baselines on the main protocol."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation.evaluate import (
    CTDGHistoryIndex,
    binary_auc,
    binary_average_precision,
    read_csv_tail,
    sample_historical_destinations,
    split_indices,
)


HERE = Path(__file__).resolve().parent
DATASETS = ("wikipedia", "reddit", "mooc", "lastfm")
SEEDS = (7, 17, 27)
MODELS = ("EdgeBank",)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def score_pair(
    last_seen: dict[tuple[int, int], float],
    source: int,
    destination: int,
    timestamp: float,
    model: str,
) -> float:
    previous = last_seen.get((source, destination))
    return float(previous is not None)


def evaluate(dataset: str, model: str, seed: int) -> dict:
    seed_all(seed)
    frame = read_csv_tail(
        ROOT / f"data/processed/ctdg/{dataset}/events.csv", 32768
    ).sort_values("timestamp", kind="stable").reset_index(drop=True)
    for column in ("src", "dst"):
        frame[column] = pd.to_numeric(frame[column]).astype("int64")
    frame["timestamp"] = pd.to_numeric(frame["timestamp"]).astype("float64")
    _, evaluation = split_indices(len(frame), 0.15)

    node_count = int(max(frame.src.max(), frame.dst.max())) + 1
    history = CTDGHistoryIndex(frame, 320, node_count)
    sources = frame.iloc[evaluation].src.to_numpy(np.int64)
    positives = frame.iloc[evaluation].dst.to_numpy(np.int64)
    history_nodes, _, history_mask = history.gather(sources, evaluation)
    destination_pool = frame.dst.drop_duplicates().to_numpy(np.int64)
    negatives, selected = sample_historical_destinations(
        sources,
        positives,
        history_nodes,
        history_mask,
        destination_pool,
        strategy="uniform",
    )
    random_negatives = np.random.choice(
        destination_pool, size=len(evaluation), replace=True
    ).astype(np.int64)
    invalid = (random_negatives == sources) | (random_negatives == positives)
    while invalid.any():
        random_negatives[invalid] = np.random.choice(
            destination_pool, size=int(invalid.sum()), replace=True
        )
        invalid = (random_negatives == sources) | (random_negatives == positives)

    last_seen: dict[tuple[int, int], float] = {}
    positive_scores: list[float] = []
    negative_scores: list[float] = []
    random_negative_scores: list[float] = []
    evaluation_set = set(evaluation.tolist())
    for row, event in frame.iterrows():
        source, destination = int(event.src), int(event.dst)
        timestamp = float(event.timestamp)
        if row in evaluation_set:
            offset = row - int(evaluation[0])
            positive_scores.append(score_pair(
                last_seen, source, int(positives[offset]), timestamp, model
            ))
            negative_scores.append(score_pair(
                last_seen, source, int(negatives[offset]), timestamp, model
            ))
            random_negative_scores.append(score_pair(
                last_seen, source, int(random_negatives[offset]), timestamp, model
            ))
        last_seen[(source, destination)] = timestamp

    labels = [1.0] * len(positive_scores) + [0.0] * len(negative_scores)
    scores = positive_scores + negative_scores
    random_scores = positive_scores + random_negative_scores
    metrics = {
        "historical_negative_auc": binary_auc(labels, scores),
        "historical_negative_ap": binary_average_precision(labels, scores),
        "historical_negative_pairwise_accuracy": float(np.mean(
            np.asarray(positive_scores) > np.asarray(negative_scores)
        )),
        "historical_negative_coverage": float(np.mean(selected)),
        "historical_negative_strategy": "uniform",
        "auc": binary_auc(labels, random_scores),
        "ap": binary_average_precision(labels, random_scores),
    }
    result = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "variant": "base",
        "implementation": "parameter-free",
        "config": {"training": {"max_examples": 32768, "epochs": 0}},
        "metrics": metrics,
    }
    destination = HERE / "outputs" / f"{dataset}_{model}_base_seed{seed}.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                result = evaluate(dataset, model, seed)
                metrics = result["metrics"]
                print(
                    f"{dataset}/{model}/seed{seed}: "
                    f"AUC={metrics['historical_negative_auc']:.4f} "
                    f"AP={metrics['historical_negative_ap']:.4f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
