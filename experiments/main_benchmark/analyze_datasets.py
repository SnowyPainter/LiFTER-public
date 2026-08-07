#!/opt/conda/bin/python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import binary_auc, binary_average_precision, read_csv_tail


DATASETS = ("wikipedia", "reddit", "mooc", "lastfm")


def entropy(values: pd.Series) -> float:
    probabilities = values.value_counts(normalize=True).to_numpy(dtype=np.float64)
    return float(-(probabilities * np.log2(probabilities)).sum())


def normalized_entropy(values: pd.Series) -> float:
    unique = values.nunique()
    return entropy(values) / math.log2(unique) if unique > 1 else 0.0


def gini(values: pd.Series) -> float:
    counts = np.sort(values.value_counts().to_numpy(dtype=np.float64))
    if len(counts) == 0 or counts.sum() == 0:
        return float("nan")
    index = np.arange(1, len(counts) + 1)
    return float((2 * (index * counts).sum()) / (len(counts) * counts.sum()) - (len(counts) + 1) / len(counts))


def quantiles(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {}
    return {
        "p25": float(np.quantile(array, 0.25)),
        "p50": float(np.quantile(array, 0.50)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
        "mean": float(array.mean()),
    }


def rank_metrics(labels: list[float], scores: list[float]) -> dict[str, float]:
    return {
        "ap": binary_average_precision(labels, scores),
        "auc": binary_auc(labels, scores),
    }


def analyze(dataset: str, max_examples: int, seed: int, history_len: int) -> dict[str, Any]:
    path = REPO_ROOT / "data" / "processed" / "ctdg" / dataset / "events.csv"
    frame = read_csv_tail(path, max_examples)
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    n = len(frame)
    train_end = min(max(1, int(n * 0.85)), n - 1)
    train = frame.iloc[:train_end]
    evaluation = frame.iloc[train_end:]
    train_src = set(train["src"])
    train_dst = set(train["dst"])
    train_pairs = set(zip(train["src"], train["dst"]))

    dst_counts = Counter(train["dst"])
    pair_counts = Counter(zip(train["src"], train["dst"]))
    source_counts = Counter(train["src"])
    last_pair_time: dict[tuple[int, int], float] = {}
    last_dst_time: dict[int, float] = {}
    longest_history = 320
    source_histories: dict[int, deque[int]] = defaultdict(lambda: deque(maxlen=longest_history))
    for row in train.itertuples():
        last_pair_time[(int(row.src), int(row.dst))] = float(row.timestamp)
        last_dst_time[int(row.dst)] = float(row.timestamp)
        source_histories[int(row.src)].append(int(row.dst))

    rng = np.random.default_rng(seed)
    dst_pool = frame["dst"].drop_duplicates().to_numpy(dtype=np.int64)
    labels: list[float] = []
    heuristic_scores: dict[str, list[float]] = defaultdict(list)
    categories = Counter()
    source_gaps: list[float] = []
    destination_gaps: list[float] = []
    last_source_event: dict[int, float] = {}
    last_destination_event = dict(last_dst_time)

    for row in evaluation.itertuples():
        src, dst, timestamp = int(row.src), int(row.dst), float(row.timestamp)
        pair = (src, dst)
        history = source_histories[src]
        categories["events"] += 1
        categories["seen_pair"] += int(pair in pair_counts)
        categories["new_pair"] += int(pair not in pair_counts)
        categories["new_src"] += int(src not in train_src)
        categories["new_dst"] += int(dst not in train_dst)
        for window in (5, 10, 20, 40, 80, 160, 320):
            categories[f"history_hit_at_{window}"] += int(dst in list(history)[-window:])
        categories["history_hit"] += int(dst in list(history)[-history_len:])
        categories["empty_history"] += int(not history)
        if src in last_source_event:
            source_gaps.append(timestamp - last_source_event[src])
        if dst in last_destination_event:
            destination_gaps.append(timestamp - last_destination_event[dst])

        negative = int(rng.choice(dst_pool))
        while negative in {src, dst}:
            negative = int(rng.choice(dst_pool))
        for candidate, label in ((dst, 1.0), (negative, 0.0)):
            candidate_pair = (src, candidate)
            labels.append(label)
            heuristic_scores["destination_popularity"].append(math.log1p(dst_counts[candidate]))
            heuristic_scores["pair_frequency"].append(math.log1p(pair_counts[candidate_pair]))
            heuristic_scores["history_membership"].append(float(candidate in list(history)[-history_len:]))
            pair_gap = timestamp - last_pair_time.get(candidate_pair, timestamp + 1.0e12)
            heuristic_scores["pair_recency"].append(-math.log1p(max(0.0, pair_gap)) if candidate_pair in last_pair_time else -30.0)
            dst_gap = timestamp - last_dst_time.get(candidate, timestamp + 1.0e12)
            heuristic_scores["destination_recency"].append(-math.log1p(max(0.0, dst_gap)) if candidate in last_dst_time else -30.0)

        pair_counts[pair] += 1
        dst_counts[dst] += 1
        source_counts[src] += 1
        last_pair_time[pair] = timestamp
        last_dst_time[dst] = timestamp
        last_source_event[src] = timestamp
        last_destination_event[dst] = timestamp
        history.append(dst)

    feature_columns = [column for column in frame if column.startswith("feat_")]
    feature_stats: dict[str, Any] = {"dimensions": len(feature_columns)}
    if feature_columns:
        features = frame[feature_columns].to_numpy(dtype=np.float32)
        variances = np.nanvar(features, axis=0)
        feature_stats.update(
            {
                "nonconstant_dimensions": int((variances > 1.0e-12).sum()),
                "mean_variance": float(np.nanmean(variances)),
                "zero_fraction": float(np.nanmean(np.nan_to_num(features) == 0.0)),
            }
        )

    return {
        "dataset": dataset,
        "events": n,
        "train_events": len(train),
        "eval_events": len(evaluation),
        "nodes": int(pd.concat([frame["src"], frame["dst"]]).nunique()),
        "sources": int(frame["src"].nunique()),
        "destinations": int(frame["dst"].nunique()),
        "unique_pairs": int(frame[["src", "dst"]].drop_duplicates().shape[0]),
        "repeat_fraction": float(1.0 - frame[["src", "dst"]].drop_duplicates().shape[0] / n),
        "events_per_source": quantiles(frame["src"].value_counts().to_numpy()),
        "events_per_destination": quantiles(frame["dst"].value_counts().to_numpy()),
        "destination_entropy_bits": entropy(frame["dst"]),
        "destination_normalized_entropy": normalized_entropy(frame["dst"]),
        "destination_degree_gini": gini(frame["dst"]),
        "top_1_destination_share": float(frame["dst"].value_counts(normalize=True).iloc[:1].sum()),
        "top_10_destination_share": float(frame["dst"].value_counts(normalize=True).iloc[:10].sum()),
        "evaluation_structure": {
            key: float(value / max(1, categories["events"]))
            for key, value in categories.items()
            if key != "events"
        },
        "source_interevent_gap": quantiles(source_gaps),
        "destination_interevent_gap": quantiles(destination_gaps),
        "features": feature_stats,
        "heuristics": {name: rank_metrics(labels, scores) for name, scores in heuristic_scores.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-examples", type=int, default=32768)
    parser.add_argument("--history-len", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    results = [analyze(dataset, args.max_examples, args.seed, args.history_len) for dataset in DATASETS]
    output = EXPERIMENT_DIR / "results" / "dataset_analysis.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
