#!/opt/conda/bin/python
"""Train and evaluate one controlled LiFTER predicate-typing variant."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.cluster import MiniBatchKMeans


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import (
    RunSpec,
    build_lifter_causal_fact_context,
    evaluate_run,
    read_csv_tail,
)


VARIANTS = (
    "single_type",
    "random_types",
    "kmeans_types",
    "learned_soft",
    "learned_hard",
)


def fit_kmeans_typing(frame: pd.DataFrame, clusters: int, seed: int) -> tuple[list, list, list]:
    train_end = min(max(1, int(len(frame) * 0.85)), len(frame) - 1)
    raw_columns = [column for column in frame if str(column).startswith("feat_")]
    raw = (
        frame[raw_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    context = build_lifter_causal_fact_context(frame, 8).reshape(len(frame), -1)
    body = np.concatenate((raw, context), axis=1)[:train_end]
    outgoing = np.concatenate((np.ones((train_end, 1), dtype=np.float32), body), axis=1)
    incoming = outgoing.copy()
    incoming[:, 0] = -1.0
    features = np.concatenate((outgoing, incoming), axis=0)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (features - mean) / scale
    estimator = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=seed,
        batch_size=2048,
        n_init=10,
        max_iter=200,
    ).fit(normalized)
    return estimator.cluster_centers_.tolist(), mean.tolist(), scale.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("wikipedia", "reddit", "mooc", "lastfm"))
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    config_path = REPO_ROOT / "experiments" / "main_benchmark" / "config_lifter.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_path = REPO_ROOT / "data" / "processed" / "ctdg" / args.dataset / "events.csv"
    frame = read_csv_tail(dataset_path, int(config["training"]["max_examples"]))
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    model_config = config["model_config"]
    if args.variant == "single_type":
        model_config.update(
            predicate_count=1,
            predicate_assignment_mode="random_fixed",
            predicate_execution_mode="hard",
        )
    elif args.variant == "random_types":
        model_config.update(
            predicate_count=8,
            predicate_assignment_mode="random_fixed",
            predicate_execution_mode="hard",
        )
    elif args.variant == "kmeans_types":
        centroids, mean, scale = fit_kmeans_typing(frame, 8, args.seed)
        model_config.update(
            predicate_count=8,
            predicate_assignment_mode="kmeans_fixed",
            predicate_execution_mode="hard",
            fixed_predicate_centroids=centroids,
            fixed_predicate_mean=mean,
            fixed_predicate_scale=scale,
        )
    elif args.variant == "learned_soft":
        model_config.update(
            predicate_count=8,
            predicate_assignment_mode="learned",
            predicate_execution_mode="soft",
            sparse_ternary_execution=False,
            sparse_renewal_execution=False,
        )
    else:
        model_config.update(
            predicate_count=8,
            predicate_assignment_mode="learned",
            predicate_execution_mode="hard",
        )

    config.update(dataset=args.dataset, model="LiFTER", seed=args.seed)
    config["training"]["epochs"] = args.epochs
    config["training"]["progress"] = args.progress
    config["explanation_evaluation"]["max_queries"] = 0
    config["lifter_diagnostics"]["enabled"] = False
    config["predicate_diagnostics"] = {"enabled": True}

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = RunSpec(
        domain="ctdg",
        dataset=args.dataset,
        dataset_path=dataset_path,
        model="LiFTER",
        variant="base",
        seed=args.seed,
        implementation="reference",
        model_config=dict(model_config),
        use_attention=False,
    )
    started = time.perf_counter()
    metrics = evaluate_run(spec, config, device)
    elapsed = time.perf_counter() - started
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "variant": args.variant,
        "seed": args.seed,
        "epochs": args.epochs,
        "typing_control": {
            "predicate_count": int(model_config["predicate_count"]),
            "assignment_mode": model_config["predicate_assignment_mode"],
            "execution_mode": model_config["predicate_execution_mode"],
            "future_link_supervises_typing": args.variant in {"learned_soft", "learned_hard"},
        },
        "elapsed_seconds": elapsed,
        "metrics": metrics,
    }
    output = HERE / "outputs" / args.variant / f"{args.dataset}_seed{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
