#!/opt/conda/bin/python
"""Run and summarize global grounding-budget selection for LiFTER."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import statistics
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
PYTHON = "/opt/conda/bin/python"
DATASETS = ("wikipedia", "reddit", "mooc", "lastfm")
HORIZONS = (10, 20, 40, 80)
SEEDS = (7, 17, 27)


def output_path(dataset: str, horizon: int, seed: int) -> Path:
    return HERE / "outputs" / f"{dataset}_H{horizon}_seed{seed}.json"


def complete(path: Path, epochs: int) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text())
        return (
            int(result["epochs"]) == epochs
            and int(result["predicate_count"]) == 1
            and result["selection_split"]["held_out_test_used"] is False
            and "historical_negative_auc" in result["metrics"]
        )
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return False


def run_cell(cell: tuple[str, int, int], epochs: int, progress: bool) -> None:
    dataset, horizon, seed = cell
    command = [
        PYTHON, str(HERE / "run_cell.py"), "--dataset", dataset,
        "--horizon", str(horizon), "--seed", str(seed),
        "--epochs", str(epochs),
    ]
    if progress:
        command.append("--progress")
    print(f"START {dataset}/H{horizon}/seed{seed}", flush=True)
    subprocess.run(command, check=True)
    print(f"DONE {dataset}/H{horizon}/seed{seed}", flush=True)


def summarize(datasets: tuple[str, ...], horizons: tuple[int, ...], seeds: tuple[int, ...], epochs: int) -> dict:
    rows = []
    for horizon in horizons:
        dataset_means = []
        for dataset in datasets:
            runs = [json.loads(output_path(dataset, horizon, seed).read_text()) for seed in seeds]
            aucs = [float(run["metrics"]["historical_negative_auc"]) for run in runs]
            aps = [float(run["metrics"]["historical_negative_ap"]) for run in runs]
            dataset_means.append(statistics.mean(aucs))
            rows.append({
                "horizon": horizon,
                "dataset": dataset,
                "historical_auc_mean": statistics.mean(aucs),
                "historical_auc_sample_std": statistics.stdev(aucs),
                "historical_ap_mean": statistics.mean(aps),
                "historical_ap_sample_std": statistics.stdev(aps),
            })
        for row in rows:
            if row["horizon"] == horizon:
                row["macro_historical_auc"] = statistics.mean(dataset_means)
    macro = {
        horizon: statistics.mean(
            row["historical_auc_mean"] for row in rows if row["horizon"] == horizon
        )
        for horizon in horizons
    }
    selected = min(horizons, key=lambda horizon: (-macro[horizon], horizon))
    summary = {
        "protocol": {
            "candidate_horizons": list(horizons),
            "datasets": list(datasets),
            "seeds": list(seeds),
            "epochs": epochs,
            "selection_metric": "macro mean validation Historical AUC",
            "tie_break": "smaller horizon",
            "held_out_test_used": False,
        },
        "selected_horizon": selected,
        "macro_historical_auc": {str(key): value for key, value in macro.items()},
        "rows": rows,
    }
    destination = HERE / "results" / "summary.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--horizons", nargs="+", type=int, default=list(HORIZONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    pending = []
    for dataset in args.datasets:
        for horizon in args.horizons:
            for seed in args.seeds:
                path = output_path(dataset, horizon, seed)
                if not args.force and complete(path, args.epochs):
                    print(f"SKIP {dataset}/H{horizon}/seed{seed}", flush=True)
                    continue
                pending.append((dataset, horizon, seed))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(run_cell, cell, args.epochs, args.progress)
            for cell in pending
        ]
        for future in as_completed(futures):
            future.result()
    summary = summarize(tuple(args.datasets), tuple(args.horizons), tuple(args.seeds), args.epochs)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
