#!/opt/conda/bin/python
"""Run the complete five-way LiFTER predicate-necessity experiment."""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
PYTHON = "/opt/conda/bin/python"
DATASETS = ("wikipedia", "reddit", "mooc", "lastfm")
VARIANTS = ("single_type", "random_types", "kmeans_types", "learned_soft", "learned_hard")
SEEDS = (7, 17, 27)


def path_for(dataset: str, variant: str, seed: int) -> Path:
    return HERE / "outputs" / variant / f"{dataset}_seed{seed}.json"


def complete(path: Path, epochs: int) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text())
        return int(result["epochs"]) == epochs and "predicate_diagnostics" in result["metrics"]
    except (KeyError, ValueError, TypeError, json.JSONDecodeError):
        return False


def run(cell: tuple[str, str, int], epochs: int, progress: bool) -> None:
    dataset, variant, seed = cell
    command = [
        PYTHON, str(HERE / "run_cell.py"), "--dataset", dataset,
        "--variant", variant, "--seed", str(seed), "--epochs", str(epochs),
    ]
    if progress:
        command.append("--progress")
    print(f"START {dataset}/{variant}/seed{seed}", flush=True)
    subprocess.run(command, check=True)
    print(f"DONE {dataset}/{variant}/seed{seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    pending = []
    for dataset in args.datasets:
        for variant in args.variants:
            for seed in args.seeds:
                output = path_for(dataset, variant, seed)
                if not args.force and complete(output, args.epochs):
                    print(f"SKIP {dataset}/{variant}/seed{seed}", flush=True)
                else:
                    pending.append((dataset, variant, seed))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run, cell, args.epochs, args.progress) for cell in pending]
        for future in as_completed(futures):
            future.result()
    subprocess.run([PYTHON, str(HERE / "summarize.py")], check=True)


if __name__ == "__main__":
    main()
