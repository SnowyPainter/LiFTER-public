#!/opt/conda/bin/python
"""Run the complete JODIE predicate-count and shuffle validation."""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = "/opt/conda/bin/python"


def complete(path: Path) -> bool:
    try:
        value = json.loads(path.read_text())
        return (
            value["training"]["epochs"] == 10
            and value["training"]["max_examples"] == 32768
            and "predicate_diagnostics" in value["normal"]
            and "historical_negative_auc" in value["predicate_shuffle"]
            and "historical_negative_auc" in value["global_predicate_shuffle"]
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["wikipedia", "reddit", "mooc", "lastfm"])
    parser.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    cells = []
    for dataset in args.datasets:
        for k in args.ks:
            for seed in args.seeds:
                path = HERE / "outputs" / f"{dataset}_k{k}_seed{seed}.json"
                if not args.force and complete(path):
                    print(f"SKIP {dataset}/K={k}/seed{seed}", flush=True)
                else:
                    cells.append((dataset, k, seed))

    def run(cell):
        dataset, k, seed = cell
        print(f"START {dataset}/K={k}/seed{seed}", flush=True)
        command = [PYTHON, str(HERE / "run_cell.py"), "--dataset", dataset, "--k", str(k), "--seed", str(seed)]
        if args.progress:
            command.append("--progress")
        subprocess.run(command, check=True)
        print(f"DONE {dataset}/K={k}/seed{seed}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, cell) for cell in cells]
        for future in as_completed(futures):
            future.result()
    subprocess.run([PYTHON, str(HERE / "summarize.py")], check=True)


if __name__ == "__main__":
    main()
