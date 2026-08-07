#!/opt/conda/bin/python
from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
PYTHON = "/opt/conda/bin/python"
DATASETS = ("wikipedia", "reddit", "mooc", "lastfm")
SEEDS = (7, 17, 27)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cells = [(dataset, seed) for dataset in DATASETS for seed in SEEDS]

    def run(cell: tuple[str, int]) -> str:
        dataset, seed = cell
        output = HERE / "outputs" / f"{dataset}_seed{seed}.json"
        if output.exists() and not args.force:
            try:
                import json
                if "query_trace" in json.loads(output.read_text()):
                    return f"SKIP {dataset}/seed{seed}"
            except (OSError, ValueError):
                pass
        command = [
            PYTHON,
            str(HERE / "run_cell.py"),
            "--dataset",
            dataset,
            "--seed",
            str(seed),
        ]
        if args.progress:
            command.append("--progress")
        subprocess.run(command, check=True)
        return f"DONE {dataset}/seed{seed}"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, cell) for cell in cells]
        for future in as_completed(futures):
            print(future.result(), flush=True)
    subprocess.run([PYTHON, str(HERE / "summarize.py")], check=True)
    subprocess.run([PYTHON, str(HERE / "analyze_deep.py")], check=True)


if __name__ == "__main__":
    main()
