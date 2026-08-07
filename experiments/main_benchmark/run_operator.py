#!/opt/conda/bin/python
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DATASETS = ("wikipedia", "reddit", "mooc", "lastfm")
SEEDS = (7, 17, 27)
PYTHON = Path("/opt/conda/bin/python")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce multi-scale kernel measure aggregation.")
    parser.add_argument("--max-examples", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    summary = []
    results_dir = EXPERIMENT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for dataset in DATASETS:
            command = [
                str(PYTHON),
                str(EXPERIMENT_DIR / "run.py"),
                "--dataset",
                dataset,
                "--model",
                "DyGFormer",
                "--variant",
                "base",
                "--max-examples",
                str(args.max_examples),
                "--history-len",
                "320",
                "--neural-history-len",
                "20",
                "--aggregation-operator",
                "multiscale_kernel",
                "--epochs",
                str(args.epochs),
                "--seed",
                str(seed),
            ]
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            source = (
                EXPERIMENT_DIR
                / "outputs"
                / f"{dataset}_DyGFormer_base_seed{seed}.json"
            )
            target = results_dir / f"{dataset}_seed{seed}.json"
            shutil.copy2(source, target)
            metrics = json.loads(target.read_text(encoding="utf-8"))["metrics"]
            summary.append(
                {
                    "seed": seed,
                    "dataset": dataset,
                    "ap": metrics["ap"],
                    "auc": metrics["auc"],
                    "accuracy": metrics["accuracy"],
                    "learned_recurrence_scales": metrics["learned_recurrence_scales"],
                }
            )
    aggregate = {}
    for dataset in DATASETS:
        rows = [row for row in summary if row["dataset"] == dataset]
        aggregate[dataset] = {
            "ap_mean": statistics.mean(row["ap"] for row in rows),
            "ap_std": statistics.pstdev(row["ap"] for row in rows),
            "auc_mean": statistics.mean(row["auc"] for row in rows),
            "auc_std": statistics.pstdev(row["auc"] for row in rows),
        }

    report = {
        "protocol": {
            "python": str(PYTHON),
            "model": "DyGFormer",
            "datasets": list(DATASETS),
            "seeds": list(SEEDS),
            "max_examples": args.max_examples,
            "epochs": args.epochs,
            "history_len": 320,
            "neural_history_len": 20,
            "aggregation_operator": "multiscale_kernel",
            "recurrence_scale_count": 4,
            "recurrence_scale_init": "full",
            "learnable_recurrence_scales": False,
        },
        "runs": summary,
        "aggregate": aggregate,
    }
    output = results_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
