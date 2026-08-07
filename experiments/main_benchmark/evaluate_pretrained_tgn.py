#!/opt/conda/bin/python
"""Evaluate the existing 10-epoch TGN checkpoints without retraining them."""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluation.evaluate import RunSpec, evaluate_run


DATASETS = ("wikipedia", "reddit", "mooc", "lastfm")
SEEDS = (7, 17, 27)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in args.datasets:
        for seed in args.seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
            checkpoint = ROOT / "experiments/main_benchmark/checkpoints" / f"{dataset}_TGN_seed{seed}.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            config = yaml.safe_load((HERE / "config_comparators.yaml").read_text())
            config.update(dataset=dataset, model="TGN", seed=seed)
            # The checkpoint has already completed ten epochs.  evaluate_run
            # performs only the full chronological evaluation in this process.
            config["training"].update(epochs=0, progress=args.progress)
            config["paths"]["load_checkpoint"] = str(checkpoint)
            config.setdefault("explanation_evaluation", {})["max_queries"] = 0
            config.setdefault("lifter_diagnostics", {})["enabled"] = False
            spec = RunSpec(
                "ctdg", dataset,
                ROOT / f"data/processed/ctdg/{dataset}/events.csv",
                "TGN", "base", seed, "reference", dict(config["model_config"]), False,
            )
            metrics = evaluate_run(spec, config, device)
            # Report the checkpoint's actual training provenance separately
            # from this evaluation-only invocation.
            config["training"]["epochs"] = 10
            config["training"]["checkpoint_training_batch_size"] = int(
                config["training"]["batch_size"]
            )
            config["training"]["evaluation_only"] = True
            result = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "device": str(device), "dataset": dataset, "model": "TGN",
                "seed": seed, "variant": "base", "implementation": "reference",
                "checkpoint": str(checkpoint), "config": config, "metrics": metrics,
            }
            output = HERE / "outputs" / f"{dataset}_TGN_base_seed{seed}.json"
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                f"{dataset}/TGN/seed{seed}: "
                f"AUC={metrics['historical_negative_auc']:.4f} "
                f"AP={metrics['historical_negative_ap']:.4f}", flush=True,
            )


if __name__ == "__main__":
    main()
