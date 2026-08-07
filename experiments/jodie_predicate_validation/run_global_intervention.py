#!/opt/conda/bin/python
"""Add global predicate-shuffle evaluation to already-trained K-sweep cells."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate import RunSpec, evaluate_run


def one(path: Path) -> None:
    result = json.loads(path.read_text())
    if "global_predicate_shuffle" in result:
        return
    dataset, k, seed = result["dataset"], result["predicate_count"], result["seed"]
    config = yaml.safe_load((ROOT / "experiments/main_benchmark/config_lifter.yaml").read_text())
    config.update(dataset=dataset, model="LiFTER", seed=seed)
    config["training"].update(epochs=0, progress=False, historical_negative_strategy="uniform")
    config["model_config"].update(
        predicate_count=k,
        predicate_assignment_mode="random_fixed" if k == 1 else "learned",
        predicate_execution_mode="hard",
        predicate_intervention="shuffle_global",
    )
    config["predicate_diagnostics"] = {"enabled": True}
    config["query_diagnostics"] = {"enabled": True}
    config["explanation_evaluation"]["max_queries"] = 0
    checkpoint = HERE / "checkpoints" / f"{dataset}_k{k}_seed{seed}" / f"{dataset}_LiFTER_seed{seed}.pt"
    config["paths"].pop("checkpoints", None)
    config["paths"]["load_checkpoint"] = str(checkpoint)
    spec = RunSpec("ctdg", dataset, ROOT / f"data/processed/ctdg/{dataset}/events.csv", "LiFTER", "base", seed, "reference", dict(config["model_config"]), False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    metrics = evaluate_run(spec, config, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    result["global_predicate_shuffle"] = metrics
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"DONE {dataset}/K={k}/seed{seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    paths = sorted((HERE / "outputs").glob("*.json"))
    # Evaluation uses a deliberately reset global NumPy RNG to guarantee that
    # every intervention sees the exact same historical alternatives. Keep
    # these checkpoint-only evaluations sequential; threads would share and
    # race that RNG state.
    for path in paths:
        if args.force:
            result = json.loads(path.read_text())
            result.pop("global_predicate_shuffle", None)
            path.write_text(json.dumps(result, indent=2) + "\n")
        one(path)


if __name__ == "__main__": main()
