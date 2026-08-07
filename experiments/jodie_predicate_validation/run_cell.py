#!/opt/conda/bin/python
"""Train one JODIE K-sweep cell and evaluate a predicate-shuffle intervention."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate import RunSpec, evaluate_run


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_spec(dataset: str, seed: int, model_config: dict) -> RunSpec:
    return RunSpec(
        domain="ctdg",
        dataset=dataset,
        dataset_path=ROOT / "data" / "processed" / "ctdg" / dataset / "events.csv",
        model="LiFTER",
        variant="base",
        seed=seed,
        implementation="reference",
        model_config=model_config,
        use_attention=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("wikipedia", "reddit", "mooc", "lastfm"))
    parser.add_argument("--k", required=True, type=int, choices=(1, 2, 4, 8))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(
        (ROOT / "experiments" / "main_benchmark" / "config_lifter.yaml").read_text()
    )
    config.update(dataset=args.dataset, model="LiFTER", seed=args.seed)
    config["training"]["progress"] = args.progress
    config["training"]["historical_negative_strategy"] = "uniform"
    config["model_config"]["predicate_count"] = args.k
    config["model_config"]["predicate_assignment_mode"] = (
        "random_fixed" if args.k == 1 else "learned"
    )
    config["model_config"]["predicate_execution_mode"] = "hard"
    config["predicate_diagnostics"] = {"enabled": True}
    config["query_diagnostics"] = {"enabled": True}
    config["explanation_evaluation"]["max_queries"] = 0
    checkpoint_dir = HERE / "checkpoints" / f"{args.dataset}_k{args.k}_seed{args.seed}"
    config["paths"]["checkpoints"] = str(checkpoint_dir)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = run_spec(args.dataset, args.seed, dict(config["model_config"]))
    normal = evaluate_run(spec, config, device)
    checkpoint = normal["checkpoint"]

    shuffled_config = copy.deepcopy(config)
    shuffled_config["training"]["epochs"] = 0
    shuffled_config["training"]["progress"] = False
    shuffled_config["paths"].pop("checkpoints", None)
    shuffled_config["paths"]["load_checkpoint"] = checkpoint
    shuffled_config["model_config"]["predicate_intervention"] = "shuffle_facts"
    shuffled_spec = run_spec(
        args.dataset, args.seed, dict(shuffled_config["model_config"])
    )
    seed_everything(args.seed)
    shuffled = evaluate_run(shuffled_spec, shuffled_config, device)

    global_config = copy.deepcopy(shuffled_config)
    global_config["model_config"]["predicate_intervention"] = "shuffle_global"
    global_spec = run_spec(args.dataset, args.seed, dict(global_config["model_config"]))
    seed_everything(args.seed)
    global_shuffled = evaluate_run(global_spec, global_config, device)

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "predicate_count": args.k,
        "seed": args.seed,
        "training": {
            "epochs": int(config["training"]["epochs"]),
            "max_examples": int(config["training"]["max_examples"]),
            "history_len": int(config["training"]["history_len"]),
            "objective": str(config["training"].get("objective", "binary_cross_entropy")),
        },
        "normal": normal,
        "predicate_shuffle": shuffled,
        "global_predicate_shuffle": global_shuffled,
    }
    output = HERE / "outputs" / f"{args.dataset}_k{args.k}_seed{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
