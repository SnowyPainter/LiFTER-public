#!/opt/conda/bin/python
"""Evaluate one trained K=1 LiFTER through exact mechanism masks."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate import RunSpec, evaluate_run

DOMINANT_COMPONENT = {
    "wikipedia": "pair_renewal",
    "reddit": "pair_renewal",
    "mooc": "ordered_transition_2",
    "lastfm": "ordered_transition_1",
    "retailrocket": "candidate_context",
}


def checkpoint(dataset: str, seed: int) -> Path:
    if dataset == "retailrocket":
        return (
            ROOT
            / "experiments/retailrocket_predicate_validation/checkpoints"
            / f"k1_seed{seed}/retailrocket_LiFTER_seed{seed}.pt"
        )
    return (
        ROOT
        / "experiments/jodie_predicate_validation/checkpoints"
        / f"{dataset}_k1_seed{seed}/{dataset}_LiFTER_seed{seed}.pt"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("wikipedia", "reddit", "mooc", "lastfm", "retailrocket"),
    )
    parser.add_argument("--seed", required=True, type=int, choices=(7, 17, 27))
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    saved_checkpoint = checkpoint(args.dataset, args.seed)
    if not saved_checkpoint.exists():
        raise FileNotFoundError(saved_checkpoint)
    saved = torch.load(saved_checkpoint, map_location="cpu", weights_only=False)
    model_config = dict(saved["model_config"])

    config = yaml.safe_load(
        (ROOT / "experiments/main_benchmark/config_lifter.yaml").read_text()
    )
    config.update(dataset=args.dataset, model="LiFTER", seed=args.seed)
    config["model_config"] = model_config
    config["training"].update(
        epochs=0,
        progress=args.progress,
        historical_negative_strategy="uniform",
        max_examples=262144 if args.dataset == "retailrocket" else 32768,
    )
    if args.dataset == "retailrocket":
        config["training"]["query_filter"] = {
            "column": "action_type",
            "value": "transaction",
        }
    config["paths"].pop("checkpoints", None)
    config["paths"]["load_checkpoint"] = str(saved_checkpoint)
    config["lifter_diagnostics"] = {"enabled": True}
    config["predicate_diagnostics"] = {"enabled": False}
    config["query_diagnostics"] = {"enabled": False}
    config["mechanism_decomposition"] = {
        "enabled": True,
        "fact_intervention": {
            "enabled": True,
            "component": DOMINANT_COMPONENT[args.dataset],
            "max_queries": 512,
        },
    }
    config["explanation_evaluation"]["max_queries"] = 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    dataset_path = (
        ROOT / f"data/processed/ctdg/{args.dataset}/events.csv"
    )
    spec = RunSpec(
        "ctdg",
        args.dataset,
        dataset_path,
        "LiFTER",
        "base",
        args.seed,
        "reference",
        model_config,
        False,
    )
    metrics = evaluate_run(
        spec,
        config,
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "seed": args.seed,
        "checkpoint": str(saved_checkpoint),
        "queries": metrics.get("test_examples"),
        "historical_auc": metrics["historical_negative_auc"],
        "historical_ap": metrics["historical_negative_ap"],
        "mechanisms": metrics["mechanism_decomposition"],
        "family_margins": metrics["historical_failure_diagnosis"],
        "query_trace": metrics["mechanism_query_trace"],
        "grounded_fact_intervention": metrics["grounded_fact_intervention"],
    }
    output = HERE / "outputs" / f"{args.dataset}_seed{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
