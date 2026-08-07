#!/opt/conda/bin/python
"""Evaluate one LiFTER grounding budget on a pre-test temporal validation split."""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import RunSpec, evaluate_run, read_csv_tail


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("wikipedia", "reddit", "mooc", "lastfm"))
    parser.add_argument("--horizon", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be positive")

    config_path = REPO_ROOT / "experiments" / "main_benchmark" / "config_lifter.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = REPO_ROOT / "data" / "processed" / "ctdg" / args.dataset / "events.csv"
    frame = read_csv_tail(source_path, 32768).sort_values("timestamp", kind="stable").reset_index(drop=True)

    # The final 15% used by the main benchmark is never exposed to model or
    # horizon selection. Validation is the final 15% of the preceding prefix.
    test_start = min(max(1, int(len(frame) * 0.85)), len(frame) - 1)
    development = frame.iloc[:test_start].reset_index(drop=True)
    config["dataset"] = args.dataset
    config["model"] = "LiFTER"
    config["seed"] = args.seed
    config["training"]["epochs"] = args.epochs
    config["training"]["max_examples"] = len(development)
    config["training"]["eval_ratio"] = 0.15
    config["training"]["progress"] = args.progress
    config["model_config"]["max_grounding_facts"] = args.horizon
    config["explanation_evaluation"]["max_queries"] = 0

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with tempfile.TemporaryDirectory(prefix=f"lifter-{args.dataset}-validation-") as directory:
        validation_path = Path(directory) / "events.csv"
        development.to_csv(validation_path, index=False)
        spec = RunSpec(
            domain="ctdg",
            dataset=args.dataset,
            dataset_path=validation_path,
            model="LiFTER",
            variant="base",
            seed=args.seed,
            implementation="reference",
            model_config=dict(config["model_config"]),
            use_attention=False,
        )
        metrics = evaluate_run(spec, config, device)

    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_split": {
            "source_events": len(frame),
            "development_events": len(development),
            "held_out_test_events": len(frame) - len(development),
            "validation_ratio_within_development": 0.15,
            "held_out_test_used": False,
        },
        "dataset": args.dataset,
        "horizon": args.horizon,
        "predicate_count": int(config["model_config"]["predicate_count"]),
        "seed": args.seed,
        "epochs": args.epochs,
        "metrics": metrics,
    }
    output = HERE / "outputs" / f"{args.dataset}_H{args.horizon}_seed{args.seed}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
