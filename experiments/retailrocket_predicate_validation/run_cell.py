#!/opt/conda/bin/python
"""Train one RetailRocket predicate-count cell and audit recovered action roles."""
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
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate import RunSpec, build_lifter_causal_fact_context, evaluate_run, read_csv_tail
from models.lifter import LiFTER


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def spec(seed: int, config: dict) -> RunSpec:
    return RunSpec(
        "ctdg", "retailrocket",
        ROOT / "data/processed/ctdg/retailrocket/events.csv",
        "LiFTER", "base", seed, "reference", dict(config), False,
    )


@torch.no_grad()
def action_recovery(checkpoint: Path, max_examples: int, device: torch.device) -> dict:
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = LiFTER(**saved["model_config"]).to(device)
    model.load_state_dict(saved["model"])
    model.eval(); model.set_symbolic_progress(1.0)
    frame = read_csv_tail(ROOT / "data/processed/ctdg/retailrocket/events.csv", max_examples)
    frame = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    context = build_lifter_causal_fact_context(frame, model.fact_context_len)
    features = np.concatenate(
        (np.ones((len(frame), 1), dtype=np.float32), context.reshape(len(frame), -1)), axis=1
    )
    start = int(len(frame) * 0.85)
    predictions = []
    for offset in range(start, len(frame), 2048):
        value = torch.tensor(features[offset:offset + 2048, None], device=device)
        valid = torch.ones(value.shape[:2], dtype=torch.bool, device=device)
        _, _, ids = model._type_facts(value, valid)
        predictions.extend(ids[:, 0].cpu().tolist())
    names = ("view", "addtocart", "transaction")
    truth = frame.iloc[start:]["action_type"].map({name: i for i, name in enumerate(names)}).to_numpy()
    predicted = np.asarray(predictions, dtype=np.int64)
    matrix = np.zeros((3, model.predicate_count), dtype=np.int64)
    for left, right in zip(truth, predicted): matrix[int(left), int(right)] += 1
    rows, columns = linear_sum_assignment(-matrix)
    mapping = {int(column): int(row) for row, column in zip(rows, columns)}
    mapped = np.asarray([mapping.get(int(value), -1) for value in predicted])
    recalls = [float(np.mean(mapped[truth == role] == role)) for role in range(3)]
    return {
        "queries": len(truth),
        "action_counts": {name: int(np.sum(truth == i)) for i, name in enumerate(names)},
        "nmi": float(normalized_mutual_info_score(truth, predicted)),
        "ari": float(adjusted_rand_score(truth, predicted)),
        "permutation_matched_accuracy": float(np.mean(mapped == truth)),
        "permutation_matched_macro_recall": float(np.mean(recalls)),
        "recall_by_action": dict(zip(names, recalls)),
        "confusion_action_by_predicate": matrix.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, required=True, choices=(1, 2, 3, 4, 8))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "experiments/main_benchmark/config_lifter.yaml").read_text())
    config.update(dataset="retailrocket", model="LiFTER", seed=args.seed)
    config["training"].update(
        progress=args.progress,
        historical_negative_strategy="uniform",
        max_examples=262144,
        query_filter={"column": "action_type", "value": "transaction"},
    )
    config["model_config"].update(
        predicate_count=args.k,
        predicate_assignment_mode="random_fixed" if args.k == 1 else "learned",
        predicate_execution_mode="hard",
    )
    config["predicate_diagnostics"] = {"enabled": True}
    config["query_diagnostics"] = {"enabled": False}
    config["explanation_evaluation"]["max_queries"] = 0
    checkpoint_dir = HERE / "checkpoints" / f"k{args.k}_seed{args.seed}"
    config["paths"]["checkpoints"] = str(checkpoint_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    normal = evaluate_run(spec(args.seed, config["model_config"]), config, device)
    checkpoint = Path(normal["checkpoint"])

    shuffled_config = copy.deepcopy(config)
    shuffled_config["training"].update(epochs=0, progress=False)
    shuffled_config["paths"].pop("checkpoints", None)
    shuffled_config["paths"]["load_checkpoint"] = str(checkpoint)
    shuffled_config["model_config"]["predicate_intervention"] = "shuffle_global"
    seed_all(args.seed)
    shuffled = evaluate_run(spec(args.seed, shuffled_config["model_config"]), shuffled_config, device)
    recovery = action_recovery(checkpoint, int(config["training"]["max_examples"]), device)
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(), "k": args.k, "seed": args.seed,
        "normal": normal, "global_predicate_shuffle": shuffled, "action_recovery": recovery,
    }
    destination = HERE / "outputs" / f"k{args.k}_seed{args.seed}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(f"saved {destination}", flush=True)


if __name__ == "__main__": main()
