#!/opt/conda/bin/python
"""Run the final uniform-historical CTDG benchmark, resuming completed cells."""
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
PYTHON = "/opt/conda/bin/python"
DATASETS = ("wikipedia", "reddit", "mooc", "lastfm")
SEEDS = (7, 17, 27)
BASELINES = ("EdgeBank", "TGN", "TGAT", "DyGFormer", "GraphMixer", "CRAFT", "CRAFT-R")
MODELS = (*BASELINES, "LiFTER")
EXPECTED_EVENTS = 32768
EXPECTED_EPOCHS = 10
EXPECTED_LIFTER_BATCH_SIZE = 512
EXPECTED_LIFTER_LEARNING_RATE = 0.004
EXPECTED_LIFTER_GROUNDING_FACTS = 10
EXPECTED_LIFTER_PREDICATE_COUNT = 1


def output_path(dataset: str, model: str, seed: int) -> Path:
    return HERE / "outputs" / f"{dataset}_{model}_base_seed{seed}.json"


def complete(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        result = json.loads(path.read_text())
        training = result["config"]["training"]
        metrics = result["metrics"]
        model_name = result.get("model", path.name.split("_")[1])
        lifter_current = (
            model_name != "LiFTER"
            or (
                result["config"]["model_config"].get("latent_transition_rule") is True
                and result["config"]["model_config"].get(
                    "latent_second_order_transition_rule"
                ) is True
                and result["config"]["model_config"].get(
                    "recurrence_guarded_transitions"
                ) is True
                and result["config"]["model_config"].get(
                    "predicate_conditioned_transitions"
                ) is True
                and result["config"]["model_config"].get(
                    "positioned_recurrence_rules"
                ) is True
                and result["config"]["model_config"].get(
                    "sparse_ternary_execution"
                ) is True
                and int(result["config"]["model_config"]["max_rule_length"]) == 2
                and int(result["config"]["model_config"]["max_grounding_facts"])
                == EXPECTED_LIFTER_GROUNDING_FACTS
                and int(result["config"]["model_config"]["predicate_count"])
                == EXPECTED_LIFTER_PREDICATE_COUNT
                and int(training["batch_size"]) == EXPECTED_LIFTER_BATCH_SIZE
                and float(training["learning_rate"])
                == EXPECTED_LIFTER_LEARNING_RATE
            )
        )
        return (
            int(training["max_examples"]) == EXPECTED_EVENTS
            and (model_name == "EdgeBank" or int(training["epochs"]) == EXPECTED_EPOCHS)
            and metrics.get("historical_negative_strategy") == "uniform"
            and "historical_negative_auc" in metrics
            and "historical_negative_ap" in metrics
            and lifter_current
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def run(config: str, dataset: str, model: str, seed: int, *, force: bool, progress: bool) -> None:
    path = output_path(dataset, model, seed)
    label = f"{dataset}/{model}/seed{seed}"
    if not force and complete(path):
        print(f"SKIP {label}", flush=True)
        return
    print(f"START {label}", flush=True)
    if model == "EdgeBank":
        subprocess.run(
            [PYTHON, str(HERE / "run_memory_baselines.py"), "--datasets", dataset,
             "--models", model, "--seeds", str(seed)],
            check=True,
        )
        if not complete(path):
            raise RuntimeError(f"run did not produce a valid result: {path}")
        print(f"DONE {label}", flush=True)
        return
    command = [
        PYTHON,
        str(HERE / "run.py"),
        "--config",
        str(HERE / config),
        "--dataset",
        dataset,
        "--model",
        model,
        "--seed",
        str(seed),
    ]
    if progress:
        command.append("--progress")
    subprocess.run(
        command,
        check=True,
        stdout=None if progress else subprocess.DEVNULL,
    )
    if not complete(path):
        raise RuntimeError(f"run did not produce a valid uniform result: {path}")
    print(f"DONE {label}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--force", action="store_true", help="Rerun completed cells.")
    parser.add_argument("--progress", action="store_true", help="Show tqdm progress bars.")
    parser.add_argument("--workers", type=int, default=1, help="Independent cells to run concurrently.")
    args = parser.parse_args()
    invalid_seeds = sorted(set(args.seeds) - set(SEEDS))
    if invalid_seeds:
        parser.error(f"unsupported seeds: {invalid_seeds}; choose from {SEEDS}")
    cells = []
    for dataset in args.datasets:
        for model in args.models:
            config = (
                "config_lifter.yaml"
                if model == "LiFTER"
                else "config_comparators.yaml"
            )
            for seed in args.seeds:
                cells.append((config, dataset, model, seed))
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(
                run, config, dataset, model, seed,
                force=args.force, progress=args.progress,
            )
            for config, dataset, model, seed in cells
        ]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
