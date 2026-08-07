#!/opt/conda/bin/python
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.evaluate import RunSpec, evaluate_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a reproducible CTDG link-prediction baseline.")
    parser.add_argument("--config", type=Path, default=EXPERIMENT_DIR / "config.yaml")
    parser.add_argument("--dataset")
    parser.add_argument(
        "--model",
        choices=["TGN", "TGAT", "DyGFormer", "GraphMixer", "CRAFT", "CRAFT-R", "TGIB", "SIG", "LiFTER"],
    )
    parser.add_argument("--variant", choices=["base", "attention"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--max-examples", type=int, help="Use only the most recent N events.")
    parser.add_argument("--history-len", type=int, help="Number of recent interactions per node.")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-rule-length", type=int, choices=(1, 2, 3))
    parser.add_argument("--max-grounding-facts", type=int)
    parser.add_argument(
        "--recurrence-guarded-transitions",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--predicate-conditioned-transitions",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--positioned-recurrence-rules",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output", type=Path, help="Override the result directory.")
    parser.add_argument(
        "--sparse-symbolic-execution",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show tqdm training and evaluation progress bars.",
    )
    parser.add_argument(
        "--aggregation-operator",
        choices=["softmax", "multiscale_kernel", "multiscale_pool"],
    )
    parser.add_argument("--neural-history-len", type=int)
    parser.add_argument("--recurrence-scale-count", type=int)
    parser.add_argument("--recurrence-scale-init", choices=["short", "full", "long"])
    parser.add_argument(
        "--recurrence-evidence-mode",
        choices=["mass", "mean", "softmax"],
    )
    parser.add_argument(
        "--freeze-recurrence-scales",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--target-agnostic-include-activity",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    for key in ("dataset", "model", "variant", "seed", "device"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.max_examples is not None:
        if args.max_examples < 2:
            raise ValueError("--max-examples must be at least 2")
        config["training"]["max_examples"] = args.max_examples
    if args.history_len is not None:
        if args.history_len < 1:
            raise ValueError("--history-len must be positive")
        config["training"]["history_len"] = args.history_len
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise ValueError("--learning-rate must be positive")
        config["training"]["learning_rate"] = args.learning_rate
    if args.max_rule_length is not None:
        config["model_config"]["max_rule_length"] = args.max_rule_length
    if args.max_grounding_facts is not None:
        if args.max_grounding_facts < 1:
            raise ValueError("--max-grounding-facts must be positive")
        config["model_config"]["max_grounding_facts"] = args.max_grounding_facts
    if args.recurrence_guarded_transitions is not None:
        config["model_config"]["recurrence_guarded_transitions"] = (
            args.recurrence_guarded_transitions
        )
    if args.predicate_conditioned_transitions is not None:
        config["model_config"]["predicate_conditioned_transitions"] = (
            args.predicate_conditioned_transitions
        )
    if args.positioned_recurrence_rules is not None:
        config["model_config"]["positioned_recurrence_rules"] = (
            args.positioned_recurrence_rules
        )
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        config["training"]["epochs"] = args.epochs
    if args.progress is not None:
        config["training"]["progress"] = args.progress
    if args.output is not None:
        config["paths"]["output"] = str(args.output)
    if args.sparse_symbolic_execution is not None:
        config["model_config"]["sparse_ternary_execution"] = args.sparse_symbolic_execution
        config["model_config"]["sparse_renewal_execution"] = args.sparse_symbolic_execution
    if args.aggregation_operator is not None:
        config["model_config"]["aggregation_operator"] = args.aggregation_operator
    if args.neural_history_len is not None:
        if args.neural_history_len < 1:
            raise ValueError("--neural-history-len must be positive")
        config["model_config"]["neural_history_len"] = args.neural_history_len
    if args.recurrence_scale_count is not None:
        if args.recurrence_scale_count < 1:
            raise ValueError("--recurrence-scale-count must be positive")
        config["model_config"]["recurrence_scale_count"] = args.recurrence_scale_count
    if args.recurrence_scale_init is not None:
        config["model_config"]["recurrence_scale_init"] = args.recurrence_scale_init
    if args.recurrence_evidence_mode is not None:
        config["model_config"]["recurrence_evidence_mode"] = args.recurrence_evidence_mode
    if args.freeze_recurrence_scales is not None:
        config["model_config"]["learnable_recurrence_scales"] = not args.freeze_recurrence_scales
    if args.target_agnostic_include_activity is not None:
        config["model_config"]["target_agnostic_include_activity"] = (
            args.target_agnostic_include_activity
        )


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    apply_overrides(config, args)
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    processed_root = resolve_path(config["paths"]["processed_root"])
    dataset_path = processed_root / "ctdg" / str(config["dataset"]) / "events.csv"
    variant = str(config["variant"])
    spec = RunSpec(
        domain="ctdg",
        dataset=str(config["dataset"]),
        dataset_path=dataset_path,
        model=str(config["model"]),
        variant=variant,
        seed=seed,
        implementation=str(config.get("implementation", "reference")),
        model_config=dict(config["model_config"]),
        use_attention=variant == "attention",
    )
    device = choose_device(str(config.get("device", "auto")))
    metrics = evaluate_run(spec, config, device)
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "config": config,
        "metrics": metrics,
    }
    output_dir = resolve_path(config["paths"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{spec.dataset}_{spec.model}_{variant}_seed{seed}.json"
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
