#!/opt/conda/bin/python
"""Validate and summarize the final uniform-historical benchmark."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


HERE = Path(__file__).resolve().parent
DATASETS = ("wikipedia", "reddit", "mooc", "lastfm")
MODELS = ("LiFTER", "EdgeBank", "TGN", "TGAT", "DyGFormer", "GraphMixer", "CRAFT", "CRAFT-R")
SEEDS = (7, 17, 27)


def load(dataset: str, model: str, seed: int) -> dict:
    path = HERE / "outputs" / f"{dataset}_{model}_base_seed{seed}.json"
    result = json.loads(path.read_text())
    training = result["config"]["training"]
    metrics = result["metrics"]
    assert int(training["max_examples"]) == 32768, path
    assert model == "EdgeBank" or int(training["epochs"]) == 10, path
    assert metrics["historical_negative_strategy"] == "uniform", path
    if model == "LiFTER":
        model_config = result["config"]["model_config"]
        assert int(model_config["predicate_count"]) == 1, path
        assert int(model_config["max_grounding_facts"]) == 10, path
    return metrics


def main() -> None:
    summary: dict[str, dict[str, dict]] = {}
    for dataset in DATASETS:
        summary[dataset] = {}
        for model in MODELS:
            runs = [load(dataset, model, seed) for seed in SEEDS]
            auc = [float(run["historical_negative_auc"]) for run in runs]
            ap = [float(run["historical_negative_ap"]) for run in runs]
            random_auc = [float(run["auc"]) for run in runs]
            random_ap = [float(run["ap"]) for run in runs]
            summary[dataset][model] = {
                "historical_auc_mean": statistics.mean(auc),
                "historical_auc_sample_std": statistics.stdev(auc),
                "historical_ap_mean": statistics.mean(ap),
                "historical_ap_sample_std": statistics.stdev(ap),
                "random_auc_mean": statistics.mean(random_auc),
                "random_auc_sample_std": statistics.stdev(random_auc),
                "random_ap_mean": statistics.mean(random_ap),
                "random_ap_sample_std": statistics.stdev(random_ap),
                "seeds": list(SEEDS),
            }
        for metric in ("historical_auc", "historical_ap", "random_auc", "random_ap"):
            ranked = sorted(
                MODELS,
                key=lambda model: summary[dataset][model][f"{metric}_mean"],
                reverse=True,
            )
            for rank, model in enumerate(ranked, 1):
                summary[dataset][model][f"{metric}_rank"] = rank

    destination = HERE / "results" / "uniform_historical_summary.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n")

    # Keep the exact single-predicate LiFTER result visible independently of
    # the larger comparator table. The assertions in ``load`` prevent a stale
    # K>1 run from being copied into this artifact.
    lifter_summary = {
        "model": "LiFTER",
        "predicate_count": 1,
        "grounding_capacity": 10,
        "events_per_dataset": 32768,
        "epochs": 10,
        "seeds": list(SEEDS),
        "historical_negative_strategy": "uniform",
        "datasets": {dataset: summary[dataset]["LiFTER"] for dataset in DATASETS},
    }
    (HERE / "results" / "lifter_k1_summary.json").write_text(
        json.dumps(lifter_summary, indent=2) + "\n"
    )

    report = [
        "# LiFTER single-predicate benchmark",
        "",
        "This artifact contains the final LiFTER forecasting result with one observed predicate "
        "($K=1$), grounding capacity $H=10$, and uniform historical negatives.",
        "",
        "| Dataset | Historical AUC | Historical AP |",
        "|---|---:|---:|",
    ]
    for dataset in DATASETS:
        row = summary[dataset]["LiFTER"]
        report.append(
            f'| {dataset.title()} | {row["historical_auc_mean"]:.4f} ± '
            f'{row["historical_auc_sample_std"]:.4f} | {row["historical_ap_mean"]:.4f} ± '
            f'{row["historical_ap_sample_std"]:.4f} |'
        )
    (HERE / "results" / "lifter_k1_summary.md").write_text("\n".join(report) + "\n")

    print("| Dataset | Model | Historical AUC (rank) | Historical AP (rank) | Random AUC (rank) | Random AP (rank) |")
    print("|---|---|---:|---:|---:|---:|")
    for dataset in DATASETS:
        for model in MODELS:
            row = summary[dataset][model]
            auc = f'{row["historical_auc_mean"]:.4f} ± {row["historical_auc_sample_std"]:.4f} (#{row["historical_auc_rank"]})'
            ap = f'{row["historical_ap_mean"]:.4f} ± {row["historical_ap_sample_std"]:.4f} (#{row["historical_ap_rank"]})'
            random_auc = f'{row["random_auc_mean"]:.4f} ± {row["random_auc_sample_std"]:.4f} (#{row["random_auc_rank"]})'
            random_ap = f'{row["random_ap_mean"]:.4f} ± {row["random_ap_sample_std"]:.4f} (#{row["random_ap_rank"]})'
            print(f"| {dataset.title()} | {model} | {auc} | {ap} | {random_auc} | {random_ap} |")


if __name__ == "__main__":
    main()
