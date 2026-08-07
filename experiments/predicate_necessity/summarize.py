#!/opt/conda/bin/python
"""Summarize forecasting, compression, and reuse of predicate variants."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
DATASETS = ("wikipedia", "reddit", "mooc", "lastfm")
VARIANTS = ("single_type", "random_types", "kmeans_types", "learned_soft", "learned_hard")
SEEDS = (7, 17, 27)


def load(dataset: str, variant: str, seed: int) -> dict:
    return json.loads((HERE / "outputs" / variant / f"{dataset}_seed{seed}.json").read_text())


def mean_std(values: list[float]) -> dict[str, float]:
    finite = [value for value in values if math.isfinite(value)]
    return {
        "mean": statistics.mean(finite) if finite else float("nan"),
        "sample_std": statistics.stdev(finite) if len(finite) > 1 else 0.0,
    }


def aligned_seed_stability(runs: list[dict]) -> float:
    profiles = [
        np.asarray(run["metrics"]["predicate_diagnostics"]["predicate_role_profiles"]["all"], dtype=float)
        for run in runs
    ]
    reference = profiles[0]
    similarities = []
    for profile in profiles[1:]:
        left = reference / np.maximum(np.linalg.norm(reference, axis=1, keepdims=True), 1e-12)
        right = profile / np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-12)
        matrix = left @ right.T
        rows, columns = linear_sum_assignment(-matrix)
        similarities.extend(matrix[rows, columns].tolist())
    return float(np.mean(similarities)) if similarities else 1.0


def predicate_role_separation(run: dict) -> float:
    profile = np.asarray(
        run["metrics"]["predicate_diagnostics"]["predicate_role_profiles"]["all"],
        dtype=float,
    )
    valid = np.linalg.norm(profile, axis=1) > 0
    profile = profile[valid]
    if len(profile) < 2:
        return float("nan")
    profile = profile / np.maximum(np.linalg.norm(profile, axis=1, keepdims=True), 1e-12)
    similarity = profile @ profile.T
    off_diagonal = (similarity.sum() - len(profile)) / (len(profile) * (len(profile) - 1))
    return float(1.0 - off_diagonal)


def main() -> None:
    summary: dict[str, dict[str, dict]] = {}
    for dataset in DATASETS:
        summary[dataset] = {}
        for variant in VARIANTS:
            runs = [load(dataset, variant, seed) for seed in SEEDS]
            diagnostics = [run["metrics"]["predicate_diagnostics"] for run in runs]
            entry = {
                "historical_auc": mean_std([run["metrics"]["historical_negative_auc"] for run in runs]),
                "historical_ap": mean_std([run["metrics"]["historical_negative_ap"] for run in runs]),
                "max_logit_reconstruction_error": max(
                    value["max_logit_reconstruction_error"] for value in diagnostics
                ),
                "effective_rule_count": mean_std([value["effective_rule_count"] for value in diagnostics]),
                "rules_for_90pct_absolute_contribution": mean_std([
                    value["rules_for_90pct_absolute_contribution"] for value in diagnostics
                ]),
                "mean_active_proofs_per_query": mean_std([
                    value["mean_active_proofs_per_query"] for value in diagnostics
                ]),
                "top_proof_absolute_logit_fraction": {
                    key: mean_std([value["top_proof_absolute_logit_fraction"][key] for value in diagnostics])
                    for key in ("1", "3", "5")
                },
                "role_reuse": {
                    key: mean_std([value["role_reuse"][key]["weighted_cosine"] for value in diagnostics])
                    for key in (
                        "seen_to_unseen_entity", "early_to_later_interval",
                        "source_activity_strata", "destination_popularity_strata",
                    )
                },
                "seed_aligned_predicate_role_stability": aligned_seed_stability(runs),
                "predicate_role_separation": mean_std([
                    predicate_role_separation(run) for run in runs
                ]),
                "elapsed_seconds": mean_std([run["elapsed_seconds"] for run in runs]),
            }
            summary[dataset][variant] = entry
    destination = HERE / "results" / "summary.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print("| Dataset | Typing | Historical AUC | Historical AP | Rules@90% | Top-3 fraction | Role separation | Seed stability |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for dataset in DATASETS:
        for variant in VARIANTS:
            row = summary[dataset][variant]
            print(
                f"| {dataset} | {variant} | {row['historical_auc']['mean']:.4f} ± {row['historical_auc']['sample_std']:.4f} "
                f"| {row['historical_ap']['mean']:.4f} ± {row['historical_ap']['sample_std']:.4f} "
                f"| {row['rules_for_90pct_absolute_contribution']['mean']:.1f} "
                f"| {row['top_proof_absolute_logit_fraction']['3']['mean']:.3f} "
                f"| {row['predicate_role_separation']['mean']:.3f} "
                f"| {row['seed_aligned_predicate_role_stability']:.3f} |"
            )


if __name__ == "__main__":
    main()
