#!/opt/conda/bin/python
"""Compute exhaustive mechanism interactions and query-regime effects."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = ("wikipedia", "reddit", "mooc", "lastfm")
SEEDS = (7, 17, 27)


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return float(
        (ranks[labels == 1].sum() - len(positive) * (len(positive) + 1) / 2)
        / (len(positive) * len(negative))
    )


def ap(labels: np.ndarray, scores: np.ndarray) -> float:
    pairs = sorted(
        zip(scores.astype(float).tolist(), labels.astype(int).tolist()),
        key=lambda pair: pair[0], reverse=True,
    )
    positives = sum(label for _, label in pairs)
    if positives == 0:
        return float("nan")
    true_positives = false_positives = 0
    previous_recall = value = 0.0
    index = 0
    while index < len(pairs):
        score = pairs[index][0]
        group_positives = group_size = 0
        while index < len(pairs) and pairs[index][0] == score:
            group_positives += pairs[index][1]
            group_size += 1
            index += 1
        true_positives += group_positives
        false_positives += group_size - group_positives
        recall = true_positives / positives
        precision = true_positives / (true_positives + false_positives)
        value += (recall - previous_recall) * precision
        previous_recall = recall
    return float(value)


def metric(trace: dict, mask: int, query_mask: np.ndarray | None = None) -> tuple[float, float]:
    names = trace["component_names"]
    positive = np.asarray(trace["positive_prior"], dtype=np.float64)
    negative = np.asarray(trace["historical_prior"], dtype=np.float64)
    for index, name in enumerate(names):
        if mask & (1 << index):
            positive += np.asarray(trace["positive_components"][name])
            negative += np.asarray(trace["historical_components"][name])
    if query_mask is not None:
        positive = positive[query_mask]
        negative = negative[query_mask]
    labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative))))
    scores = np.concatenate((positive, negative))
    return auc(labels, scores), ap(labels, scores)


def interaction_for_trace(trace: dict) -> dict:
    names = trace["component_names"]
    count = len(names)
    values = {mask: metric(trace, mask) for mask in range(1 << count)}
    factorial = math.factorial
    denominator = factorial(count)
    shapley = {name: [0.0, 0.0] for name in names}
    for index, name in enumerate(names):
        bit = 1 << index
        for mask in range(1 << count):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = factorial(size) * factorial(count - size - 1) / denominator
            for metric_index in (0, 1):
                shapley[name][metric_index] += weight * (
                    values[mask | bit][metric_index] - values[mask][metric_index]
                )
    pairwise = []
    empty = values[0]
    full_mask = (1 << count) - 1
    for left, right in itertools.combinations(range(count), 2):
        lv, rv, both = values[1 << left], values[1 << right], values[(1 << left) | (1 << right)]
        without_left = values[full_mask ^ (1 << left)]
        without_right = values[full_mask ^ (1 << right)]
        without_both = values[full_mask ^ (1 << left) ^ (1 << right)]
        full = values[full_mask]
        pairwise.append({
            "left": names[left], "right": names[right],
            "auc_synergy": both[0] - lv[0] - rv[0] + empty[0],
            "ap_synergy": both[1] - lv[1] - rv[1] + empty[1],
            "full_context_auc_synergy": full[0] - without_left[0] - without_right[0] + without_both[0],
            "full_context_ap_synergy": full[1] - without_left[1] - without_right[1] + without_both[1],
        })
    full = values[full_mask]
    return {
        "prior_auc": empty[0], "prior_ap": empty[1],
        "full_auc": full[0], "full_ap": full[1],
        "shapley": {name: {"auc": value[0], "ap": value[1]} for name, value in shapley.items()},
        "pairwise_synergy": pairwise,
        "shapley_efficiency_error": {
            "auc": sum(value[0] for value in shapley.values()) - (full[0] - empty[0]),
            "ap": sum(value[1] for value in shapley.values()) - (full[1] - empty[1]),
        },
        "all_subsets": {
            str(mask): {"auc": value[0], "ap": value[1]} for mask, value in values.items()
        },
    }


def regimes_for_trace(trace: dict) -> dict:
    names = trace["component_names"]
    full_mask = (1 << len(names)) - 1
    recurrence = np.asarray(trace["direct_recurrence"], dtype=bool)
    activity = np.asarray(trace["source_activity"], dtype=np.float64)
    popularity = np.asarray(trace["destination_popularity"], dtype=np.float64)
    position = np.asarray(trace["test_position"], dtype=np.float64)
    activity_threshold = float(np.median(activity))
    popularity_threshold = float(np.median(popularity))
    groups = {
        "direct_recurrence_present": recurrence,
        "direct_recurrence_absent": ~recurrence,
        "low_source_activity": activity <= activity_threshold,
        "high_source_activity": activity > activity_threshold,
        "low_destination_popularity": popularity <= popularity_threshold,
        "high_destination_popularity": popularity > popularity_threshold,
        "early_test_interval": position < 0.5,
        "late_test_interval": position >= 0.5,
    }
    result = {
        "source_activity_median": activity_threshold,
        "destination_popularity_median": popularity_threshold,
        "groups": {},
    }
    for group, query_mask in groups.items():
        full_auc, full_ap = metric(trace, full_mask, query_mask)
        deletion = {}
        for index, name in enumerate(names):
            removed_auc, removed_ap = metric(trace, full_mask ^ (1 << index), query_mask)
            deletion[name] = {
                "auc_drop": full_auc - removed_auc,
                "ap_drop": full_ap - removed_ap,
            }
        result["groups"][group] = {
            "queries": int(query_mask.sum()),
            "full_auc": full_auc,
            "full_ap": full_ap,
            "deletion": deletion,
        }
    return result


def aggregate(records: list[dict], section: str) -> dict:
    if section == "interaction":
        names = records[0]["shapley"].keys()
        return {
            "full_auc": float(np.mean([row["full_auc"] for row in records])),
            "full_ap": float(np.mean([row["full_ap"] for row in records])),
            "shapley": {
                name: {
                    key: {
                        "mean": float(np.mean([row["shapley"][name][key] for row in records])),
                        "std": float(np.std([row["shapley"][name][key] for row in records], ddof=1)),
                    }
                    for key in ("auc", "ap")
                }
                for name in names
            },
            "pairwise_ap_synergy": sorted(
                [
                    {
                        "left": pair["left"], "right": pair["right"],
                        "mean": float(np.mean([
                            next(item["full_context_ap_synergy"] for item in row["pairwise_synergy"]
                                 if item["left"] == pair["left"] and item["right"] == pair["right"])
                            for row in records
                        ])),
                    }
                    for pair in records[0]["pairwise_synergy"]
                ], key=lambda item: item["mean"], reverse=True
            ),
        }
    groups = records[0]["groups"].keys()
    names = next(iter(records[0]["groups"].values()))["deletion"].keys()
    return {
        group: {
            "queries": int(round(np.mean([row["groups"][group]["queries"] for row in records]))),
            "full_auc": float(np.mean([row["groups"][group]["full_auc"] for row in records])),
            "full_ap": float(np.mean([row["groups"][group]["full_ap"] for row in records])),
            "deletion": {
                name: {
                    "auc_drop": float(np.mean([row["groups"][group]["deletion"][name]["auc_drop"] for row in records])),
                    "ap_drop": float(np.mean([row["groups"][group]["deletion"][name]["ap_drop"] for row in records])),
                } for name in names
            },
        } for group in groups
    }


def main() -> None:
    per_seed = {}
    summary = {}
    for dataset in DATASETS:
        interactions, regimes, interventions = [], [], []
        for seed in SEEDS:
            source = HERE / "outputs" / f"{dataset}_seed{seed}.json"
            payload = json.loads(source.read_text())
            trace = payload["query_trace"]
            interaction = interaction_for_trace(trace)
            regime = regimes_for_trace(trace)
            intervention = payload["grounded_fact_intervention"]
            per_seed[f"{dataset}_seed{seed}"] = {
                "interaction": interaction,
                "regime": regime,
                "grounded_fact_intervention": intervention,
            }
            interactions.append(interaction)
            regimes.append(regime)
            interventions.append(intervention)
        intervention_keys = (
            "top_auc_drop", "top_ap_drop", "matched_random_auc_drop",
            "matched_random_ap_drop", "top_mean_abs_logit_delta",
            "matched_random_mean_abs_logit_delta", "delta_ratio",
        )
        summary[dataset] = {
            "interaction": aggregate(interactions, "interaction"),
            "regime": aggregate(regimes, "regime"),
            "grounded_fact_intervention": {
                "component": interventions[0]["component"],
                "queries": int(round(np.mean([row["queries"] for row in interventions]))),
                **{
                    key: {
                        "mean": float(np.mean([row[key] for row in interventions])),
                        "std": float(np.std([row[key] for row in interventions], ddof=1)),
                    }
                    for key in intervention_keys
                },
                "signed_margin_intervention": {
                    key: {
                        "mean": float(np.mean([row["signed_margin_intervention"][key] for row in interventions])),
                        "std": float(np.std([row["signed_margin_intervention"][key] for row in interventions], ddof=1)),
                    }
                    for key in (
                        "supporting_queries", "supporting_top_mean_margin_drop",
                        "supporting_random_mean_margin_drop", "supporting_direction_consistency",
                        "opposing_queries", "opposing_top_mean_margin_gain",
                        "opposing_random_mean_margin_gain", "opposing_direction_consistency",
                    )
                },
            },
        }
    results = HERE / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "deep_per_seed.json").write_text(json.dumps(per_seed, indent=2) + "\n")
    (results / "deep_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = [
        "# Deep mechanism decomposition", "",
        "## Shapley allocation", "",
        "Values are Historical AP percentage-point contributions across all 128 subsets.", "",
        "| Dataset | Largest positive component | Shapley AP (%p) |",
        "|---|---|---:|",
    ]
    for dataset, result in summary.items():
        name, value = max(
            result["interaction"]["shapley"].items(),
            key=lambda item: item[1]["ap"]["mean"],
        )
        lines.append(f"| {dataset.title()} | {name} | {100 * value['ap']['mean']:.2f} |")
    lines += [
        "", "## Full-context pair interactions", "",
        "Positive values indicate complementarity and negative values indicate overlap or interference after conditioning on the other five components.", "",
        "| Dataset | Strongest complement | AP interaction (%p) | Strongest interference | AP interaction (%p) |",
        "|---|---|---:|---|---:|",
    ]
    for dataset, result in summary.items():
        pairs = result["interaction"]["pairwise_ap_synergy"]
        positive, negative = pairs[0], pairs[-1]
        lines.append(
            f"| {dataset.title()} | {positive['left']} + {positive['right']} | "
            f"{100 * positive['mean']:+.2f} | {negative['left']} + {negative['right']} | "
            f"{100 * negative['mean']:+.2f} |"
        )
    lines += [
        "", "## Signed decision-margin intervention", "",
        "Supporting and opposing facts are selected by their signed contribution to the positive-versus-historical-negative margin. The fact database is edited and the complete program is then re-grounded.", "",
        "| Dataset | Supporting: top / random margin drop | Direction consistency | Opposing: top / random margin gain | Direction consistency |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset, result in summary.items():
        signed = result["grounded_fact_intervention"]["signed_margin_intervention"]
        lines.append(
            f"| {dataset.title()} | {signed['supporting_top_mean_margin_drop']['mean']:.3f} / "
            f"{signed['supporting_random_mean_margin_drop']['mean']:.3f} | "
            f"{100*signed['supporting_direction_consistency']['mean']:.1f}% | "
            f"{signed['opposing_top_mean_margin_gain']['mean']:.3f} / "
            f"{signed['opposing_random_mean_margin_gain']['mean']:.3f} | "
            f"{100*signed['opposing_direction_consistency']['mean']:.1f}% |"
        )
    lines += [
        "", "## Grounded fact intervention", "",
        "The top grounded fact and a random fact from the same history bank and recency quartile are deleted before the full program is re-grounded.", "",
        "| Dataset | Component | Queries | Top / matched-random logit change | Top / matched-random AP drop (%p) |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset, result in summary.items():
        item = result["grounded_fact_intervention"]
        lines.append(
            f"| {dataset.title()} | {item['component']} | {item['queries']} | "
            f"{item['top_mean_abs_logit_delta']['mean']:.3f} / "
            f"{item['matched_random_mean_abs_logit_delta']['mean']:.3f} | "
            f"{100 * item['top_ap_drop']['mean']:.2f} / "
            f"{100 * item['matched_random_ap_drop']['mean']:.2f} |"
        )
    lines += ["", "## Query regimes", ""]
    for dataset, result in summary.items():
        lines += [f"### {dataset.title()}", "", "| Regime | Queries | Full AP (%) | Largest deletion effect | AP drop (%p) |", "|---|---:|---:|---|---:|"]
        for group, values in result["regime"].items():
            if values["queries"] == 0:
                continue
            name, effect = max(values["deletion"].items(), key=lambda item: item[1]["ap_drop"])
            lines.append(f"| {group} | {values['queries']} | {100 * values['full_ap']:.2f} | {name} | {100 * effect['ap_drop']:.2f} |")
        lines.append("")
    (results / "deep_report.md").write_text("\n".join(lines) + "\n")
    print(f"saved {results / 'deep_summary.json'}")


if __name__ == "__main__":
    main()
