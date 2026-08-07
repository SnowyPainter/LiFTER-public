#!/opt/conda/bin/python
"""Aggregate exact mechanism-only and mechanism-deletion evaluations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATASETS = ("wikipedia", "reddit", "mooc", "lastfm")
MECHANISMS = (
    "direct_pair",
    "source_context",
    "candidate_context",
    "pair_renewal",
    "positioned_recurrence",
    "ordered_transition_1",
    "ordered_transition_2",
)


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
    }


def main() -> None:
    summary: dict[str, dict] = {}
    for dataset in DATASETS:
        rows = [
            json.loads(path.read_text())
            for path in sorted((HERE / "outputs").glob(f"{dataset}_seed*.json"))
        ]
        if len(rows) != 3:
            raise RuntimeError(f"expected three seeds for {dataset}, found {len(rows)}")
        full_auc = [row["mechanisms"]["full"]["historical_negative_auc"] for row in rows]
        full_ap = [row["mechanisms"]["full"]["historical_negative_ap"] for row in rows]
        mechanisms = {}
        for mechanism in MECHANISMS:
            only = f"only_{mechanism}"
            without = f"without_{mechanism}"
            mechanisms[mechanism] = {
                "only_auc": mean_std([
                    row["mechanisms"][only]["historical_negative_auc"] for row in rows
                ]),
                "only_ap": mean_std([
                    row["mechanisms"][only]["historical_negative_ap"] for row in rows
                ]),
                "auc_drop_when_removed": mean_std([
                    row["mechanisms"]["full"]["historical_negative_auc"]
                    - row["mechanisms"][without]["historical_negative_auc"]
                    for row in rows
                ]),
                "ap_drop_when_removed": mean_std([
                    row["mechanisms"]["full"]["historical_negative_ap"]
                    - row["mechanisms"][without]["historical_negative_ap"]
                    for row in rows
                ]),
                "mean_pairwise_logit_margin": mean_std([
                    row["mechanisms"][only]["mean_historical_logit_margin"]
                    for row in rows
                ]),
            }
        dominant = max(
            MECHANISMS,
            key=lambda name: mechanisms[name]["ap_drop_when_removed"]["mean"],
        )
        summary[dataset] = {
            "full_auc": mean_std(full_auc),
            "full_ap": mean_std(full_ap),
            "dominant_by_ap_deletion": dominant,
            "mechanisms": mechanisms,
        }
    result_dir = HERE / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    labels = {
        "direct_pair": "Direct pair",
        "source_context": "Source context",
        "candidate_context": "Candidate context",
        "pair_renewal": "Pair renewal",
        "positioned_recurrence": "Positioned recurrence",
        "ordered_transition_1": "One-event transition",
        "ordered_transition_2": "Two-event transition",
    }
    lines = [
        "# Mechanism decomposition",
        "",
        "All values use the frozen K=1 checkpoints and the same uniform-historical",
        "candidates as the forecasting benchmark. `Only AP` retains the prior and one",
        "mechanism. `Deletion AP drop` subtracts that mechanism's exact signed logit",
        "contribution from the full prediction; positive values indicate necessity.",
        "Metric drops need not add because AUC and AP are nonlinear ranking metrics.",
        "",
        "| Dataset | Full AUC/AP (%) | Largest AP deletion effect | AP drop (%p) |",
        "|---|---:|---|---:|",
    ]
    for dataset in DATASETS:
        item = summary[dataset]
        dominant = item["dominant_by_ap_deletion"]
        lines.append(
            f"| {dataset.title()} | {100*item['full_auc']['mean']:.2f} / "
            f"{100*item['full_ap']['mean']:.2f} | {labels[dominant]} | "
            f"{100*item['mechanisms'][dominant]['ap_drop_when_removed']['mean']:.2f} |"
        )
    lines.extend([
        "",
        "## Complete decomposition",
        "",
        "| Dataset | Mechanism | Only AUC/AP (%) | Deletion AUC/AP drop (%p) | Mean logit margin |",
        "|---|---|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        for mechanism in MECHANISMS:
            item = summary[dataset]["mechanisms"][mechanism]
            lines.append(
                f"| {dataset.title()} | {labels[mechanism]} | "
                f"{100*item['only_auc']['mean']:.2f} / {100*item['only_ap']['mean']:.2f} | "
                f"{100*item['auc_drop_when_removed']['mean']:+.2f} / "
                f"{100*item['ap_drop_when_removed']['mean']:+.2f} | "
                f"{item['mean_pairwise_logit_margin']['mean']:+.4f} |"
            )
    (result_dir / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:15]))


if __name__ == "__main__":
    main()
