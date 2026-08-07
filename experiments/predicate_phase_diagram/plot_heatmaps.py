#!/opt/conda/bin/python
"""Render the complete predicate phase diagram from saved experiment results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


HERE = Path(__file__).resolve().parent


def main() -> None:
    payload = json.loads((HERE / "results" / "summary.json").read_text())
    rows = payload["summary"]
    levels = sorted({float(row["alpha"]) for row in rows})

    def matrix(accessor):
        return np.asarray([
            [accessor(next(row for row in rows if row["alpha"] == alpha and row["beta"] == beta)) for beta in levels]
            for alpha in levels
        ])

    panels = (
        ("Learned K=3 AUC gain (pp)", matrix(lambda r: 100 * r["auc_gain"]), "RdBu_r", 0.0),
        ("Predicate-shuffle AUC drop (pp)", matrix(lambda r: 100 * r["shuffle_auc_drop"]), "mako", None),
        ("Type recovery accuracy (%)", matrix(lambda r: 100 * r["k3"]["recovery_accuracy"]["mean"]), "viridis", None),
        ("Type recovery NMI", matrix(lambda r: r["k3"]["nmi"]["mean"]), "viridis", None),
        ("Oracle K=3 AUC gain (pp)", matrix(lambda r: 100 * r["oracle_auc_gain"]), "mako", None),
        ("Random K=3 AUC gain (pp)", matrix(lambda r: 100 * r["random_auc_gain"]), "RdBu_r", 0.0),
    )
    sns.set_theme(style="white", context="paper", font_scale=0.92)
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.7), constrained_layout=True)
    labels = [f"{value:g}" for value in levels]
    for axis, (title, values, cmap, centre) in zip(axes.flat, panels):
        sns.heatmap(
            values,
            ax=axis,
            cmap=cmap,
            center=centre,
            annot=True,
            fmt=".1f" if "NMI" not in title else ".2f",
            linewidths=0.5,
            linecolor="white",
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"shrink": 0.75},
        )
        axis.set_title(title, weight="bold", pad=8)
        axis.set_xlabel(r"Identifiability $\beta$")
        axis.set_ylabel(r"Predictive effect $\alpha$")
        axis.tick_params(axis="y", rotation=0)
    output = HERE / "results" / "predicate_phase_diagram"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
