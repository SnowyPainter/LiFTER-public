#!/opt/conda/bin/python
"""Summarize JODIE predicate-count validation results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1))}


def main() -> None:
    records = [json.loads(path.read_text()) for path in sorted((HERE / "outputs").glob("*.json"))]
    summary = {}
    for dataset in ("wikipedia", "reddit", "mooc", "lastfm"):
        summary[dataset] = {}
        for k in (1, 2, 4, 8):
            rows = [r for r in records if r["dataset"] == dataset and r["predicate_count"] == k]
            if not rows:
                continue
            metrics = {}
            for key in ("historical_negative_auc", "historical_negative_ap"):
                metrics[key] = mean_std([r["normal"][key] for r in rows])
                metrics[f"shuffle_{key}"] = mean_std([r["predicate_shuffle"][key] for r in rows])
                metrics[f"shuffle_delta_{key}"] = mean_std([
                    r["normal"][key] - r["predicate_shuffle"][key] for r in rows
                ])
                if all("global_predicate_shuffle" in r for r in rows):
                    metrics[f"global_shuffle_{key}"] = mean_std([
                        r["global_predicate_shuffle"][key] for r in rows
                    ])
                    metrics[f"global_shuffle_delta_{key}"] = mean_std([
                        r["normal"][key] - r["global_predicate_shuffle"][key]
                        for r in rows
                    ])
            diagnostics = [r["normal"]["predicate_diagnostics"] for r in rows]
            for key in ("effective_predicates", "predicate_role_separation"):
                if all(key in d for d in diagnostics):
                    metrics[key] = mean_std([d[key] for d in diagnostics])
            summary[dataset][str(k)] = metrics
    destination = HERE / "results" / "summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"records": len(records), "summary": summary}, indent=2) + "\n")
    print(f"saved {destination}")


if __name__ == "__main__":
    main()
