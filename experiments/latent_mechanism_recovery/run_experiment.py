#!/opt/conda/bin/python
"""Controlled recovery of hidden fact roles from one observed Link relation."""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    normalized_mutual_info_score,
    roc_auc_score,
)
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
MECHANISMS = ("recurrence", "transition", "exploration")


@dataclass
class Split:
    features: torch.Tensor       # [N, 3, D], every row is an untyped Link fact
    valid: torch.Tensor          # [N, 3]
    template: torch.Tensor       # structural rule template, not a predicate label
    labels: torch.Tensor         # future Link label
    hidden_types: torch.Tensor   # evaluation only; never passed to the model
    entity_ids: torch.Tensor     # audit only; never passed to the model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _valid_label(template: int, types: np.ndarray) -> bool:
    if template == 0:  # a previous Link has the recurrence role
        return int(types[0]) == 0
    if template == 1:  # two typed facts form a directed transition
        return (int(types[0]), int(types[1])) in {(0, 1), (1, 2), (2, 0)}
    # Three facts jointly express an exploratory bridge.
    return tuple(map(int, types[:3])) in {
        (0, 1, 2), (1, 2, 0), (2, 0, 1),
    }


def make_split(n: int, seed: int, entity_start: int) -> Split:
    rng = np.random.default_rng(seed)
    dim = 10
    # The role signal is distributed across continuous context dimensions. Large
    # nuisance directions make geometry-only clustering an imperfect control.
    centres = np.zeros((3, dim), dtype=np.float32)
    centres[:, :3] = np.asarray(
        [[0.90, -0.25, 0.10], [-0.35, 0.85, -0.15], [-0.30, -0.30, 0.90]],
        dtype=np.float32,
    )
    features = np.zeros((n, 3, dim), dtype=np.float32)
    valid = np.zeros((n, 3), dtype=bool)
    hidden = np.full((n, 3), -1, dtype=np.int64)
    templates = np.arange(n, dtype=np.int64) % 3
    labels = np.arange(n, dtype=np.int64) % 2
    rng.shuffle(templates)
    rng.shuffle(labels)
    for i, (template, target) in enumerate(zip(templates, labels)):
        length = int(template) + 1
        while True:
            types = rng.integers(0, 3, size=length)
            if _valid_label(int(template), types) == bool(target):
                break
        hidden[i, :length] = types
        valid[i, :length] = True
        signal = centres[types] + rng.normal(0, 0.34, size=(length, dim)).astype(np.float32)
        # Context contains recency/activity measurements but no entity id or type label.
        signal[:, 3:] += rng.normal(0, 1.4, size=(length, dim - 3)).astype(np.float32)
        features[i, :length] = signal
    # Completely disjoint identities across splits; IDs are retained only to audit this.
    entity_ids = rng.integers(entity_start, entity_start + 100_000, size=(n, 4))
    return Split(
        *(torch.from_numpy(x) for x in (features, valid, templates, labels, hidden, entity_ids))
    )


class TypedRuleProgram(nn.Module):
    """Neural fact typing followed only by an explicit finite rule program."""

    def __init__(self, dim: int, predicates: int, mode: str, seed: int) -> None:
        super().__init__()
        self.predicates = predicates
        self.mode = mode
        self.encoder = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, predicates))
        self.rule1 = nn.Parameter(torch.zeros(predicates))
        self.rule2 = nn.Parameter(torch.zeros(predicates, predicates))
        self.rule3 = nn.Parameter(torch.zeros(predicates, predicates, predicates))
        if mode != "learned":
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("random_table", torch.randint(0, predicates, (200_000,), generator=generator))
        self.register_buffer("centroids", torch.empty(0))

    def set_centroids(self, value: torch.Tensor) -> None:
        self.centroids = value

    def assignments(self, x: torch.Tensor, fact_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self.predicates == 1:
            return torch.ones(*x.shape[:-1], 1, device=x.device)
        if self.mode == "random":
            if fact_ids is None:
                raise ValueError("random typing requires fact ids")
            ids = self.random_table[fact_ids % len(self.random_table)]
            return F.one_hot(ids, self.predicates).float()
        if self.mode == "kmeans":
            distance = (x[..., None, :] - self.centroids).square().sum(-1)
            return F.one_hot(distance.argmin(-1), self.predicates).float()
        return torch.softmax(self.encoder(x), dim=-1)

    def forward(self, x: torch.Tensor, template: torch.Tensor, fact_ids: torch.Tensor) -> torch.Tensor:
        p = self.assignments(x, fact_ids)
        score1 = torch.einsum("bi,i->b", p[:, 0], self.rule1)
        score2 = torch.einsum("bi,bj,ij->b", p[:, 0], p[:, 1], self.rule2)
        score3 = torch.einsum("bi,bj,bk,ijk->b", p[:, 0], p[:, 1], p[:, 2], self.rule3)
        return torch.where(template == 0, score1, torch.where(template == 1, score2, score3))


def batches(n: int, size: int, generator: torch.Generator):
    order = torch.randperm(n, generator=generator)
    for start in range(0, n, size):
        yield order[start:start + size]


def fit_variant(train: Split, mode: str, seed: int, device: torch.device) -> TypedRuleProgram:
    k = 1 if mode == "single" else 3
    model = TypedRuleProgram(train.features.shape[-1], k, mode, seed).to(device)
    if mode == "kmeans":
        raw = train.features[train.valid].numpy()
        km = KMeans(n_clusters=3, n_init=20, random_state=seed).fit(raw)
        model.set_centroids(torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=3e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    x, template, labels = train.features.to(device), train.template.to(device), train.labels.float().to(device)
    ids = torch.arange(len(x), device=device)[:, None].expand(-1, 3)
    ids = ids * 3 + torch.arange(3, device=device)[None]
    model.train()
    for _ in range(80):
        for ix in batches(len(x), 256, generator):
            logits = model(x[ix], template[ix], ids[ix])
            loss = F.binary_cross_entropy_with_logits(logits, labels[ix])
            if mode == "learned":
                probs = model.assignments(x[ix])
                entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
                usage = probs[train.valid[ix].to(device)].mean(0)
                balance = (usage * (usage.clamp_min(1e-9).log() + math.log(k))).sum()
                loss = loss + 0.002 * entropy + 0.01 * balance
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def evaluate(model: TypedRuleProgram, split: Split, device: torch.device) -> dict:
    model.eval()
    x = split.features.to(device)
    ids = torch.arange(len(x), device=device)[:, None].expand(-1, 3) * 3
    ids = ids + torch.arange(3, device=device)[None] + 100_000
    logits = model(x, split.template.to(device), ids).cpu().numpy()
    labels = split.labels.numpy()
    pred = model.assignments(x, ids).argmax(-1).cpu()
    mask = split.valid
    true_types, predicted = split.hidden_types[mask].numpy(), pred[mask].numpy()
    confusion = np.zeros((3, model.predicates), dtype=np.int64)
    for truth, guess in zip(true_types, predicted):
        confusion[truth, guess] += 1
    row, col = linear_sum_assignment(-confusion)
    accuracy = confusion[row, col].sum() / confusion.sum()
    by_mechanism = {}
    for template, name in enumerate(MECHANISMS):
        selected = split.template.numpy() == template
        by_mechanism[name] = {
            "auc": float(roc_auc_score(labels[selected], logits[selected])),
            "ap": float(average_precision_score(labels[selected], logits[selected])),
            "queries": int(selected.sum()),
        }
    return {
        "auc": float(roc_auc_score(labels, logits)),
        "ap": float(average_precision_score(labels, logits)),
        "type_accuracy_permutation_matched": float(accuracy),
        "type_nmi": float(normalized_mutual_info_score(true_types, predicted)),
        "type_ari": float(adjusted_rand_score(true_types, predicted)),
        "confusion_hidden_by_learned": confusion.tolist(),
        "forecasting_by_mechanism": by_mechanism,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27, 37, 47])
    parser.add_argument("--train-size", type=int, default=12000)
    parser.add_argument("--test-size", type=int, default=6000)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = ("single", "random", "kmeans", "learned")
    records = []
    for seed in args.seeds:
        set_seed(seed)
        train = make_split(args.train_size, seed, 0)
        test = make_split(args.test_size, seed + 10_000, 1_000_000)
        assert set(train.entity_ids.flatten().tolist()).isdisjoint(test.entity_ids.flatten().tolist())
        for variant in variants:
            model = fit_variant(train, variant, seed, device)
            metrics = evaluate(model, test, device)
            records.append({"seed": seed, "variant": variant, **metrics})
            print(seed, variant, f"AUC={metrics['auc']:.4f}", f"type_acc={metrics['type_accuracy_permutation_matched']:.4f}", flush=True)
    summary = {}
    for variant in variants:
        rows = [r for r in records if r["variant"] == variant]
        summary[variant] = {
            key: {"mean": float(np.mean([r[key] for r in rows])), "std": float(np.std([r[key] for r in rows], ddof=1))}
            for key in ("auc", "ap", "type_accuracy_permutation_matched", "type_nmi", "type_ari")
        }
    output = {"design": {
        "observed_relation_count": 1,
        "observed_relation": "Link(source,destination,time)",
        "hidden_mechanisms": list(MECHANISMS),
        "typing_supervision": "future-link labels only",
        "entity_split": "disjoint train/test; entity IDs excluded from model",
        "matched_controls": "fact count, rule template, and class balance",
    }, "records": records, "summary": summary}
    out = HERE / "results" / "summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
