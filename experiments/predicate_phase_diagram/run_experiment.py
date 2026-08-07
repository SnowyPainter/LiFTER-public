#!/opt/conda/bin/python
"""Two-axis test of when a latent predicate vocabulary becomes learnable and useful."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, average_precision_score, normalized_mutual_info_score, roc_auc_score
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from experiments.latent_mechanism_recovery.run_experiment import Split, TypedRuleProgram, batches

CENTRES = np.asarray([[1.0, -0.4, 0.1], [-0.4, 1.0, -0.3], [-0.2, -0.4, 1.0]], dtype=np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def semantic_target(types: np.ndarray) -> bool:
    return tuple(map(int, types)) in {(0, 1, 2), (1, 2, 0), (2, 0, 1)}


def make_split(n: int, alpha: float, beta: float, seed: int, entity_start: int) -> Split:
    rng = np.random.default_rng(seed)
    dim, length = 10, 3
    features = np.zeros((n, length, dim), np.float32)
    valid = np.ones((n, length), bool)
    hidden = np.empty((n, length), dtype=np.int64)
    labels = np.empty(n, np.int64)
    # The template is held fixed so type-dependent outcome is the only reason
    # to split the observed Link relation into multiple predicates.
    template = np.full(n, 2, np.int64)
    semantic_classes = np.arange(n, dtype=np.int64) % 2
    rng.shuffle(semantic_classes)
    for row, semantic_class in enumerate(semantic_classes):
        while True:
            types = rng.integers(0, 3, size=length, dtype=np.int64)
            if semantic_target(types) == bool(semantic_class):
                break
        hidden[row] = types
        # alpha=0 makes the future label independent of semantics; alpha=1
        # makes it exactly equal. Class balance is preserved in expectation.
        labels[row] = int(semantic_class if rng.random() < 0.5 + 0.5 * alpha else 1 - semantic_class)
        signal = np.zeros((length, dim), np.float32)
        signal[:, :3] = 3.0 * beta * CENTRES[types]
        signal[:, :3] += rng.normal(0, 1.0, size=(length, 3)).astype(np.float32)
        signal[:, 3:] = rng.normal(0, 1.0, size=(length, dim - 3)).astype(np.float32)
        features[row] = signal
    entities = rng.integers(entity_start, entity_start + 1_000_000, size=(n, 4), dtype=np.int64)
    return Split(*(torch.from_numpy(x) for x in (features, valid, template, labels, hidden, entities)))


def fit(split: Split, predicates: int, seed: int, device: torch.device, epochs: int, mode: str = "learned") -> TypedRuleProgram:
    model = TypedRuleProgram(split.features.shape[-1], predicates, mode, seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    x, templates, labels = split.features.to(device), split.template.to(device), split.labels.float().to(device)
    ids = torch.arange(len(x), device=device)[:, None] * 3 + torch.arange(3, device=device)[None]
    for _ in range(epochs):
        for ix in batches(len(x), 256, generator):
            if mode == "oracle":
                assignments = F.one_hot(split.hidden_types[ix].to(device), predicates).float()
                logits = torch.einsum("bi,bj,bk,ijk->b", assignments[:, 0], assignments[:, 1], assignments[:, 2], model.rule3)
            else:
                logits = model(x[ix], templates[ix], ids[ix])
            loss = F.binary_cross_entropy_with_logits(logits, labels[ix])
            if predicates > 1 and mode == "learned":
                probabilities = model.assignments(x[ix])
                entropy = -(probabilities * probabilities.clamp_min(1e-9).log()).sum(-1).mean()
                usage = probabilities.mean((0, 1))
                balance = (usage * (usage.clamp_min(1e-9).log() + math.log(predicates))).sum()
                loss = loss + 0.002 * entropy + 0.01 * balance
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return model


@torch.no_grad()
def evaluate(model: TypedRuleProgram, split: Split, device: torch.device) -> dict:
    x = split.features.to(device)
    ids = torch.arange(len(x), device=device)[:, None] * 3 + torch.arange(3, device=device)[None] + 100_000
    if model.mode == "oracle":
        probability = F.one_hot(split.hidden_types.to(device), model.predicates).float()
        logits = torch.einsum("bi,bj,bk,ijk->b", probability[:, 0], probability[:, 1], probability[:, 2], model.rule3)
    else:
        probability = model.assignments(x, ids)
        logits = model(x, split.template.to(device), ids)
    shuffled = probability.flatten(0, 1)[torch.randperm(probability.shape[0] * 3, device=device)].reshape_as(probability)
    shuffle_logits = torch.einsum("bi,bj,bk,ijk->b", shuffled[:, 0], shuffled[:, 1], shuffled[:, 2], model.rule3)
    labels = split.labels.numpy(); predictions = probability.argmax(-1).cpu().numpy(); truth = split.hidden_types.numpy()
    confusion = np.zeros((3, model.predicates), np.int64)
    for actual, predicted in zip(truth.ravel(), predictions.ravel()): confusion[actual, predicted] += 1
    rows, columns = linear_sum_assignment(-confusion)
    recovery = confusion[rows, columns].sum() / confusion.sum()
    usage = np.bincount(predictions.ravel(), minlength=model.predicates) / predictions.size
    active_rules = int((model.rule3.detach().abs() > 0.1).sum().cpu())
    return {
        "auc": float(roc_auc_score(labels, logits.cpu())), "ap": float(average_precision_score(labels, logits.cpu())),
        "shuffle_auc": float(roc_auc_score(labels, shuffle_logits.cpu())), "shuffle_ap": float(average_precision_score(labels, shuffle_logits.cpu())),
        "recovery_accuracy": float(recovery), "nmi": float(normalized_mutual_info_score(truth.ravel(), predictions.ravel())),
        "ari": float(adjusted_rand_score(truth.ravel(), predictions.ravel())), "vocabulary_usage": usage.tolist(),
        "active_typed_rules": active_rules,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--train-size", type=int, default=6000); parser.add_argument("--test-size", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--augment-controls",
        action="store_true",
        help="Reuse the saved K=1/learned grid and add oracle/random controls.",
    )
    args = parser.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    destination = HERE / "results" / "summary.json"
    if args.augment_controls:
        saved = json.loads(destination.read_text())
        records = [
            {**row, "variant": row.get("variant", "k1" if row["predicates"] == 1 else "learned")}
            for row in saved["records"]
            if row.get("variant", "k1" if row["predicates"] == 1 else "learned")
            in {"k1", "learned"}
        ]
        for alpha in args.levels:
            for seed in args.seeds:
                set_seed(seed)
                # Oracle and random assignments do not use context, hence their
                # distribution is invariant to beta and needs one fit per alpha.
                train = make_split(args.train_size, alpha, 0.0, seed, 0)
                test = make_split(args.test_size, alpha, 0.0, seed + 10_000, 2_000_000)
                for variant in ("oracle", "random"):
                    model = fit(train, 3, seed, device, args.epochs, variant)
                    metrics = evaluate(model, test, device)
                    for beta in args.levels:
                        records.append({"alpha": alpha, "beta": beta, "seed": seed, "variant": variant, "predicates": 3, **metrics})
            print(f"completed controls alpha={alpha:.2f}", flush=True)
    else:
        for alpha in args.levels:
            for beta in args.levels:
                for seed in args.seeds:
                    set_seed(seed)
                    train = make_split(args.train_size, alpha, beta, seed, 0)
                    test = make_split(args.test_size, alpha, beta, seed + 10_000, 2_000_000)
                    for variant, predicates, mode in (
                        ("k1", 1, "learned"),
                        ("learned", 3, "learned"),
                        ("oracle", 3, "oracle"),
                        ("random", 3, "random"),
                    ):
                        model = fit(train, predicates, seed, device, args.epochs, mode)
                        metrics = evaluate(model, test, device)
                        records.append({"alpha": alpha, "beta": beta, "seed": seed, "variant": variant, "predicates": predicates, **metrics})
                print(f"completed alpha={alpha:.2f} beta={beta:.2f}", flush=True)
    summary = []
    for alpha in args.levels:
        for beta in args.levels:
            cells = {}
            for variant in ("k1", "learned", "oracle", "random"):
                rows = [r for r in records if r["alpha"] == alpha and r["beta"] == beta and r["variant"] == variant]
                cells[variant] = {key: {"mean": float(np.mean([r[key] for r in rows])), "std": float(np.std([r[key] for r in rows], ddof=1))} for key in ("auc", "ap", "shuffle_auc", "shuffle_ap", "recovery_accuracy", "nmi", "ari", "active_typed_rules")}
            learned, k1 = cells["learned"], cells["k1"]
            summary.append({
                "alpha": alpha, "beta": beta, "k1": k1, "k3": learned,
                "oracle": cells["oracle"], "random": cells["random"],
                "auc_gain": learned["auc"]["mean"] - k1["auc"]["mean"],
                "ap_gain": learned["ap"]["mean"] - k1["ap"]["mean"],
                "shuffle_auc_drop": learned["auc"]["mean"] - learned["shuffle_auc"]["mean"],
                "oracle_auc_gain": cells["oracle"]["auc"]["mean"] - k1["auc"]["mean"],
                "random_auc_gain": cells["random"]["auc"]["mean"] - k1["auc"]["mean"],
            })
    output = {"design": {
        "alpha": "P(label follows hidden semantic class)=0.5+0.5*alpha",
        "beta": "context class-centre amplitude=3*beta under unit Gaussian noise",
        "hidden_types": 3,
        "sequence_length": 3,
        "positive_semantic_sequences": [[0, 1, 2], [1, 2, 0], [2, 0, 1]],
        "observed_relations": 1,
        "class_balance": "exactly 50/50 hidden semantic classes; label balance in expectation under symmetric flips",
        "train_test_entities": "disjoint; entity IDs are not model inputs",
        "train_size": args.train_size, "test_size": args.test_size,
        "epochs": args.epochs, "seeds": args.seeds,
        "recovery": "Hungarian permutation-matched fact-level accuracy; NMI and ARI also reported",
        "oracle_predicate": "ground-truth hidden type supplied only to the oracle control",
        "random_predicate": "fixed uniform hash assignment independent of context and label",
    }, "records": records, "summary": summary}
    destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(output, indent=2) + "\n")
    print(f"saved {destination}")


if __name__ == "__main__": main()
