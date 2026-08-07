from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln
from scipy.stats import hypergeom


@dataclass
class FullCatalogAccumulator:
    reciprocal_ranks: list[float]
    hits_at_1: list[float]
    hits_at_10: list[float]
    ranks: list[float]
    catalog_sizes: list[int]
    tie_counts: list[int]

    @classmethod
    def create(cls) -> "FullCatalogAccumulator":
        return cls([], [], [], [], [], [])

    def add(self, scores: np.ndarray, positive_index: int) -> float:
        scores = np.asarray(scores, dtype=np.float64)
        positive = float(scores[positive_index])
        competitors = np.delete(scores, positive_index)
        greater = int((competitors > positive).sum())
        ties = int((competitors == positive).sum())
        rank = 1.0 + float(greater) + 0.5 * float(ties)
        self.reciprocal_ranks.append(1.0 / rank)
        self.hits_at_1.append(float(rank <= 1.0))
        self.hits_at_10.append(float(rank <= 10.0))
        self.ranks.append(rank)
        self.catalog_sizes.append(len(scores))
        self.tie_counts.append(ties)
        return rank

    def summarize(self) -> dict[str, float | int]:
        if not self.reciprocal_ranks:
            return {
                "events": 0,
                "mrr": float("nan"),
                "hits_at_1": float("nan"),
                "hits_at_10": float("nan"),
            }
        return {
            "events": len(self.reciprocal_ranks),
            "mrr": float(np.mean(self.reciprocal_ranks)),
            "hits_at_1": float(np.mean(self.hits_at_1)),
            "hits_at_10": float(np.mean(self.hits_at_10)),
            "mean_catalog_size": float(np.mean(self.catalog_sizes)),
            "events_with_ties": int(sum(count > 0 for count in self.tie_counts)),
        }


def sampled_rank(positive: float, negatives: np.ndarray) -> tuple[float, float, float]:
    negatives = np.asarray(negatives, dtype=np.float64)
    rank = 1.0 + float((negatives > positive).sum())
    rank += 0.5 * float((negatives == positive).sum())
    return 1.0 / rank, float(rank <= 1.0), float(rank <= 10.0)


def expected_uniform_sampled_rr(
    rank: float,
    catalog_size: int,
    negative_count: int,
    tied_count: int = 0,
) -> float:
    """Exact expected RR with average-rank tie handling."""

    if catalog_size < 1:
        raise ValueError("catalog_size must be positive")
    if not 1 <= rank <= catalog_size:
        raise ValueError("rank must lie in [1, catalog_size]")
    tied_count = int(tied_count)
    higher_count = int(round(float(rank) - 1.0 - 0.5 * tied_count))
    lower_count = catalog_size - 1 - higher_count - tied_count
    draws = min(max(int(negative_count), 0), catalog_size - 1)
    if draws == catalog_size - 1:
        return 1.0 / rank
    if tied_count:
        def log_choose(n: int, k: int) -> float:
            if k < 0 or k > n:
                return float("-inf")
            return float(
                gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
            )

        denominator = log_choose(catalog_size - 1, draws)
        expectation = 0.0
        for higher in range(
            max(0, draws - tied_count - lower_count),
            min(draws, higher_count) + 1,
        ):
            for tied in range(
                max(0, draws - higher - lower_count),
                min(draws - higher, tied_count) + 1,
            ):
                lower = draws - higher - tied
                log_probability = (
                    log_choose(higher_count, higher)
                    + log_choose(tied_count, tied)
                    + log_choose(lower_count, lower)
                    - denominator
                )
                expectation += np.exp(log_probability) / (
                    1.0 + higher + 0.5 * tied
                )
        return float(expectation)
    support = np.arange(
        max(0, draws - lower_count),
        min(draws, higher_count) + 1,
        dtype=np.int64,
    )
    probabilities = hypergeom.pmf(
        support,
        catalog_size - 1,
        higher_count,
        draws,
    )
    return float(np.sum(probabilities / (1.0 + support)))
