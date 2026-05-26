"""Graded NDCG@K and paired-bootstrap CI for the evaluation report.

Both functions are pure (no I/O, no torch). Suitable to import from
the SM judge job (CPU) and from any future ECS migration.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


def _dcg(scores: list[float], k: int) -> float:
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores[:k]))


def ndcg_at_k_graded(
    ranking: list[str],
    scores: dict[str, float],
    k: int,
) -> float:
    if not ranking:
        return 0.0
    ranked = [scores.get(d, 0.0) for d in ranking[:k]]
    dcg = _dcg(ranked, k)
    ideal = sorted(scores.values(), reverse=True)
    idcg = _dcg(ideal, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


@dataclass(frozen=True)
class BootstrapResult:
    mean_diff: float
    ci_low: float
    ci_high: float


def paired_bootstrap_ci(
    base_per_query: list[float],
    ft_per_query: list[float],
    num_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> BootstrapResult:
    if len(base_per_query) != len(ft_per_query):
        raise ValueError("base and ft must be the same length")
    n = len(base_per_query)
    if n == 0:
        raise ValueError("inputs must not be empty")

    diffs = [ft - b for b, ft in zip(base_per_query, ft_per_query)]
    mean_diff = sum(diffs) / n

    rng = random.Random(seed)
    sample_means: list[float] = []
    for _ in range(num_resamples):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        sample_means.append(s / n)

    sample_means.sort()
    lo_idx = int((alpha / 2) * num_resamples)
    hi_idx = int((1 - alpha / 2) * num_resamples) - 1
    return BootstrapResult(
        mean_diff=mean_diff,
        ci_low=sample_means[max(0, lo_idx)],
        ci_high=sample_means[min(num_resamples - 1, hi_idx)],
    )
