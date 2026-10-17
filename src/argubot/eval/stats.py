"""Statistics (owner: Cassey): paired bootstrap confidence intervals and Cohen's kappa."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class BootstrapResult:
    mean_diff: float
    ci_low: float
    ci_high: float
    n: int


def paired_bootstrap_ci(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapResult:
    """95% CI (by default) for mean(a - b), resampling debates with replacement.

    Both lists must be per-debate scores in the same debate order, because every system
    faces the same debates and the comparison is paired.
    """
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("scores must be the same non-zero length, one per debate")
    diffs = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_boot, diffs.size))
    boot_means = diffs[idx].mean(axis=1)
    low, high = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return BootstrapResult(float(diffs.mean()), float(low), float(high), int(diffs.size))


def cohens_kappa(rater1: Sequence, rater2: Sequence) -> float:
    """Agreement between two raters, corrected for chance."""
    if len(rater1) != len(rater2) or len(rater1) == 0:
        raise ValueError("ratings must be the same non-zero length")
    n = len(rater1)
    labels = sorted(set(rater1) | set(rater2))
    observed = sum(x == y for x, y in zip(rater1, rater2)) / n
    expected = sum((list(rater1).count(l) / n) * (list(rater2).count(l) / n) for l in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)
