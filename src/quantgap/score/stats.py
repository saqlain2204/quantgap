"""Percentiles and a bootstrap confidence interval that resamples whole sequences."""

from __future__ import annotations

import numpy as np

N_BOOT = 1000
SEED = 0


def percentile(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if len(x) else float("nan")


def bootstrap_mean_ci(
    seq_sums: np.ndarray, seq_counts: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED
) -> tuple[float, float]:
    """95% interval for the mean per-position value, resampling *sequences*.

    Positions inside one sequence are not independent, so resampling positions would give an
    interval that is too narrow. `seq_sums[i]` is the sum of the value over sequence i and
    `seq_counts[i]` its number of positions.
    """
    seq_sums = np.asarray(seq_sums, dtype=np.float64)
    seq_counts = np.asarray(seq_counts, dtype=np.float64)
    n = len(seq_sums)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = seq_sums[idx].sum(axis=1) / seq_counts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))
