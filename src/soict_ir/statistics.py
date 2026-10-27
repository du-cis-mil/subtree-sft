# Adapted for the standalone SoICT release (2026): local imports and CLI paths revised.
# MOOSE-Star-derived code; see LICENSE and NOTICE.
import math
import numpy as np

def bootstrap_diff(
    a_vals: list[bool],
    b_vals: list[bool],
    *,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    n = len(a_vals)
    diff_per_sample = np.asarray(a_vals, dtype=np.int8) - np.asarray(b_vals, dtype=np.int8)
    observed = float(diff_per_sample.mean())
    idx = rng.integers(0, n, size=(iterations, n), endpoint=False)
    diffs = diff_per_sample[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return observed, lo, hi

def log_comb(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

def binom_pmf(n: int, k: int, p: float = 0.5) -> float:
    return math.exp(log_comb(n, k) + k * math.log(p) + (n - k) * math.log1p(-p))

def exact_mcnemar_p(a_vals: list[bool], b_vals: list[bool]) -> tuple[int, int, float]:
    a_only = sum(1 for a, b in zip(a_vals, b_vals) if a and not b)
    b_only = sum(1 for a, b in zip(a_vals, b_vals) if b and not a)
    n = a_only + b_only
    if n == 0:
        return a_only, b_only, 1.0
    k = min(a_only, b_only)
    p = min(1.0, 2.0 * sum(binom_pmf(n, i) for i in range(k + 1)))
    return a_only, b_only, p
