"""First-order inclusion probabilities for fixed-size annotation designs.

Critical contract (roadmap v1.1):
selection_score is a generator score. It is NOT the propensity used by IPW.
True propensity is P(M_{n,c}=1), which for fixed-size sampling without
replacement is generally not equal to a normalized sigmoid score.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def uniform_inclusion_prob(num_classes: int, labels_per_volume: int) -> np.ndarray:
    """Analytic inclusion prob for uniform fixed-size sampling: k / C."""
    if labels_per_volume < 0 or labels_per_volume > num_classes:
        raise ValueError("labels_per_volume must be in [0, num_classes]")
    return np.full(num_classes, labels_per_volume / float(num_classes), dtype=np.float64)


def monte_carlo_inclusion_prob(
    sample_fn: Callable[[np.random.Generator], np.ndarray],
    *,
    num_volumes: int,
    num_classes: int,
    n_draws: int = 2000,
    seed: int = 0,
    batch_size: int = 256,
) -> np.ndarray:
    """Estimate P(M[n,c]=1) by Monte-Carlo resampling of a mask generator.

    sample_fn(rng) -> binary mask [N, C] for the same volume set.
    """
    if n_draws <= 0:
        raise ValueError("n_draws must be positive")
    rng = np.random.default_rng(seed)
    counts = np.zeros((num_volumes, num_classes), dtype=np.float64)
    done = 0
    while done < n_draws:
        b = min(batch_size, n_draws - done)
        for _ in range(b):
            m = np.asarray(sample_fn(rng), dtype=np.float64)
            if m.shape != (num_volumes, num_classes):
                raise ValueError(
                    f"sample_fn returned shape {m.shape}, expected {(num_volumes, num_classes)}"
                )
            if not np.isin(m, (0, 1)).all():
                raise ValueError("sample_fn must return binary masks")
            counts += m
        done += b
    return counts / float(n_draws)


def enforce_positivity(
    inclusion_prob: np.ndarray,
    *,
    floor: float,
) -> np.ndarray:
    """Clip inclusion probabilities from below for reporting/diagnostics only.

    Warning: do NOT feed the clipped array to Oracle-IPW. Oracle must use the
    unclipped design inclusion probability. Prefer enforcing positivity in the
    sampling score construction instead.
    """
    if floor <= 0 or floor >= 1:
        raise ValueError("floor must be in (0, 1)")
    pi = np.asarray(inclusion_prob, dtype=np.float64).copy()
    if np.any(pi < 0) or np.any(pi > 1):
        raise ValueError("inclusion_prob must lie in [0, 1]")
    np.clip(pi, floor, 1.0, out=pi)
    return pi


def summarize_weights(
    inclusion_prob: np.ndarray,
    *,
    eps: float = 1e-6,
) -> dict[str, float]:
    """IPW diagnostics: weight max/P95 and effective sample size proxy."""
    pi = np.asarray(inclusion_prob, dtype=np.float64)
    w = 1.0 / np.maximum(pi, eps)
    # ESS for self-normalized IPW over all entries.
    ess = float((w.sum() ** 2) / max(float((w ** 2).sum()), eps))
    return {
        "pi_min": float(pi.min()) if pi.size else 0.0,
        "pi_max": float(pi.max()) if pi.size else 0.0,
        "weight_max": float(w.max()) if w.size else 0.0,
        "weight_p95": float(np.percentile(w, 95)) if w.size else 0.0,
        "ess": ess,
        "n": float(pi.size),
    }
