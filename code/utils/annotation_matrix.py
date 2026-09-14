"""Annotation-mask statistics used as pattern metadata and severity calibration."""

from __future__ import annotations

from typing import Any

import numpy as np


def class_coverage(mask: np.ndarray) -> np.ndarray:
    """Return per-class coverage mean_n M[n, c] for mask [N, C]."""
    m = _as_binary_matrix(mask)
    return m.mean(axis=0)


def coverage_entropy(mask: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy (nats) of the class-coverage distribution."""
    p = class_coverage(mask)
    s = float(p.sum())
    if s <= 0:
        return 0.0
    p = p / s
    return float(-(p * np.log(p + eps)).sum())


def coverage_gini(mask: np.ndarray) -> float:
    """Gini coefficient of class-coverage vector in [0, 1]."""
    x = np.sort(class_coverage(mask).astype(np.float64))
    n = x.size
    if n == 0:
        return 0.0
    total = x.sum()
    if total <= 0:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (idx * x).sum()) / (n * total) - (n + 1.0) / n)


def co_annotation_matrix(mask: np.ndarray) -> np.ndarray:
    """Pairwise co-annotation rate P(M_c=1 and M_d=1), shape [C, C]."""
    m = _as_binary_matrix(mask).astype(np.float64)
    n = m.shape[0]
    if n == 0:
        return np.zeros((m.shape[1], m.shape[1]), dtype=np.float64)
    return (m.T @ m) / n


def pattern_meta(
    mask: np.ndarray,
    *,
    mode: str,
    severity: str,
    seed: int,
    budget: int,
    labels_per_volume: int,
    positivity_floor: float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the frozen pattern_meta payload for simulator_output."""
    m = _as_binary_matrix(mask)
    coverage = class_coverage(m)
    meta: dict[str, Any] = {
        "mode": mode,
        "severity": severity,
        "seed": int(seed),
        "budget": int(budget),
        "labels_per_volume": int(labels_per_volume),
        "num_volumes": int(m.shape[0]),
        "num_classes": int(m.shape[1]),
        "class_coverage": coverage.tolist(),
        "gini": coverage_gini(m),
        "entropy": coverage_entropy(m),
        "co_annotation": co_annotation_matrix(m).tolist(),
        "positivity_floor": None if positivity_floor is None else float(positivity_floor),
        "rows_with_budget": int((m.sum(axis=1) == labels_per_volume).sum()),
        "min_row_sum": int(m.sum(axis=1).min()) if m.size else 0,
        "max_row_sum": int(m.sum(axis=1).max()) if m.size else 0,
    }
    if extra:
        meta.update(extra)
    return meta


def _as_binary_matrix(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError(f"mask must be [N, C], got shape {m.shape}")
    if not np.isin(m, (0, 1)).all():
        raise ValueError("mask must be binary {0,1}")
    return m.astype(np.uint8, copy=False)
