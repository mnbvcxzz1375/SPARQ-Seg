"""AOVA (v,c)-pair acquisition strategies (Phase A, single-shot).

Pure numpy; deterministic given an rng seed. Inputs:
- mask: uint8 [N, C] binary, 1 = organ c annotated on volume v (given, frozen).
- budget: number of NEW (v, c) pairs to acquire.
- scores: float [N, C], higher = more valuable; only un-annotated cells count.
Selections never touch already-annotated cells.
"""

from __future__ import annotations

import numpy as np


def coverage(mask: np.ndarray) -> np.ndarray:
    """Per-class annotated fraction over volumes, shape [C]."""
    return np.asarray(mask).mean(axis=0)


def candidate_cells(mask: np.ndarray) -> np.ndarray:
    """[M, 2] int array of un-annotated (v, c) pairs, deterministic order."""
    return np.argwhere(np.asarray(mask) == 0)


def _as_mask(pairs: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if len(pairs):
        out[pairs[:, 0], pairs[:, 1]] = True
    return out


def select_random(mask: np.ndarray, budget: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random over un-annotated pairs."""
    mask = np.asarray(mask)
    cands = candidate_cells(mask)
    if budget <= 0 or len(cands) == 0:
        return np.zeros(mask.shape, dtype=bool)
    idx = rng.choice(len(cands), size=min(budget, len(cands)), replace=False)
    return _as_mask(cands[idx], mask.shape)


def select_class_balanced(
    mask: np.ndarray, budget: int, rng: np.random.Generator, eps: float = 1.0
) -> np.ndarray:
    """Random with per-class weight 1/(annotated_count + eps): prefer
    under-annotated organs first; order within class random."""
    mask = np.asarray(mask)
    cands = candidate_cells(mask)
    if budget <= 0 or len(cands) == 0:
        return np.zeros(mask.shape, dtype=bool)
    cov = mask.sum(axis=0)
    w_cell = 1.0 / (cov[cands[:, 1]] + eps)
    p = w_cell / w_cell.sum()
    k = min(budget, len(cands))
    idx = rng.choice(len(cands), size=k, replace=False, p=p)
    return _as_mask(cands[idx], mask.shape)


def select_top_scores(mask: np.ndarray, scores: np.ndarray, budget: int) -> np.ndarray:
    """Deterministic top-K by score over un-annotated cells (ties: lower flat index)."""
    mask = np.asarray(mask)
    cands = candidate_cells(mask)
    if budget <= 0 or len(cands) == 0:
        return np.zeros(mask.shape, dtype=bool)
    s = np.asarray(scores, dtype=np.float64)[cands[:, 0], cands[:, 1]]
    order = np.lexsort((np.arange(len(s)), -s))
    k = min(budget, len(order))
    return _as_mask(cands[order[:k]], mask.shape)


def score_entropy_x_coverage(entropy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """priority[v,c] = entropy[v,c] * (1 - coverage[c])."""
    cov = coverage(mask)
    return np.asarray(entropy, dtype=np.float64) * (1.0 - cov)[None, :]
