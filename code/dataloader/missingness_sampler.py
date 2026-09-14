"""Structured missingness simulator for organ-level partial labels.

Roadmap v1.1 locks:
- Every main mode emits exactly `labels_per_volume` organs per volume.
- Returned `inclusion_prob` is the true first-order inclusion probability
  (analytic for uniform; Monte-Carlo for weighted fixed-size designs).
- `selection_score` is only the generator score and must not be used as π.
- Main modes enforce a positivity floor (default 0.05); true zeros only in
  `mnar_stress`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from code.utils.annotation_matrix import pattern_meta
from code.utils.inclusion_prob import (
    monte_carlo_inclusion_prob,
    uniform_inclusion_prob,
)

Mode = Literal["random", "longtail", "clustered", "conditional", "mnar_stress"]
Severity = Literal["moderate", "severe"]


@dataclass(frozen=True)
class SimulatorOutput:
    mask: np.ndarray
    selection_score: np.ndarray
    inclusion_prob: np.ndarray
    pattern_meta: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mask": self.mask,
            "selection_score": self.selection_score,
            "inclusion_prob": self.inclusion_prob,
            "pattern_meta": self.pattern_meta,
        }


class MissingnessSampler:
    def __init__(
        self,
        num_classes: int,
        labels_per_volume: int,
        mode: Mode = "random",
        severity: Severity = "moderate",
        seed: int = 42,
        *,
        positivity_floor: float = 0.05,
        n_mc_draws: int = 4000,
        longtail_lambda: float | None = None,
        n_sites: int = 3,
        conditional_strength: float = 1.8,
        uniform_blend: float | None = None,
    ) -> None:
        if num_classes <= 1:
            raise ValueError("num_classes must be > 1")
        if not (1 <= labels_per_volume <= num_classes):
            raise ValueError("labels_per_volume must be in [1, num_classes]")
        if severity not in ("moderate", "severe"):
            raise ValueError("severity must be moderate|severe")
        if mode not in ("random", "longtail", "clustered", "conditional", "mnar_stress"):
            raise ValueError(f"unknown mode {mode}")
        if not (0.0 < positivity_floor < 1.0):
            raise ValueError("positivity_floor must be in (0,1)")
        if n_mc_draws <= 0:
            raise ValueError("n_mc_draws must be positive")

        self.num_classes = int(num_classes)
        self.labels_per_volume = int(labels_per_volume)
        self.mode = mode
        self.severity = severity
        self.seed = int(seed)
        self.positivity_floor = float(positivity_floor)
        self.n_mc_draws = int(n_mc_draws)
        self.n_sites = int(n_sites)
        self.conditional_strength = float(conditional_strength)

        if longtail_lambda is None:
            self.longtail_lambda = 2.0 if severity == "moderate" else 3.0
        else:
            self.longtail_lambda = float(longtail_lambda)

        # Lower α keeps structured patterns distinguishable from MCAR while
        # still protecting fixed-k positivity (π_min ≥ floor).
        if uniform_blend is None:
            self.uniform_blend = 0.48 if severity == "moderate" else 0.32
        else:
            self.uniform_blend = float(uniform_blend)
        if not (0.0 <= self.uniform_blend < 1.0):
            raise ValueError("uniform_blend must be in [0, 1)")

    def generate(
        self,
        volume_ids: list[str] | np.ndarray,
        features: np.ndarray | None = None,
        site_ids: np.ndarray | None = None,
    ) -> SimulatorOutput:
        ids = list(volume_ids)
        n = len(ids)
        if n == 0:
            raise ValueError("volume_ids must be non-empty")
        c = self.num_classes
        k = self.labels_per_volume

        scores = self._selection_scores(n, features=features, site_ids=site_ids)

        if self.mode == "random":
            # Uniform fixed-size: analytic inclusion is exact; emit [N, C].
            inclusion = np.broadcast_to(uniform_inclusion_prob(c, k), (n, c)).copy()
            mask = self._sample_uniform_fixed_size(n, c, k)
        elif self.mode == "mnar_stress":
            inclusion = self._mc_inclusion_from_scores(n, scores)
            mask = self._sample_gumbel_topk(scores)
            # Stress mode may violate positivity; do not clip to a floor.
        else:
            inclusion = self._mc_inclusion_from_scores(n, scores)
            if self._min_class_mass(scores) <= 0:
                raise RuntimeError("main mode constructed non-positive class mass")
            mask = self._sample_gumbel_topk(scores)
            # Do NOT clip MC inclusion: Oracle-IPW needs the true design π.
            # Positivity is enforced earlier on selection scores. Fail closed
            # if the estimated first-order prob still falls below the floor.
            if float(np.min(inclusion)) < self.positivity_floor - 0.008:
                raise RuntimeError(
                    "main mode inclusion_prob below positivity_floor "
                    f"(min={float(np.min(inclusion)):.4f}, floor={self.positivity_floor}); "
                    "increase score floors or MC draws"
                )

        if not np.all(mask.sum(axis=1) == k):
            raise RuntimeError("sampler violated fixed-size budget")

        inclusion = np.asarray(inclusion, dtype=np.float64)
        if inclusion.ndim == 1:
            inclusion = np.broadcast_to(inclusion, (n, c)).copy()
        if inclusion.shape != (n, c):
            raise RuntimeError(f"inclusion_prob must be [N, C], got {inclusion.shape}")

        meta = pattern_meta(
            mask,
            mode=self.mode,
            severity=self.severity,
            seed=self.seed,
            budget=n * k,
            labels_per_volume=k,
            positivity_floor=None if self.mode == "mnar_stress" else self.positivity_floor,
            extra={
                "inclusion_prob_mean_class": inclusion.mean(axis=0).tolist(),
                "inclusion_min": float(inclusion.min()),
                "inclusion_max": float(inclusion.max()),
                "n_mc_draws": self.n_mc_draws if self.mode != "random" else 0,
            },
        )
        return SimulatorOutput(
            mask=mask.astype(np.uint8),
            selection_score=scores.astype(np.float64),
            inclusion_prob=inclusion.astype(np.float64),
            pattern_meta=meta,
        )

    # ------------------------------------------------------------------
    # score construction
    # ------------------------------------------------------------------
    def _selection_scores(
        self,
        n: int,
        *,
        features: np.ndarray | None,
        site_ids: np.ndarray | None,
    ) -> np.ndarray:
        c = self.num_classes
        if self.mode == "random":
            return np.ones((n, c), dtype=np.float64)

        if self.mode == "longtail":
            rank = np.arange(c, dtype=np.float64)  # 0 = most annotated
            lam = self.longtail_lambda
            raw = np.exp(-lam * rank / max(c - 1, 1))
            scores = np.broadcast_to(raw, (n, c)).copy()
            return self._floor_scores(scores)

        if self.mode == "clustered":
            if site_ids is None:
                site_ids = self._default_site_ids(n)
            site_ids = np.asarray(site_ids)
            if site_ids.shape[0] != n:
                raise ValueError("site_ids must have length N")
            if site_ids.max(initial=0) >= self.n_sites:
                raise ValueError("site_ids out of range")
            scores = np.full((n, c), 0.08, dtype=np.float64)
            # Each site prefers a contiguous organ block; little bleed.
            block = int(np.ceil(c / self.n_sites))
            for i in range(n):
                s = int(site_ids[i])
                lo = s * block
                hi = min(c, lo + block)
                scores[i, lo:hi] = 1.0
                if hi < c:
                    scores[i, hi] = max(scores[i, hi], 0.20)
                if lo > 0:
                    scores[i, lo - 1] = max(scores[i, lo - 1], 0.20)
            if self.severity == "severe":
                scores = np.where(scores < 1.0, scores * 0.15, scores)
            return self._floor_scores(scores)

        if self.mode == "conditional":
            if features is None:
                raise ValueError("conditional mode requires features [N, D]")
            feats = np.asarray(features, dtype=np.float64)
            if feats.ndim != 2 or feats.shape[0] != n:
                raise ValueError("features must be [N, D]")
            rng = np.random.default_rng(self.seed + 17)
            # Fixed linear map feature -> organ logits.
            d = feats.shape[1]
            w = rng.normal(0.0, self.conditional_strength / np.sqrt(d), size=(d, c))
            a = rng.normal(0.0, 0.2, size=(c,))
            logits = feats @ w + a
            scores = 1.0 / (1.0 + np.exp(-logits))
            return self._floor_scores(scores)

        if self.mode == "mnar_stress":
            # Stress: rare/small-organ classes get near-zero score (allowed).
            rank = np.arange(c, dtype=np.float64)
            lam = 3.5 if self.severity == "severe" else 2.5
            raw = np.exp(-lam * rank / max(c - 1, 1))
            scores = np.broadcast_to(raw, (n, c)).copy()
            # Tiny floor so Gumbel-topk can still sample if needed, but
            # inclusion can approach 0 for tail classes.
            scores = np.maximum(scores, 1e-4)
            return scores

        raise RuntimeError(f"unhandled mode {self.mode}")

    def _floor_scores(self, scores: np.ndarray) -> np.ndarray:
        """Make main-mode scores overlap-safe under fixed-size k sampling.

        Fixed-size Gumbel-topk couples classes: even mildly skewed scores can
        drive tail-class inclusion far below a naive score floor. We therefore
        (1) row-normalize, (2) blend toward the uniform score vector, and
        (3) keep a hard floor. Oracle π is still estimated by Monte-Carlo on
        the resulting design — never by clipping after the fact.
        """
        s = np.asarray(scores, dtype=np.float64).copy()
        s = np.maximum(s, 1e-8)
        s = s / s.max(axis=1, keepdims=True)
        # Uniform blend α controls severity while protecting positivity.
        # moderate: stronger overlap; severe: more structure, still no true zeros.
        alpha = self.uniform_blend
        s = (1.0 - alpha) * s + alpha
        s = np.maximum(s, self.positivity_floor * 0.5)
        return s

    @staticmethod
    def _min_class_mass(scores: np.ndarray) -> float:
        return float(scores.min())

    @staticmethod
    def _default_site_ids(n: int) -> np.ndarray:
        return np.arange(n) % 3

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def _sample_uniform_fixed_size(self, n: int, c: int, k: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        mask = np.zeros((n, c), dtype=np.uint8)
        for i in range(n):
            chosen = rng.choice(c, size=k, replace=False)
            mask[i, chosen] = 1
        return mask

    def _sample_gumbel_topk(self, scores: np.ndarray) -> np.ndarray:
        """Weighted fixed-size sampling via Gumbel top-k on log-scores.

        This is a proper Plackett-Luce / exponential-race style design:
        exactly k successes per row, with selection propensity driven by scores.
        """
        rng = np.random.default_rng(self.seed)
        s = np.asarray(scores, dtype=np.float64)
        s = np.clip(s, 1e-12, None)
        logits = np.log(s)
        u = rng.uniform(size=s.shape)
        gumbel = -np.log(-np.log(u))
        keys = logits + gumbel
        topk = np.argpartition(-keys, self.labels_per_volume - 1, axis=1)[
            :, : self.labels_per_volume
        ]
        mask = np.zeros_like(s, dtype=np.uint8)
        rows = np.arange(s.shape[0])[:, None]
        mask[rows, topk] = 1
        return mask

    def _mc_inclusion_from_scores(self, n: int, scores: np.ndarray) -> np.ndarray:
        k = self.labels_per_volume

        def sample_fn(rng: np.random.Generator) -> np.ndarray:
            s = np.clip(scores, 1e-12, None)
            logits = np.log(s)
            u = rng.uniform(size=s.shape)
            gumbel = -np.log(-np.log(u))
            keys = logits + gumbel
            topk = np.argpartition(-keys, k - 1, axis=1)[:, :k]
            m = np.zeros_like(s, dtype=np.uint8)
            rows = np.arange(s.shape[0])[:, None]
            m[rows, topk] = 1
            return m

        return monte_carlo_inclusion_prob(
            sample_fn,
            num_volumes=n,
            num_classes=self.num_classes,
            n_draws=self.n_mc_draws,
            seed=self.seed + 991,
        )


def generate_word_masks(
    *,
    num_volumes: int,
    num_classes: int = 16,
    labels_per_volume: int = 2,
    modes: tuple[Mode, ...] = ("random", "longtail", "clustered", "conditional"),
    mask_seeds: tuple[int, ...] = (0, 1, 2),
    severity: Severity = "moderate",
    feature_dim: int = 8,
    positivity_floor: float = 0.05,
    n_mc_draws: int = 1000,
) -> dict[str, SimulatorOutput]:
    """Convenience generator for E1 mask packs (multiple mask-seeds)."""
    rng = np.random.default_rng(12345)
    features = rng.normal(size=(num_volumes, feature_dim))
    site_ids = np.arange(num_volumes) % 3
    out: dict[str, SimulatorOutput] = {}
    for mode in modes:
        for seed in mask_seeds:
            key = f"{mode}_2of{num_classes}_maskseed{seed}_{severity}"
            sampler = MissingnessSampler(
                num_classes=num_classes,
                labels_per_volume=labels_per_volume,
                mode=mode,
                severity=severity,
                seed=seed,
                positivity_floor=positivity_floor,
                n_mc_draws=n_mc_draws,
            )
            out[key] = sampler.generate(
                volume_ids=[f"word_{i:04d}" for i in range(num_volumes)],
                features=features if mode == "conditional" else None,
                site_ids=site_ids if mode == "clustered" else None,
            )
    return out
