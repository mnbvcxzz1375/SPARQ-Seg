"""Partial-label dataset contract: volume-level annotation vs patch presence.

Phase 0 implements the visibility contract and a numpy/torch-optional batch
collate. It does not load WORD volumes yet; callers pass arrays so that unit
tests can prove hidden organs never enter `partial_label`.

Roadmap v1.1:
- annotation_mask: volume-level human annotation availability M[n, ·]
- presence_mask:   patch-level organ presence (has positive voxels in crop)
- full_label is oracle-only; training path must only see partial_label
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


BACKGROUND = 0


@dataclass(frozen=True)
class PartialSample:
    image: np.ndarray
    partial_label: np.ndarray
    annotation_mask: np.ndarray
    presence_mask: np.ndarray
    volume_id: str
    full_label: np.ndarray | None = None  # oracle/debug only

    def as_train_dict(self) -> dict[str, Any]:
        """Training-facing view. Never includes full_label."""
        return {
            "image": self.image,
            "partial_label": self.partial_label,
            "annotation_mask": self.annotation_mask,
            "presence_mask": self.presence_mask,
            "volume_id": self.volume_id,
        }

    def as_oracle_dict(self) -> dict[str, Any]:
        d = self.as_train_dict()
        d["full_label"] = self.full_label
        return d


def apply_annotation_mask(
    full_label: np.ndarray,
    annotation_mask: np.ndarray,
    *,
    unannotated_value: int = BACKGROUND,
    num_foreground: int | None = None,
) -> np.ndarray:
    """Zero-out (or fill) organs that are not annotated in this volume.

    Classes are 0..C with 0=background. annotation_mask is [C] for foreground
    organs 1..C, or [C+1] including background (background always visible).
    Prefer passing `num_foreground` explicitly; do not infer C from a single
    volume's max label (that volume may omit high class ids).
    """
    lab = np.asarray(full_label)
    m = np.asarray(annotation_mask)
    if lab.ndim < 3:
        raise ValueError("full_label must be at least 3D spatial")
    if m.ndim != 1:
        raise ValueError("annotation_mask must be 1D")

    if num_foreground is None:
        # Inference only as fallback; still require mask consistency.
        num_foreground = int(m.size) if m.size and int(lab.max(initial=0)) <= m.size else int(lab.max(initial=0))

    if m.size == num_foreground:
        keep = np.concatenate([[True], m.astype(bool)])
    elif m.size == num_foreground + 1:
        keep = m.astype(bool).copy()
        keep[BACKGROUND] = True
    else:
        raise ValueError(
            f"annotation_mask size {m.size} incompatible with num_foreground {num_foreground}"
        )

    out = lab.copy()
    for cls, ok in enumerate(keep):
        if cls == BACKGROUND:
            continue
        if not ok:
            out[lab == cls] = unannotated_value
    return out


def patch_presence_mask(
    label_patch: np.ndarray,
    num_foreground: int,
) -> np.ndarray:
    """Binary presence of each foreground organ inside a patch label."""
    lab = np.asarray(label_patch)
    pres = np.zeros(num_foreground, dtype=np.uint8)
    for c in range(1, num_foreground + 1):
        if np.any(lab == c):
            pres[c - 1] = 1
    return pres


class PartialLabelView:
    """Index over volumes with a fixed annotation matrix M [N, C_fg]."""

    def __init__(
        self,
        volume_ids: Sequence[str],
        images: Sequence[np.ndarray],
        full_labels: Sequence[np.ndarray] | None,
        annotation_matrix: np.ndarray,
        *,
        allow_full_label_in_train: bool = False,
    ) -> None:
        self.volume_ids = list(volume_ids)
        self.images = list(images)
        self.full_labels = None if full_labels is None else list(full_labels)
        self.annotation_matrix = np.asarray(annotation_matrix, dtype=np.uint8)
        if self.annotation_matrix.ndim != 2:
            raise ValueError("annotation_matrix must be [N, C_fg]")
        if len(self.volume_ids) != self.annotation_matrix.shape[0]:
            raise ValueError("annotation_matrix rows must match volumes")
        if len(self.images) != len(self.volume_ids):
            raise ValueError("images length mismatch")
        if self.full_labels is not None and len(self.full_labels) != len(self.volume_ids):
            raise ValueError("full_labels length mismatch")
        self.allow_full_label_in_train = bool(allow_full_label_in_train)
        if self.allow_full_label_in_train:
            raise ValueError(
                "allow_full_label_in_train=True is forbidden; full_label is oracle-only"
            )

    def __len__(self) -> int:
        return len(self.volume_ids)

    def _apply(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.full_labels is None:
            raise RuntimeError("no full_label store configured")
        full = np.asarray(self.full_labels[index])
        amask = self.annotation_matrix[index]
        partial = apply_annotation_mask(full, amask, num_foreground=amask.size)
        return full, amask, partial

    def get(self, index: int, *, oracle: bool = False) -> PartialSample:
        full, amask, partial = self._apply(index)
        return PartialSample(
            image=np.asarray(self.images[index]),
            partial_label=partial,
            annotation_mask=amask.copy(),
            presence_mask=patch_presence_mask(partial, amask.size),
            volume_id=self.volume_ids[index],
            full_label=full if oracle else None,
        )

    def train_sample(self, index: int) -> dict[str, Any]:
        """Only legal training payload."""
        _, amask, partial = self._apply(index)
        return {
            "image": np.asarray(self.images[index]),
            "partial_label": partial,
            "annotation_mask": amask.copy(),
            "presence_mask": patch_presence_mask(partial, amask.size),
            "volume_id": self.volume_ids[index],
        }

    def assert_no_hidden_leak(self, index: int) -> None:
        """Fail if any unannotated organ voxels survive in partial_label."""
        payload = self.train_sample(index)
        partial = payload["partial_label"]
        amask = payload["annotation_mask"]
        for c, ok in enumerate(amask, start=1):
            if not ok and np.any(partial == c):
                raise AssertionError(
                    f"hidden organ {c} leaked into partial_label for volume {index}"
                )


def save_mask_pack(path: str, simulator_output: dict[str, Any]) -> None:
    """Save a simulator output pack to .npz (masks + meta arrays)."""
    payload = {
        "mask": simulator_output["mask"],
        "selection_score": simulator_output["selection_score"],
        "inclusion_prob": simulator_output["inclusion_prob"],
    }
    meta = simulator_output["pattern_meta"]
    # store JSON-ish meta fields that are array-like as object-safe scalars
    for key in (
        "mode",
        "severity",
        "seed",
        "budget",
        "labels_per_volume",
        "gini",
        "entropy",
        "positivity_floor",
        "inclusion_min",
        "inclusion_max",
    ):
        if key in meta:
            payload[f"meta_{key}"] = np.asarray(meta[key])
    if "class_coverage" in meta:
        payload["meta_class_coverage"] = np.asarray(meta["class_coverage"], dtype=np.float64)
    np.savez_compressed(path, **payload)


def load_mask_pack(path: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        out: dict[str, Any] = {
            "mask": z["mask"],
            "selection_score": z["selection_score"],
            "inclusion_prob": z["inclusion_prob"],
        }
        meta = {}
        for key in z.files:
            if key.startswith("meta_"):
                val = z[key]
                meta[key[len("meta_") :]] = val.item() if val.shape == () else val.tolist()
        out["pattern_meta"] = meta
    return out
