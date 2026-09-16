"""Shared single-step training function for PL-Seg clones (calibration).

Extracted VERBATIM (semantics) from sparq/e1/train_e1_arm.py @ 8a4169d.
The equivalence suite (test_step_equivalence.py) proves it computes exactly
what the legacy inline block computed, before any recipe change.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def forward_losses(model, img, onehot, hadfs, distill, cocl_fn, cw, alpha):
    """Compute the official PL-Seg training loss for one batch.

    Returns (total, parts, raw_list). parts values are tensors, except
    parts["debug"]["sup_scales"] = per-branch HADFL losses as floats
    (diagnostic only; not part of any computation).
    """
    outputs = model(img)
    raw_list = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
    if len(raw_list) < 4:
        raise RuntimeError(f"expected 4 decoder branches, got {len(raw_list)}")
    softs = [
        torch.clamp(torch.softmax(t, dim=1), min=1e-10, max=1.0)
        for t in raw_list[:4]
    ]
    sup_terms = []
    for i in range(4):
        lab_i = (
            onehot
            if i == 0
            else F.interpolate(onehot, size=raw_list[i].shape[2:], mode="nearest")
        )
        sup_terms.append(hadfs[i](softs[i], lab_i))
    sup = sum(sup_terms) / 4.0
    pseudo_terms = []
    for i in range(3):
        teacher = F.interpolate(
            raw_list[i].detach(),
            size=raw_list[i + 1].shape[2:],
            mode="trilinear",
            align_corners=True,
        )
        pseudo_terms.append(distill(raw_list[i + 1], teacher))
    pseudo = sum(pseudo_terms) / 3.0
    cocr = cocl_fn(softs[0])
    total = sup + cw * pseudo + alpha * cocr
    parts = {"sup": sup, "pseudo": pseudo, "cocl": cocr, "total": total}
    parts["debug"] = {"sup_scales": [float(x.detach()) for x in sup_terms]}
    return total, parts, raw_list


def train_step(model, img, onehot, opt, hadfs, distill, cocl_fn, cw, alpha,
               grad_clip=5.0):
    """One optimizer step. Returns (parts_float, finite)."""
    opt.zero_grad(set_to_none=True)
    total, parts, _ = forward_losses(model, img, onehot, hadfs, distill, cocl_fn,
                                     cw, alpha)
    if not torch.isfinite(total):
        opt.zero_grad(set_to_none=True)
        return {k: float("nan") for k in ("sup", "pseudo", "cocl", "total")}, False
    total.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    opt.step()
    return {k: float(parts[k].detach())
            for k in ("sup", "pseudo", "cocl", "total")}, True
