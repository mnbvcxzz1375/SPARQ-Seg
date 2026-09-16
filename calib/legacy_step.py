"""ORACLE: verbatim copy of the inline training step from
sparq/e1/train_e1_arm.py at commit 8a4169d (current pilot code).

Do not refactor this file; it exists only so test_step_equivalence can diff
the extracted shared calib/train_step.py against the exact legacy semantics.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def legacy_step_forward(model, img, onehot, hadfs, distill, cw, args_alpha):
    """The legacy inline block from train_e1_arm.py @8a4169d (verbatim
    structure): clamp-softmax -> 4xHADFL(nearest-downsampled onehot, mean) ->
    3x DistillationLoss on detached trilinear(align_corners=True) logits,
    mean -> CO_Contrastive(branch-1 softmax) -> total."""
    from utils.losses import CO_Contrastive  # vendor, resolved at runtime
    outputs = model(img)
    if isinstance(outputs, (tuple, list)):
        raw_list = list(outputs)
    else:
        raw_list = [outputs]
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
    cocr = CO_Contrastive(softs[0])
    total = sup + cw * pseudo + args_alpha * cocr
    parts = {"sup": sup, "pseudo": pseudo, "cocl": cocr, "total": total}
    parts["debug"] = {"sup_scales": [float(x.detach()) for x in sup_terms]}
    return total, parts, raw_list
