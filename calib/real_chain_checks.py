"""Step-2 calibration: REAL training-chain checks (replaces the surrogate
preflight's coverage; uses the actual vendor model/losses via the shared step).

Checks:
1. hidden-GT counterfactual: perturbing ONLY hidden-class voxels in the full
   GT must leave the masked data view, the loss, gradients and HADFL history
   EXACTLY unchanged, and must touch >0 voxels.
2. visible positive control: perturbing visible GT must change the loss.
3. input shape log (D,H,W as actually fed to model) + runtime import
   fingerprints (real __file__ + sha256 of model/loss modules).

Run on a GPU host. Usage:
  python calib/real_chain_checks.py --out calib_out/real_chain.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

PLSEG_ROOT = os.environ.get(
    "PLSEG_CODE_ROOT",
    "/public/home/heyecheng/word_paper_registry_20260904/code",
)
if PLSEG_ROOT not in sys.path:
    sys.path.insert(0, PLSEG_ROOT)
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.Proposed.unet3d_ProgCon import UNet3D_ProgCon  # noqa: E402
from models.Proposed.sing_class_loss import HADFLoss  # noqa: E402
from models.weight_init import initialize_weights  # noqa: E402
from utils.losses import DistillationLoss, CO_Contrastive  # noqa: E402

from calib.train_step import forward_losses  # noqa: E402

SEED = 2026
PARAMS = {"in_chns": 1, "class_num": 17,
          "feature_chns": [16, 32, 64, 128], "dropout": [0, 0, 0.1, 0.2],
          "trilinear": True}
VISIBLE = (2, 7)          # annotated classes
HIDDEN_PERT_FROM, HIDDEN_PERT_TO = 13, 15
CWL, ALPHA = 0.35, 0.1


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    m = UNet3D_ProgCon(PARAMS).to(device)
    initialize_weights(m, "xavier")
    return m, [HADFLoss(num_classes=17) for _ in range(4)], DistillationLoss(T=10)


def onehot_from(lab, device):
    oh = torch.zeros(1, 17, *lab.shape, device=device)
    for c in range(17):
        oh[0, c] = (lab == c).float()
    return oh


def run_case(model, hadfs, distill, img, lab_view, device):
    """One forward+backward through the REAL chain on a masked label view.
    Fresh model/hadfs required for exact comparison (history mutates)."""
    torch.manual_seed(SEED + 11)
    model.train()
    oh = onehot_from(lab_view, device)
    total, parts, _ = forward_losses(model, img, oh, hadfs, distill,
                                     CO_Contrastive, CWL, ALPHA)
    model.zero_grad(set_to_none=True)
    total.backward()
    grads = torch.cat([p.grad.detach().flatten() for p in model.parameters()
                       if p.grad is not None])
    hist = [h.clone() for hadf in hadfs for h in hadf.batch_loss_history]
    return {k: float(v.detach()) for k, v in parts.items()}, grads, hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="calib_out/real_chain.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("requires CUDA (vendor HADFLoss criteria)")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    report = {"runtime_imports": {}}
    for mod in (UNet3D_ProgCon, HADFLoss, DistillationLoss, CO_Contrastive):
        f = sys.modules[mod.__module__].__file__
        report["runtime_imports"][mod.__name__] = {"file": f,
                                                   "sha256": sha256_file(f)}

    g = torch.Generator().manual_seed(SEED)
    D, H, W = 64, 80, 96   # asymmetric on purpose to catch axis mixups
    report["input_DHW"] = [D, H, W]
    img = torch.rand(1, 1, D, H, W, generator=g).to(device)

    gg = torch.Generator().manual_seed(SEED)
    lab_full = torch.randint(0, 2, (D, H, W), generator=gg).to(device)
    lab_full = torch.where(lab_full == 1, torch.full_like(lab_full, 2), lab_full)
    m7 = torch.rand((D, H, W), generator=gg) > 0.94
    m13 = torch.rand((D, H, W), generator=gg) > 0.96
    lab_full = torch.where(m7.to(device), torch.full_like(lab_full, 7), lab_full)
    lab_full = torch.where(m13.to(device) & ~m7.to(device),
                           torch.full_like(lab_full, HIDDEN_PERT_FROM), lab_full)

    vis_t = torch.tensor(list(VISIBLE), device=device)

    def view_of(lab):
        return torch.where(torch.isin(lab, vis_t), lab, torch.zeros_like(lab))

    # hidden-only perturbation of the FULL GT
    n_pert = int((lab_full == HIDDEN_PERT_FROM).sum())
    report["hidden_perturbed_voxels"] = n_pert
    assert n_pert > 0, "hidden perturbation touches zero voxels (bad test)"
    lab_pert_full = torch.where(lab_full == HIDDEN_PERT_FROM,
                                torch.full_like(lab_full, HIDDEN_PERT_TO),
                                lab_full)
    view_a, view_b = view_of(lab_full), view_of(lab_pert_full)
    report["data_view_identical_under_hidden_pert"] = bool(torch.equal(view_a, view_b))

    # visible perturbation positive control (acts ON the view)
    m_vis = view_a == 7
    cut = torch.rand((D, H, W), generator=torch.Generator(device="cpu")
                     .manual_seed(5)).to(device) > 0.5
    view_c = torch.where(m_vis & cut, torch.zeros_like(view_a), view_a)
    report["visible_control_changed_voxels"] = int((view_c != view_a).sum())
    assert report["visible_control_changed_voxels"] > 0

    mA, hA, dA = build(device)
    c1_l, g1, h1 = run_case(mA, hA, dA, img, view_a, device)
    mB, hB, dB = build(device)
    c1b_l, g1b, h1b = run_case(mB, hB, dB, img, view_a, device)  # noise baseline
    mC, hC, dC = build(device)
    c2_l, g2, h2 = run_case(mC, hC, dC, img, view_b, device)
    mD, hD, dD = build(device)
    c3_l, _, _ = run_case(mD, hD, dD, img, view_c, device)

    def maxdiff(x, y):
        return float((x - y).abs().max())

    grad_noise = maxdiff(g1, g1b)            # same input twice: GPU atomics noise
    grad_cross = maxdiff(g1, g2)             # hidden-GT perturbed
    hidden_invariant_losses = c1_l == c2_l
    hidden_invariant_grads = grad_cross <= max(grad_noise * 2.0, 1e-6)
    hidden_invariant_hist = len(h1) == len(h2) and all(
        torch.equal(a, b) for a, b in zip(h1, h2))
    visible_changes = any(abs(c1_l[k] - c3_l[k]) > 1e-9 for k in ("sup", "total"))

    report.update({
        "loss_viewA": c1_l, "loss_hidden_pert": c2_l, "loss_visible_pert": c3_l,
        "grad_noise_baseline_max": grad_noise,
        "grad_cross_max": grad_cross,
        "hidden_counterfactual": {
            "loss_exact": hidden_invariant_losses,
            "grads_within_noise": hidden_invariant_grads,
            "hadfl_history_exact": hidden_invariant_hist,
        },
        "visible_positive_control_changes_loss": visible_changes,
        "note": "backward atomics are non-deterministic on GPU; gradient "
                "equality judged against a same-input self-difference "
                "baseline (×2), losses/history judged exactly",
    })
    ok = bool(hidden_invariant_losses and hidden_invariant_grads
              and hidden_invariant_hist and visible_changes)
    report["pass"] = ok

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("pass", "hidden_counterfactual",
                       "visible_positive_control_changes_loss",
                       "hidden_perturbed_voxels", "input_DHW")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
