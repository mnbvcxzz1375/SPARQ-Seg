"""Step-1 calibration: prove the extracted shared train_step is compute-
equivalent to the legacy inline block (8a4169d) BEFORE any recipe change.

Run on a GPU host (vendor HADFLoss instantiates CUDA criteria). Two branches
build identical initial state independently; RNG is re-seeded before each
branch so dropout masks coincide. Compares: 4-way supervised losses, PSD
loss, COCL loss, total, gradients, post-step parameter update, HADFL history.

Usage (school):  python calib/test_step_equivalence.py --out calib_out/equivalence.json
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

from calib.legacy_step import legacy_step_forward  # noqa: E402
from calib.train_step import forward_losses  # noqa: E402

SEED = 1234
PARAMS = {
    "in_chns": 1, "class_num": 17,
    "feature_chns": [16, 32, 64, 128], "dropout": [0, 0, 0.1, 0.2],
    "trilinear": True,
}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_branch(device):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = UNet3D_ProgCon(PARAMS).to(device)
    initialize_weights(model, "xavier")
    hadfs = [HADFLoss(num_classes=17) for _ in range(4)]
    distill = DistillationLoss(T=10)
    opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=0.01)
    return model, hadfs, distill, opt


def make_batch(device):
    g = torch.Generator().manual_seed(99)
    # (D,H,W) small but valid through 4 downsamples
    img = torch.rand(2, 1, 64, 64, 64, generator=g).to(device)
    onehot = torch.zeros(2, 17, 64, 64, 64)
    onehot[:, 0] = 1.0
    for cls in (2, 7, 13):  # Spleen / Esophagus / Rectum
        onehot[:, 0, cls - 1] = 0.0
        m = torch.rand(2, 64, 64, 64, generator=g) > 0.93
        onehot[:, cls] = m.float()
        onehot[:, 0] *= (~m).float()
    onehot = onehot.to(device)
    return img, onehot


def flat_grads(model):
    return torch.cat([p.grad.detach().flatten() for p in model.parameters()
                      if p.grad is not None])


def flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="calib_out/equivalence.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("HADFLoss criteria require CUDA; run on a GPU host")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    img, onehot = make_batch(device)
    cw, alpha = 0.35, 0.1

    report = {"device": str(device), "input_shape_DHW": list(img.shape[2:]),
              "cw": cw, "alpha": alpha,
              "runtime_imports": {}}
    for mod in (UNet3D_ProgCon, HADFLoss, DistillationLoss, CO_Contrastive):
        f = sys.modules[mod.__module__].__file__
        report["runtime_imports"][mod.__name__] = {"file": f,
                                                   "sha256": sha256_file(f)}

    def run_branch(kind):
        """Fresh identical branch; execute one step. kind: 'legacy'|'shared'."""
        m, h, d, o = build_branch(device)
        torch.manual_seed(SEED + 7)
        m.train()
        o.zero_grad(set_to_none=True)
        if kind == "legacy":
            t, parts, _ = legacy_step_forward(m, img, onehot, h, d, cw, alpha)
        else:
            t, parts, _ = forward_losses(m, img, onehot, h, d, CO_Contrastive,
                                         cw, alpha)
        t.backward()
        g = flat_grads(m)                 # pre-clip gradients
        p0 = flat_params(m)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        o.step()
        p1 = flat_params(m)
        hist = [[hh.clone() for hh in hadf.batch_loss_history] for hadf in h]
        return parts, g, p0, p1 - p0, hist

    a1 = run_branch("legacy")
    a2 = run_branch("legacy")   # self-difference baseline: pure GPU noise
    b = run_branch("shared")

    def cmp(x, y):
        x, y = x.float(), y.float()
        return {"max_abs_diff": float((x - y).abs().max()),
                "exact_equal": bool(torch.equal(x, y))}

    report["losses_selfdiff"] = {k: cmp(a1[0][k].detach().cpu(),
                                        a2[0][k].detach().cpu())
                                 for k in a1[0]}
    report["gradients_selfdiff"] = cmp(a1[1].cpu(), a2[1].cpu())
    report["param_update_selfdiff"] = cmp(a1[3].cpu(), a2[3].cpu())
    report["initial_params_exact"] = bool(torch.equal(a1[2].cpu(), b[2].cpu()))
    report["losses"] = {k: cmp(a1[0][k].detach().cpu(), b[0][k].detach().cpu())
                        for k in a1[0]}
    report["gradients"] = cmp(a1[1].cpu(), b[1].cpu())
    report["param_update"] = cmp(a1[3].cpu(), b[3].cpu())
    report["hadfl_history_equal"] = bool(
        all(len(ha) == len(hb)
            and all(torch.equal(x, y) for x, y in zip(ha, hb))
            for ha, hb in zip(a1[4], b[4])))

    # gate: losses must be exact; grads/update cross-diffs must sit at GPU
    # noise level (bounded by the legacy self-difference baseline), since
    # bit-exactness across distinct cuDNN executions is not achievable.
    def within_noise(cross, selfd, scale=2.0):
        return (cross["exact_equal"]
                or cross["max_abs_diff"] <= max(selfd["max_abs_diff"] * scale,
                                                1e-6))

    l_ok = all(v["exact_equal"] for v in report["losses"].values())
    g_ok = within_noise(report["gradients"], report["gradients_selfdiff"])
    p_ok = within_noise(report["param_update"], report["param_update_selfdiff"],
                        scale=4.0)
    report["gate"] = {"losses_exact": bool(l_ok),
                      "grads_within_noise": bool(g_ok),
                      "param_update_within_noise": bool(p_ok),
                      "note": "bit-exactness across separate cuDNN executions "
                              "is not achievable; noise bounded by legacy "
                              "self-difference (audit F2 step-1 method)"}
    ok = bool(l_ok and g_ok and p_ok and report["hadfl_history_equal"])
    report["equivalence"] = ok
    report["equivalence_exact"] = bool(
        ok and report["gradients"]["exact_equal"]
        and report["param_update"]["exact_equal"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("equivalence", "gate", "hadfl_history_equal", "losses",
                       "gradients", "param_update", "gradients_selfdiff",
                       "param_update_selfdiff")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
