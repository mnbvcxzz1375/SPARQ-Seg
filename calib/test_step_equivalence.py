"""Step-1 calibration: numeric-consistency check between the extracted shared
train_step and the legacy inline block (8a4169d).

Method (audit round 2):
- fixture: legal single-class one-hot labels built from class-ID maps, with
  an assert on sum(dim=1)==1 (the previous fixture was malformed: ~5.2% of
  voxels had empty or multi-hot channel vectors);
- three branches: legacy, legacy again (self-difference baseline = GPU
  execution noise), shared;
- losses must match bit-exactly; gradients / parameter updates are judged by
  a TOLERANCE-BASED NUMERIC-CONSISTENCY CHECK against the self-difference
  reference (this is NOT a proof of bitwise faithfulness, and the report
  says so);
- diagnostics: top parameter-update differences mapped to parameter names
  with gradient magnitudes, per-branch HADFL per-scale losses, and
  torch.use_deterministic_algorithms(warn_only) captured warnings to locate
  known-non-deterministic ops.

Run on a GPU host (vendor HADFLoss criteria are CUDA-bound):
  python calib/test_step_equivalence.py --out calib_out/equivalence.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

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
PARAMS = {"in_chns": 1, "class_num": 17,
          "feature_chns": [16, 32, 64, 128], "dropout": [0, 0, 0.1, 0.2],
          "trilinear": True}


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
    """Legal single-class labels: class-ID map first, one-hot second."""
    g = torch.Generator().manual_seed(99)
    img = torch.rand(2, 1, 64, 64, 64, generator=g).to(device)
    lab = torch.zeros(2, 64, 64, 64, dtype=torch.long)
    m2 = torch.rand(2, 64, 64, 64, generator=g) > 0.93
    lab[m2] = 2
    m7 = (torch.rand(2, 64, 64, 64, generator=g) > 0.94) & ~m2
    lab[m7] = 7
    m13 = (torch.rand(2, 64, 64, 64, generator=g) > 0.95) & ~m2 & ~m7
    lab[m13] = 13
    onehot = F.one_hot(lab, num_classes=17).permute(0, 4, 1, 2, 3).float()
    assert torch.all(onehot.sum(dim=1) == 1), "fixture not single-class"
    assert (onehot[:, 1:].sum() > 0), "fixture has no foreground"
    return img, onehot.to(device)


def flat_grads(model):
    return torch.cat([p.grad.detach().flatten() for p in model.parameters()
                      if p.grad is not None])


def flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


def param_name_for(model, flat_idx):
    off = 0
    for name, p in model.named_parameters():
        n = p.numel()
        if off <= flat_idx < off + n:
            return name, flat_idx - off
        off += n
    return "?", -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="calib_out/equivalence.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("HADFLoss criteria require CUDA; run on a GPU host")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    det_warnings = []

    img, onehot = make_batch(device)
    cw, alpha = 0.35, 0.1

    report = {"device": str(device), "input_shape_DHW": list(img.shape[2:]),
              "cw": cw, "alpha": alpha, "runtime_imports": {},
              "deterministic_warnings": det_warnings}
    for mod in (UNet3D_ProgCon, HADFLoss, DistillationLoss, CO_Contrastive):
        f = sys.modules[mod.__module__].__file__
        report["runtime_imports"][mod.__name__] = {"file": f,
                                                   "sha256": sha256_file(f)}

    def run_branch(kind):
        m, h, d, o = build_branch(device)
        torch.manual_seed(SEED + 7)
        m.train()
        o.zero_grad(set_to_none=True)
        with warnings.catch_warnings(record=True) as wlog:
            warnings.simplefilter("always")
            if kind == "legacy":
                t, parts, _ = legacy_step_forward(m, img, onehot, h, d, cw, alpha)
            else:
                t, parts, _ = forward_losses(m, img, onehot, h, d, CO_Contrastive,
                                             cw, alpha)
            t.backward()
            for w in wlog:
                det_warnings.append(f"[{kind}] {w.category.__name__}: {w.message}")
        g = flat_grads(m)
        p0 = flat_params(m)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
        o.step()
        return parts, g, p0, flat_params(m) - p0, \
            [[hh.clone() for hh in hadf.batch_loss_history] for hadf in h], m

    a1 = run_branch("legacy")
    a2 = run_branch("legacy")
    b = run_branch("shared")

    def cmp(x, y):
        x, y = x.float(), y.float()
        return {"max_abs_diff": float((x - y).abs().max()),
                "exact_equal": bool(torch.equal(x, y))}

    report["initial_params_exact"] = bool(torch.equal(a1[2].cpu(), b[2].cpu()))
    report["losses"] = {k: cmp(a1[0][k].detach().cpu(), b[0][k].detach().cpu())
                        for k in ("sup", "pseudo", "cocl", "total")}
    report["losses_selfdiff"] = {k: cmp(a1[0][k].detach().cpu(),
                                        a2[0][k].detach().cpu())
                                 for k in ("sup", "pseudo", "cocl", "total")}
    report["sup_scales"] = {"legacy": a1[0]["debug"]["sup_scales"],
                            "shared": b[0]["debug"]["sup_scales"],
                            "selfdiff_legacy2": a2[0]["debug"]["sup_scales"]}
    report["gradients"] = cmp(a1[1].cpu(), b[1].cpu())
    report["gradients_selfdiff"] = cmp(a1[1].cpu(), a2[1].cpu())
    report["param_update"] = cmp(a1[3].cpu(), b[3].cpu())
    report["param_update_selfdiff"] = cmp(a1[3].cpu(), a2[3].cpu())
    report["hadfl_history_equal"] = bool(
        all(len(ha) == len(hb)
            and all(torch.equal(x, y) for x, y in zip(ha, hb))
            for ha, hb in zip(a1[4], b[4])))

    # diagnostics: largest parameter-update differences, named
    dupd = (a1[3] - b[3]).abs().cpu()
    k = min(5, dupd.numel())
    vals, idxs = dupd.flatten().topk(k)
    report["top_update_diff_params"] = [
        {"param": param_name_for(b[5], int(i))[0],
         "grad_max_abs_preclip": float(a1[1][i].abs()),
         "grad_cross_diff": float(a1[1][i].abs() - b[1][i].abs())
         if a1[1][i] != b[1][i] else 0.0,
         "update_abs_diff": float(v)}
        for v, i in zip(vals.tolist(), idxs.tolist())
    ]

    l_ok = all(v["exact_equal"] for v in report["losses"].values())
    hist_ok = report["hadfl_history_equal"]

    # TOLERANCE-BASED NUMERIC-CONSISTENCY (not a bitwise-faithfulness proof):
    # cross diffs must sit at the level of legacy's own re-execution noise.
    def within_noise(cross, selfd, scale=2.0, floor=1e-6):
        return (cross["exact_equal"]
                or cross["max_abs_diff"] <= max(selfd["max_abs_diff"] * scale,
                                                floor))

    g_ok = within_noise(report["gradients"], report["gradients_selfdiff"])
    p_ok = within_noise(report["param_update"], report["param_update_selfdiff"],
                        scale=4.0)
    report["gate"] = {
        "initial_params_exact": report["initial_params_exact"],
        "losses_exact": bool(l_ok), "hadfl_history_equal": hist_ok,
        "grads_numeric_consistency": bool(g_ok),
        "param_update_numeric_consistency": bool(p_ok),
        "note": "tolerance-based numeric consistency vs one self-difference "
                "reference; single-seed-diff is neither a strict lower bound "
                "nor a proven upper bound of execution noise",
    }
    ok = bool(report["gate"]["initial_params_exact"] and l_ok and hist_ok
              and g_ok and p_ok)
    report["equivalence"] = ok

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("equivalence", "gate", "sup_scales", "gradients",
                       "param_update", "gradients_selfdiff",
                       "param_update_selfdiff", "top_update_diff_params",
                       "deterministic_warnings")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
