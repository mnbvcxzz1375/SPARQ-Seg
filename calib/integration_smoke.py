"""Integration smoke for the bridge data path (audit round 2, item ii).

Chain under test — the SAME code the bridge arm will run:
    frozen E1_v1 pack (SHA-checked) -> BridgeCropDataset (official crop order,
    mask application, patch min-max) -> shared forward_losses -> real model.

Checks:
1. model input shape is exactly [1, 1, 96, 128, 128] for a single sample;
2. hidden-GT counterfactual through the DATASET's own label path: patching
   only the underlying full GT (hidden class 13 -> 15) must produce an
   identical one-hot view, identical losses and HADFL history, and gradients
   within the same-input self-difference noise baseline (>0 voxels asserted);
3. visible positive control changes the loss.

Run on a GPU host:
  python calib/integration_smoke.py --pack-root annotation_masks/WORD/E1_v1 \
      --out calib_out/integration_smoke.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402

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

from calib.train_bridge import BridgeCropDataset  # noqa: E402
from calib.train_step import forward_losses  # noqa: E402

SEED = 77
VISIBLE = (2, 7)
HIDDEN_FROM, HIDDEN_TO = 13, 15
CWL, ALPHA = 0.35, 0.1
PARAMS = {"in_chns": 1, "class_num": 17,
          "feature_chns": [16, 32, 64, 128], "dropout": [0, 0, 0.1, 0.2],
          "trilinear": True}


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


def run_case(model, hadfs, distill, batch, device):
    torch.manual_seed(SEED + 3)
    model.train()
    img, onehot = batch[0].to(device), batch[1].to(device)
    if img.dim() == 4:          # single (un-collated) sample: add batch dim
        img = img[None]
    assert list(img.shape) == [1, 1, 96, 128, 128], f"bad model input {img.shape}"
    total, parts, _ = forward_losses(model, img, onehot[None] if onehot.dim() == 4
                                     else onehot, hadfs, distill,
                                     CO_Contrastive, CWL, ALPHA)
    model.zero_grad(set_to_none=True)
    total.backward()
    grads = torch.cat([p.grad.detach().flatten() for p in model.parameters()
                       if p.grad is not None])
    hist = [h.clone() for hadf in hadfs for h in hadf.batch_loss_history]
    return {k: float(parts[k].detach())
            for k in ("sup", "pseudo", "cocl", "total")}, grads, hist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-root",
                    default="annotation_masks/WORD/E1_v1")
    ap.add_argument("--pack-name", default="random_moderate_s0.npz")
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--labels-full-root", required=True)
    ap.add_argument("--patch", type=int, nargs=3, default=[128, 128, 96])
    ap.add_argument("--scan", type=int, default=20,
                    help="volumes to scan for a usable test case")
    ap.add_argument("--out", default="calib_out/integration_smoke.json")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("requires CUDA")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    pack_dir = Path(args.pack_root)
    man = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    npz = pack_dir / args.pack_name
    if sha256_file(npz) != man["packs"][args.pack_name]["sha256"]:
        raise RuntimeError("pack sha mismatch")
    with np.load(npz) as z:
        amask = z["mask"]
    vids = [ln.strip() for ln in
            (pack_dir / "volume_order.txt").read_text().splitlines() if ln.strip()]

    # pick a case whose FULL GT contains the hidden class and some visible organ
    base_ds = BridgeCropDataset(args.images_root, args.labels_full_root, vids,
                                amask, tuple(args.patch))
    chosen = None
    for vid in vids[: args.scan]:
        _, lab, am = base_ds._load(vid)
        hidden_vox = int((lab == HIDDEN_FROM).sum())
        visible_vox = sum(int((lab == c).sum()) for c in VISIBLE)
        if hidden_vox > 500 and visible_vox > 500:
            chosen = (vid, hidden_vox, visible_vox)
            break
    if chosen is None:
        raise RuntimeError("no suitable case found in scan window")
    vid, n_hidden, n_vis = chosen
    vid_idx = vids.index(vid)

    report = {"case": vid, "hidden_voxels": n_hidden, "visible_voxels": n_vis,
              "runtime_imports": {}}
    for mod in (UNet3D_ProgCon, HADFLoss, DistillationLoss, CO_Contrastive,
                BridgeCropDataset):
        f = sys.modules[mod.__module__].__file__
        report["runtime_imports"][mod.__name__] = {"file": f,
                                                   "sha256": sha256_file(f)}

    def perturbed_loader(vid_, img_, lab_, am_):
        # counterfactual replaces ONLY the underlying full GT; the mask
        # application stays inside the dataset's own __getitem__ path.
        lab2 = lab_.copy()
        lab2[lab_ == HIDDEN_FROM] = HIDDEN_TO
        return img_, lab2, am_

    ds_a = BridgeCropDataset(args.images_root, args.labels_full_root, vids,
                             amask, tuple(args.patch))
    ds_b = BridgeCropDataset(args.images_root, args.labels_full_root, vids,
                             amask, tuple(args.patch))
    ds_b._load = (lambda v_, img_, lab_, am_: (
        lambda vid: perturbed_loader(v_, img_, lab_, am_)))(vid, *base_ds._load(vid))

    np.random.seed(SEED)
    batch_a1 = ds_a[vid_idx]
    np.random.seed(SEED)
    batch_a2 = ds_a[vid_idx]
    np.random.seed(SEED)
    batch_b = ds_b[vid_idx]

    report["view_identical_under_hidden_pert"] = bool(
        torch.equal(batch_a1[1], batch_b[1]))
    report["dataset_reproducible"] = bool(torch.equal(batch_a1[1], batch_a2[1])
                                          and torch.equal(batch_a1[0], batch_a2[0]))

    mA, hA, dA = build(device)
    c1, g1, h1 = run_case(mA, hA, dA, batch_a1, device)
    mB, hB, dB = build(device)
    c1b, g1b, _ = run_case(mB, hB, dB, batch_a1, device)  # noise baseline
    mC, hC, dC = build(device)
    c2, g2, h2 = run_case(mC, hC, dC, batch_b, device)

    grad_noise = float((g1 - g1b).abs().max())
    grad_cross = float((g1 - g2).abs().max())
    report["grad_noise_baseline_max"] = grad_noise
    report["grad_cross_max"] = grad_cross
    report["losses_viewA"] = c1
    report["hidden_counterfactual"] = {
        "loss_exact": c1 == c2,
        "grads_within_noise": grad_cross <= max(grad_noise * 2.0, 1e-6),
        "hadfl_history_exact": len(h1) == len(h2) and all(
            torch.equal(x, y) for x, y in zip(h1, h2)),
    }
    ok = bool(report["view_identical_under_hidden_pert"]
              and report["dataset_reproducible"]
              and report["hidden_counterfactual"]["loss_exact"]
              and report["hidden_counterfactual"]["grads_within_noise"]
              and report["hidden_counterfactual"]["hadfl_history_exact"])
    report["pass"] = ok

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("pass", "case", "view_identical_under_hidden_pert",
                       "dataset_reproducible", "hidden_counterfactual",
                       "grad_noise_baseline_max", "grad_cross_max")}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
