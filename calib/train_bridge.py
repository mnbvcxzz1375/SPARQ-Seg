"""MCAR bridge experiment: ONE training arm under a fully-specified
REFERENCE recipe (official train_PLSeg semantics), on a frozen E1 random mask,
to establish a credible baseline before any propensity/AOVA conclusion.

Reference recipe locked here (all explicit, no "official" hand-waving):
- crop: D=patch_z x H=patch_y x W=patch_x (official TrainerCrop uses
  patch_size[::-1]); start = randint(0, dim-patch) inclusive; retry while the
  cropped IMAGE is constant; per-patch min-max normalization (official
  ToTensor order: crop THEN normalize).
- batch=4, drop_last; epoch = one pass over the 100 train volumes
  (25 updates/epoch); 500 epochs => 12,500 updates (vs pilot 100,000).
- Adam(betas=(0.9,0.99)) lr=0.01, NO weight decay, NO gradient clipping.
- loss: shared calib.train_step (4xHADFL clamped softmax + Distill PSD T=10
  + CO_Contrastive*alpha), consistency ramp 0.1*sigmoid_rampup(epoch, total).
- plateau: ReduceLROnPlateau(mode="max", factor=0.95, patience=4, min_lr=1e-4)
  stepped on mean ALL-17-class val Dice, validated every val_interval=2 epochs
  via vendor test_single_case_fourpre with official validate strides
  (stride_xy=patch_x, stride_z=patch_z).
- checkpoints: BOTH best (by val mean_all17, official >= rule) and final.
- mask: frozen E1_v1 pack (NEW-MASK CALIBRATION semantics, not a W3 replay);
  provenance + runtime import SHAs recorded in run_manifest.json.

DO NOT run until calib_out/equivalence.json passes and w3_anchor.json has been
reviewed (anchor convention first). Budget/purpose: one arm, ~9h on 4090D.

Usage (GPU):
  python calib/train_bridge.py --pack-root <E1_v1> --pack-name random_moderate_s0.npz \
    --images-root <imagesTr> --labels-full-root <labelsTr_All> \
    --val-images-root <imagesVal> --val-labels-root <labelsVal> \
    --out-dir <runs/CALIB_v1/bridge_mcar_official_r0>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PLSEG_ROOT = os.environ.get("PLSEG_CODE_ROOT",
    "/public/home/heyecheng/word_paper_registry_20260904/code")
if PLSEG_ROOT not in sys.path:
    sys.path.insert(0, PLSEG_ROOT)
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from models.Proposed.unet3d_ProgCon import UNet3D_ProgCon  # noqa: E402
from models.Proposed.sing_class_loss import HADFLoss  # noqa: E402
from models.weight_init import initialize_weights  # noqa: E402
from utils.losses import DistillationLoss, CO_Contrastive  # noqa: E402
from utils.val3D import test_single_case_fourpre  # noqa: E402
from utils.evaluation_seg import get_multi_class_evaluation_score  # noqa: E402
from utils.ramps import sigmoid_rampup  # noqa: E402

from calib.train_step import forward_losses  # noqa: E402

PARAMS = {"in_chns": 1, "class_num": 17,
          "feature_chns": [16, 32, 64, 128], "dropout": [0, 0, 0.1, 0.2],
          "trilinear": True}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class BridgeCropDataset(Dataset):
    """Official TrainerCrop+ToTensor order: crop FIRST (D=patch_z), then
    min-max normalize the crop. 1 sample per volume per epoch (len=N)."""

    def __init__(self, images_root, labels_full_root, volume_ids, amask,
                 patch_xyz=(128, 128, 96)):
        self.images_root = Path(images_root)
        self.labels_full_root = Path(labels_full_root)
        self.volume_ids = list(volume_ids)
        self.amask = np.asarray(amask, dtype=np.uint8)
        # official TrainerCrop: zip(shape(D,H,W), patch[::-1])
        px, py, pz = patch_xyz
        self.dz, self.dy, self.dx = pz, py, px  # D=96,H=128,W=128 default
        self._cache = {}

    def __len__(self):
        return len(self.volume_ids)

    def _load(self, vid):
        if vid not in self._cache:
            import SimpleITK as sitk
            img = sitk.GetArrayFromImage(sitk.ReadImage(
                str(self.images_root / f"{vid}.nii.gz"))).astype(np.float32)
            lab = sitk.GetArrayFromImage(sitk.ReadImage(
                str(self.labels_full_root / f"{vid}.nii.gz"))).astype(np.uint8)
            self._cache[vid] = (img, lab, self.amask[self.volume_ids.index(vid)])
        return self._cache[vid]

    def __getitem__(self, idx):
        vid = self.volume_ids[idx]
        img, full, amask = self._load(vid)
        lab = full.copy()
        for c in range(1, amask.size + 1):
            if not amask[c - 1]:
                lab[full == c] = 0
        dz, dy, dx = self.dz, self.dy, self.dx
        for _ in range(200):  # official unbounded retry; bounded here
            z0 = np.random.randint(0, max(img.shape[0] - dz, 0) + 1)
            y0 = np.random.randint(0, max(img.shape[1] - dy, 0) + 1)
            x0 = np.random.randint(0, max(img.shape[2] - dx, 0) + 1)
            ic = img[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]
            lc = lab[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]
            if ic.shape == (dz, dy, dx) and ic.max() != ic.min():
                break
        else:
            raise RuntimeError(f"volume {vid} smaller than crop {dz,dy,dx}; "
                               "pad policy must be made explicit")
        mn, mx = float(ic.min()), float(ic.max())
        if mx > mn:
            ic = (ic - mn) / (mx - mn)
        onehot = np.zeros((17, dz, dy, dx), dtype=np.float32)
        for c in range(17):
            onehot[c] = (lc == c).astype(np.float32)
        present = set(int(x) for x in np.unique(lc) if x != 0)
        allowed = {c for c, ok in enumerate(amask, start=1) if ok}
        if not present <= allowed:
            raise RuntimeError(f"hidden leak in {vid}: {present - allowed}")
        return (torch.from_numpy(np.ascontiguousarray(ic[None, None].astype(np.float32)))[0],
                torch.from_numpy(onehot))


def evaluate(model, val_pairs, device, stride_xy, stride_z, patch_xyz):
    import SimpleITK as sitk
    model.eval()
    per = []
    for ip, lp in val_pairs:
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(ip))).astype(np.float32)
        mn, mx = float(img.min()), float(img.max())
        if mx > mn:
            img = (img - mn) / (mx - mn)
        with torch.no_grad():
            pred = test_single_case_fourpre(model, torch.from_numpy(img).to(device),
                                            stride_xy=stride_xy, stride_z=stride_z,
                                            patch_size=list(patch_xyz),
                                            num_classes=17)
        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(lp))).astype(np.uint8)
        d = np.asarray(get_multi_class_evaluation_score(
            pred, gt, list(range(17)), False, "dice"), dtype=np.float64)
        per.append({"case": ip.name, "all17": float(d.mean()),
                    "fg16": float(d[1:].mean()), "per_class": d.tolist()})
    model.train()
    return per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-root", required=True)
    ap.add_argument("--pack-name", default="random_moderate_s0.npz")
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--labels-full-root", required=True)
    ap.add_argument("--val-images-root", required=True)
    ap.add_argument("--val-labels-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epoches", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--consistency", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--temperature", type=float, default=10)
    ap.add_argument("--patch", type=int, nargs=3, default=[128, 128, 96])
    ap.add_argument("--val-interval", type=int, default=2)
    ap.add_argument("--opt-seed", type=int, default=42)
    ap.add_argument("--allow-nonempty", action="store_true",
                    help="explicit override to continue into a non-empty "
                         "out-dir; bridge policy is NO-RESUME, so prefer a "
                         "fresh directory")
    args = ap.parse_args()

    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        if not args.allow_nonempty:
            print(f"REFUSING non-empty out-dir {out} (bridge policy: no "
                  f"resume, no mixing; use a fresh directory or pass "
                  f"--allow-nonempty explicitly)")
            return 2
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        print("already completed")
        return 0

    torch.manual_seed(args.opt_seed)
    torch.cuda.manual_seed_all(args.opt_seed)
    np.random.seed(args.opt_seed)
    random.seed(args.opt_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")

    pack_dir = Path(args.pack_root)
    man = json.loads((pack_dir / "manifest.json").read_text(encoding="utf-8"))
    npz = pack_dir / args.pack_name
    if sha256_file(npz) != man["packs"][args.pack_name]["sha256"]:
        raise RuntimeError("pack sha mismatch")
    with np.load(npz) as z:
        amask = z["mask"]
    vids = [ln.strip() for ln in
            (pack_dir / "volume_order.txt").read_text().splitlines()
            if ln.strip()]
    # ---- startup assertions (submission gates, not experiments) ----
    assert amask.shape == (100, 16), f"mask shape {amask.shape}"
    assert set(np.unique(amask).tolist()) <= {0, 1}, "mask not binary"
    assert (amask.sum(axis=1) == 2).all(), "mask rows must sum to 2 (2/16)"
    assert len(set(vids)) == 100, "train volume ids not unique/100"
    val_ids = {p.name for p in Path(args.val_images_root).glob("*.nii.gz")}
    assert len(val_ids) == 20, f"expected 20 val cases, got {len(val_ids)}"
    assert not (set(vids) & {v.replace('.nii.gz', '') for v in val_ids}), \
        "train/val overlap"
    order_hash = sha256_file(pack_dir / "volume_order.txt")
    pm_order = man.get("volume_order_sha256")
    assert pm_order is None or pm_order == order_hash, \
        "volume_order.txt does not match pack manifest"

    run_manifest = {
        "purpose": "MCAR bridge, REFERENCE recipe (see module docstring); "
                   "NO-RESUME policy: interruption -> fresh directory",
        "pack": {"path": str(npz), "sha256": sha256_file(npz),
                 "mask_seed": man["packs"][args.pack_name].get("mask_seed")},
        "recipe": {"crop_DHW_from_patch": [args.patch[2], args.patch[1],
                                           args.patch[0]],
                   "batch": args.batch_size, "updates_per_epoch":
                   len(vids) // args.batch_size, "epoches": args.epoches,
                   "optimizer": "Adam(0.9,0.99)", "lr": args.lr,
                   "grad_clip": None, "plateau": "mode=max on val mean_all17",
                   "val_interval": args.val_interval,
                   "val_stride_train_select": [args.patch[0], args.patch[2]],
                   "final_eval_stride": [64, 64],
                   "consistency": args.consistency, "alpha": args.alpha,
                   "temperature": args.temperature, "opt_seed": args.opt_seed},
        "val_roots": [str(args.val_images_root), str(args.val_labels_root)],
        "software": {"python": sys.version, "torch": torch.__version__,
                     "cuda": torch.version.cuda,
                     "gpu": torch.cuda.get_device_name(0)},
        "source_hashes": {
            "calib/train_bridge.py": sha256_file(Path(__file__)),
            "calib/train_step.py": sha256_file(REPO / "calib" / "train_step.py")},
        "runtime_imports": {},
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for mod in (UNet3D_ProgCon, HADFLoss, DistillationLoss, CO_Contrastive,
                test_single_case_fourpre, get_multi_class_evaluation_score,
                sigmoid_rampup, initialize_weights):
        f = sys.modules[mod.__module__].__file__
        run_manifest["runtime_imports"][mod.__name__] = {
            "file": f, "sha256": sha256_file(f)}
    (out / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8")

    ds = BridgeCropDataset(args.images_root, args.labels_full_root, vids, amask,
                           tuple(args.patch))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, drop_last=True, pin_memory=True,
                        generator=torch.Generator().manual_seed(args.opt_seed))
    vi = sorted(Path(args.val_images_root).glob("*.nii.gz"))
    val_pairs = [(p, Path(args.val_labels_root) / p.name) for p in vi
                 if (Path(args.val_labels_root) / p.name).exists()]
    if len(val_pairs) != len(vi):
        raise RuntimeError("val image/label id mismatch")

    model = UNet3D_ProgCon(PARAMS).to(device)
    initialize_weights(model, "xavier")
    hadfs = [HADFLoss(num_classes=17) for _ in range(4)]
    distill = DistillationLoss(T=args.temperature)
    opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.95, patience=4, min_lr=1e-4)

    best = float("-inf")
    log = out / "train.log"
    with open(log, "a", encoding="utf-8") as lf:
        for epoch in range(args.epoches):
            model.train()
            tot_sum, nb, skipped = 0.0, 0, 0
            cw = args.consistency * sigmoid_rampup(epoch, args.epoches)
            for bi, (img, onehot) in enumerate(loader):
                img, onehot = img.to(device), onehot.to(device)
                opt.zero_grad(set_to_none=True)
                total, _, _ = forward_losses(model, img, onehot, hadfs, distill,
                                             CO_Contrastive, cw, args.alpha)
                if not torch.isfinite(total):
                    # calibration arm: non-finite loss is a stop condition,
                    # not something to skip (audit round-2 requirement)
                    (out / "DIAGNOSTIC_nonfinite_loss").write_text(
                        f"epoch={epoch} batch={bi} loss={float(total)}\n")
                    return 4
                total.backward()
                bad = [n for n, p in model.named_parameters()
                       if p.grad is not None and not torch.isfinite(p.grad).all()]
                if bad:
                    (out / "DIAGNOSTIC_nonfinite_grad").write_text(
                        f"epoch={epoch} batch={bi} params={bad[:8]}\n")
                    return 4
                opt.step()          # NO grad clip (reference recipe)
                tot_sum += float(total.detach())
                nb += 1
            if nb == 0:
                (out / "FAILED").write_text(f"all non-finite epoch {epoch}")
                return 4
            avg = tot_sum / max(nb, 1)
            line = f"epoch={epoch} train_loss={avg:.6f} skipped={skipped} "
            if (epoch + 1) % args.val_interval == 0:
                per = evaluate(model, val_pairs, device,
                               args.patch[0], args.patch[2], args.patch)
                v_all = float(np.mean([c["all17"] for c in per]))
                v_fg = float(np.mean([c["fg16"] for c in per]))
                sched.step(v_all)
                line += f"val_all17={v_all:.6f} val_fg16={v_fg:.6f} "
                json.dump(per, open(out / f"val_epoch{epoch:04d}.json", "w"))
                if v_all >= best:
                    best = v_all
                    torch.save(model.state_dict(), out / "best_model.pth")
                    line += "BEST "
            line += f"lr={opt.param_groups[0]['lr']:.6g}"
            lf.write(line + "\n")
            lf.flush()
            print(line, flush=True)

    torch.save(model.state_dict(), out / "final_model.pth")
    # final report: both checkpoints under the LOCKED 64/64 evaluator so the
    # bridge is comparable to W3/E1 numbers without mixing stride conventions
    final_eval = {}
    for cname in ("best_model.pth", "final_model.pth"):
        p = out / cname
        if not p.exists():
            continue
        ck = torch.load(p, map_location=device)
        model.load_state_dict(ck)
        per = evaluate(model, val_pairs, device, 64, 64, args.patch)
        final_eval[cname] = {
            "mean_all17": float(np.mean([c["all17"] for c in per])),
            "mean_fg16": float(np.mean([c["fg16"] for c in per])),
            "per_case": per,
            "sha256": sha256_file(p),
        }
    (out / "final_eval_stride64.json").write_text(
        json.dumps(final_eval, indent=2), encoding="utf-8")
    (out / "DONE").write_text(json.dumps(
        {"best_val_all17(train-select-stride)": best,
         "final_eval_stride64": {k: v["mean_fg16"] for k, v in final_eval.items()},
         "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
