#!/usr/bin/env python3
"""E1-v1 single-arm PL-Seg training with frozen annotation mask.

Contract:
- reads immutable E1_v1 pack (mask + inclusion_prob), never regenerates
- optimization_seed fixed at 42; mask_seed is the experimental variable
- labels come from labelsTr_All then masked; training never sees hidden organs
- propensity OFF, Stage II OFF
- training step mirrors official train_PLSeg.py verbatim:
  * softmax -> clamp(min=1e-10) on every decoder branch
  * 4 independent HADFLoss instances on nearest-downsampled onehot labels,
    supervised_loss = mean of the 4
  * PSD = DistillationLoss(T=10) on raw logits (deep branch detached,
    trilinear align_corners=True), weight = consistency * sigmoid_rampup
  * COCL = CO_Contrastive on branch-1 clamped softmax, weight alpha=0.1
  * Adam(betas=(0.9,0.99)) lr 0.01, xavier init, patch-level min-max
    normalization applied AFTER random crop (official ToTensor)
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# PL-Seg vendor tree (existing school package)
PLSEG_ROOT = os.environ.get(
    "PLSEG_CODE_ROOT",
    "/public/home/heyecheng/word_paper_registry_20260904/code",
)
if PLSEG_ROOT not in sys.path:
    sys.path.insert(0, PLSEG_ROOT)

from models.Proposed.unet3d_ProgCon import UNet3D_ProgCon  # noqa: E402
from models.Proposed.sing_class_loss import HADFLoss  # noqa: E402
from models.weight_init import initialize_weights  # noqa: E402
from utils.losses import *  # noqa: E402,F401,F403
from utils.losses import CO_Contrastive, DistillationLoss  # noqa: E402
from utils.ramps import sigmoid_rampup  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class E1PartialVolumeDataset(Dataset):
    """Random-crop training set with volume-level annotation mask applied."""

    def __init__(
        self,
        images_root: Path,
        labels_full_root: Path,
        volume_ids: list[str],
        annotation_matrix: np.ndarray,
        patch_size=(128, 128, 96),
        num_classes=17,
        samples_per_epoch=250,
        seed=42,
    ):
        self.images_root = images_root
        self.labels_full_root = labels_full_root
        self.volume_ids = list(volume_ids)
        self.amask = np.asarray(annotation_matrix, dtype=np.uint8)
        self.patch = tuple(patch_size)
        self.num_classes = num_classes
        self.n = samples_per_epoch
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def __len__(self):
        return self.n

    def _load(self, vid: str):
        if vid in self._cache:
            return self._cache[vid]
        import SimpleITK as sitk

        img_p = self.images_root / f"{vid}.nii.gz"
        lab_p = self.labels_full_root / f"{vid}.nii.gz"
        # Raw HU: official ToTensor normalizes the CROP, so load unnormalized.
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_p))).astype(np.float32)
        lab = sitk.GetArrayFromImage(sitk.ReadImage(str(lab_p))).astype(np.uint8)
        self._cache[vid] = (img, lab, self.amask[self.volume_ids.index(vid)])
        return self._cache[vid]

    def __getitem__(self, idx):
        vid = self.volume_ids[idx % len(self.volume_ids)]
        img, full, amask = self._load(vid)
        # apply annotation mask: hidden organs -> background
        lab = full.copy()
        for c in range(1, amask.size + 1):
            if not amask[c - 1]:
                lab[full == c] = 0
        dz, dy, dx = self.patch
        # Official TrainerCrop: random crop, retry while the image crop is
        # constant. Use the global numpy RNG, which the DataLoader re-seeds per
        # worker (base seed derives from --optimization-seed), so streams are
        # deterministic yet distinct across workers.
        zs = max(img.shape[0] - dz, 0)
        ys = max(img.shape[1] - dy, 0)
        xs = max(img.shape[2] - dx, 0)
        for _attempt in range(50):
            z0 = int(np.random.randint(0, zs + 1))
            y0 = int(np.random.randint(0, ys + 1))
            x0 = int(np.random.randint(0, xs + 1))
            img_c = img[z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx]
            lab_c = lab[z0 : z0 + dz, y0 : y0 + dy, x0 : x0 + dx]
            if img_c.max() != img_c.min():
                break
        # pad if needed (volumes smaller than the patch)
        if img_c.shape != (dz, dy, dx):
            img_c = np.pad(img_c, [(0, max(0, dz - img_c.shape[0])), (0, max(0, dy - img_c.shape[1])), (0, max(0, dx - img_c.shape[2]))], mode="constant")[:dz, :dy, :dx]
            lab_c = np.pad(lab_c, [(0, max(0, dz - lab_c.shape[0])), (0, max(0, dy - lab_c.shape[1])), (0, max(0, dx - lab_c.shape[2]))], mode="constant")[:dz, :dy, :dx]
        # Official ToTensor: min-max normalization on the cropped patch.
        mn, mx = float(img_c.min()), float(img_c.max())
        if mx > mn:
            img_c = (img_c - mn) / (mx - mn)
        img_c = img_c.astype(np.float32)
        onehot = np.zeros((self.num_classes, dz, dy, dx), dtype=np.float32)
        for c in range(self.num_classes):
            onehot[c] = (lab_c == c).astype(np.float32)
        # purity
        present = sorted(int(x) for x in np.unique(lab_c) if x != 0)
        allowed = {c for c, ok in enumerate(amask, start=1) if ok}
        if not set(present).issubset(allowed):
            raise RuntimeError(f"hidden class leaked in crop of {vid}: {present} vs {sorted(allowed)}")
        return (
            torch.from_numpy(np.ascontiguousarray(img_c[None])),
            torch.from_numpy(onehot),
            torch.tensor(amask, dtype=torch.uint8),
            vid,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--pack-root", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--labels-full-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epoches", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--samples-per-epoch", type=int, default=250)
    ap.add_argument("--optimization-seed", type=int, default=42)
    ap.add_argument("--expected-code-sha", default="")
    ap.add_argument("--consistency", type=float, default=0.1)  # official PSD weight
    ap.add_argument("--alpha", type=float, default=0.1)  # official COCL weight
    ap.add_argument("--temperature", type=float, default=10)  # official DistillationLoss T
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        print(f"Already completed: {args.arm}")
        return 0
    if any(out.iterdir()):
        # allow only empty or fresh
        if not (out / "run_manifest.json").exists():
            print(f"Non-empty unfinished run directory: {out}")
            return 2

    set_seed(args.optimization_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pack_path = Path(args.pack_root) / f"{args.arm}.npz"
    man_path = Path(args.pack_root) / "manifest.json"
    vol_path = Path(args.pack_root) / "volume_order.txt"
    if not pack_path.exists() or not man_path.exists():
        raise FileNotFoundError(pack_path)
    man = json.loads(man_path.read_text(encoding="utf-8"))
    expected_sha = man["packs"][pack_path.name]["sha256"]
    actual_sha = sha256_file(pack_path)
    if actual_sha != expected_sha:
        raise RuntimeError(f"pack sha mismatch {actual_sha} != {expected_sha}")
    vids = [ln.strip() for ln in vol_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    z = np.load(pack_path)
    mask = z["mask"]

    git_sha = ""
    try:
        git_root = Path(__file__).resolve().parents[2]
        git_sha = os.popen(f"git -C {git_root} rev-parse HEAD 2>/dev/null").read().strip()
    except Exception:
        git_sha = ""
    if args.expected_code_sha and git_sha and not git_sha.startswith(args.expected_code_sha):
        raise RuntimeError(f"code drift: {git_sha} != {args.expected_code_sha}")

    run_manifest = {
        "arm": args.arm,
        "pack_path": str(pack_path),
        "pack_sha256": actual_sha,
        "phase0_git_commit": man.get("phase0_git_commit"),
        "git_sha_guess": git_sha,
        "optimization_seed": args.optimization_seed,
        "mask_seed": man["packs"][pack_path.name].get("mask_seed"),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "epoches": args.epoches,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "propensity": False,
        "stage2": False,
        "loss_recipe": "official train_PLSeg mirror: 4xHADFL on clamped softmax, "
                       "logit DistillationLoss PSD, CO_Contrastive COCL",
        "consistency": args.consistency,
        "alpha_cocl": args.alpha,
        "temperature": args.temperature,
        "grad_clip": 5.0,
        "init": "xavier",
        "normalization": "per-patch min-max after crop (official ToTensor)",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")

    ds = E1PartialVolumeDataset(
        Path(args.images_root),
        Path(args.labels_full_root),
        vids,
        mask,
        samples_per_epoch=args.samples_per_epoch,
        seed=args.optimization_seed,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.optimization_seed),
    )

    params = {
        "in_chns": 1,
        "class_num": 17,
        "feature_chns": [16, 32, 64, 128],
        "dropout": [0, 0, 0.1, 0.2],
        "trilinear": True,
    }
    model = UNet3D_ProgCon(params).to(device)
    initialize_weights(model, "xavier")
    # Official uses one HADFLoss per decoder branch (each has its own
    # sliding-window hardness history).
    hadfs = [HADFLoss(num_classes=17) for _ in range(4)]
    distill = DistillationLoss(T=args.temperature)
    # Match official train_PLSeg.py: Adam + plateau (not raw SGD).
    opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.99), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.95, patience=4, min_lr=1e-4
    )

    log_path = out / "train.log"
    best_loss = float("inf")
    with open(log_path, "a", encoding="utf-8") as logf:
        for epoch in range(args.epoches):
            model.train()
            epoch_loss = 0.0
            nb = 0
            skipped = 0
            cw = args.consistency * sigmoid_rampup(epoch, args.epoches)
            for step, (img, onehot, amask, vids_b) in enumerate(loader):
                img = img.to(device)
                onehot = onehot.to(device)
                opt.zero_grad(set_to_none=True)
                outputs = model(img)
                if isinstance(outputs, (tuple, list)):
                    raw_list = list(outputs)
                else:
                    raw_list = [outputs]
                if len(raw_list) < 4:
                    raise RuntimeError(f"expected 4 decoder branches, got {len(raw_list)}")
                # Official step: clamped softmax per branch. The clamp is the
                # NaN fix -- HADFL's decoupled CE takes log(p_foreground) and
                # an unclamped softmax can hit exactly 0 under Adam lr=0.01.
                softs = [
                    torch.clamp(torch.softmax(t, dim=1), min=1e-10, max=1.0)
                    for t in raw_list[:4]
                ]
                # Pyramid Partial Label Supervision: 4-branch HADFL on
                # nearest-downsampled onehot labels, averaged.
                sup_terms = []
                for i in range(4):
                    lab_i = (
                        onehot
                        if i == 0
                        else F.interpolate(onehot, size=raw_list[i].shape[2:], mode="nearest")
                    )
                    sup_terms.append(hadfs[i](softs[i], lab_i))
                sup = sum(sup_terms) / 4.0
                # Progressive Self-Distillation on raw logits (official:
                # trilinear align_corners=True teacher, DistillationLoss).
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
                # Classwise Orthogonal Contrastive Regularization on branch 1.
                cocr = CO_Contrastive(softs[0])
                total = sup + cw * pseudo + args.alpha * cocr
                if not torch.isfinite(total):
                    skipped += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                epoch_loss += float(total.detach())
                nb += 1
            if nb == 0:
                (out / "FAILED").write_text(
                    f"all steps non-finite at epoch {epoch}; skipped={skipped}\n",
                    encoding="utf-8",
                )
                return 4
            if skipped:
                # warn but continue only if some steps were finite
                logf.write(f"epoch={epoch} skipped_nonfinite={skipped}/{nb + skipped}\n")
            avg = epoch_loss / nb
            if not np.isfinite(avg):
                (out / "FAILED").write_text(f"non-finite loss at epoch {epoch}\n", encoding="utf-8")
                return 4
            sched.step(avg)
            line = f"epoch={epoch} loss={avg:.6f} lr={opt.param_groups[0]['lr']:.6g}\n"
            logf.write(line)
            logf.flush()
            print(line, end="", flush=True)
            if avg < best_loss:
                best_loss = avg
                torch.save(model.state_dict(), out / "best_model.pth")
            if (epoch + 1) % 50 == 0:
                torch.save(
                    {"epoch": epoch, "model": model.state_dict(), "optim": opt.state_dict()},
                    out / f"epoch_{epoch:04d}.pth",
                )

    torch.save(model.state_dict(), out / "final_model.pth")
    (out / "DONE").write_text(
        json.dumps({"arm": args.arm, "best_loss": best_loss, "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
