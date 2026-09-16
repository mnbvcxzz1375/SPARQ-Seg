#!/usr/bin/env python3
"""AOVA Phase A: teacher-entropy scores for un-annotated (v,c) pairs.

Loads a trained PL-Seg teacher (default: an E1 arm's final_model.pth), runs
one non-overlapping sliding-window pass over every TRAINING volume, and for
each class c computes the mean predictive Shannon entropy inside the teacher
argmax region of class c on volume v (volume mean if region empty).

Only image intensities and teacher predictions are read - never hidden GT.

Output npz: entropy [N, C] float32 (column c <-> organ class c+1, matching the
E1/AOVA mask column order), volume_ids.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PLSEG_ROOT = os.environ.get(
    "PLSEG_CODE_ROOT",
    "/public/home/heyecheng/word_paper_registry_20260904/code",
)
if PLSEG_ROOT not in sys.path:
    sys.path.insert(0, PLSEG_ROOT)

import torch  # noqa: E402
from models.Proposed.unet3d_ProgCon import UNet3D_ProgCon  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-root", required=True, help="base pack dir (for volume order)")
    ap.add_argument("--teacher-ckpt", required=True, help="final_model.pth of teacher run")
    ap.add_argument("--images-root", required=True, help="WORD imagesTr")
    ap.add_argument("--out", required=True, help="output entropy npz path")
    ap.add_argument("--patch", type=int, nargs=3, default=[128, 128, 96])
    ap.add_argument("--num-classes", type=int, default=17)
    args = ap.parse_args()

    import SimpleITK as sitk

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = {
        "in_chns": 1,
        "class_num": args.num_classes,
        "feature_chns": [16, 32, 64, 128],
        "dropout": [0, 0, 0.1, 0.2],
        "trilinear": True,
    }
    model = UNet3D_ProgCon(params).to(device)
    ck = torch.load(args.teacher_ckpt, map_location=device)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state)
    model.eval()

    vids = [ln.strip() for ln in
            (Path(args.pack_root) / "volume_order.txt").read_text(encoding="utf-8").splitlines()
            if ln.strip()]
    dz, dy, dx = args.patch
    K = args.num_classes
    entropy = np.zeros((len(vids), K - 1), dtype=np.float32)

    for i, vid in enumerate(vids):
        img = sitk.GetArrayFromImage(
            sitk.ReadImage(str(Path(args.images_root) / f"{vid}.nii.gz"))
        ).astype(np.float32)
        mn, mx = float(img.min()), float(img.max())
        if mx > mn:
            img = (img - mn) / (mx - mn)
        D, H, W = img.shape
        pd, ph, pw = (-D) % dz, (-H) % dy, (-W) % dx
        imgp = np.pad(img, ((0, pd), (0, ph), (0, pw)), mode="edge")
        p_acc = np.zeros((K,) + imgp.shape, dtype=np.float32)
        for z0 in range(0, imgp.shape[0], dz):
            for y0 in range(0, imgp.shape[1], dy):
                for x0 in range(0, imgp.shape[2], dx):
                    win = imgp[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]
                    with torch.no_grad():
                        out = model(torch.from_numpy(win[None, None]).to(device))
                        prob = torch.softmax(out[0] if isinstance(out, (list, tuple))
                                             else out, dim=1)[0].cpu().numpy()
                    p_acc[:, z0:z0 + dz, y0:y0 + dy, x0:x0 + dx] += prob
        p = p_acc[:, :D, :H, :W]
        pred = p.argmax(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.log(np.clip(p, 1e-10, None))
        ent = -np.where(p > 0, p * logp, 0.0).sum(axis=0)  # [D,H,W]
        vm = float(ent.mean())
        for c in range(1, K):
            reg = pred == c
            entropy[i, c - 1] = float(ent[reg].mean()) if reg.any() else vm
        if (i + 1) % 10 == 0 or i == len(vids) - 1:
            print(f"[{i + 1}/{len(vids)}] scored", flush=True)

    np.savez_compressed(args.out, entropy=entropy, volume_ids=np.array(vids))
    print(f"WROTE {args.out} shape={entropy.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
