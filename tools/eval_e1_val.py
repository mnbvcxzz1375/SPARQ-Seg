#!/usr/bin/env python3
"""E1 arm evaluation on locked WORD imagesVal.

Mirrors validate_slice/test_PLSeg.py from the vendored official tree:
whole-volume min-max normalization (Test_ToTensor), sliding-window inference
via test_single_case_fourpre, per-class Dice via
get_multi_class_evaluation_score. Writes runs/E1_v1/<arm>/val_dice.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PLSEG_ROOT = os.environ.get(
    "PLSEG_CODE_ROOT",
    "/public/home/heyecheng/word_paper_registry_20260904/code",
)
if PLSEG_ROOT not in sys.path:
    sys.path.insert(0, PLSEG_ROOT)

from models.Proposed.unet3d_ProgCon import UNet3D_ProgCon  # noqa: E402
from utils.val3D import test_single_case_fourpre  # noqa: E402
from utils.evaluation_seg import get_multi_class_evaluation_score  # noqa: E402

ORGANS = [
    "bg", "Liver", "Spleen", "L.Kidney", "R.Kidney", "Stomach", "Gallbladder",
    "Esophagus", "Pancreas", "Duodenum", "Colon", "Intestine", "Adrenal",
    "Rectum", "Bladder", "L.Hip", "R.Hip",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out-dir", required=True, help="runs/E1_v1/<arm>")
    ap.add_argument("--checkpoint", default="final_model.pth")
    ap.add_argument("--images-root", required=True, help="WORD imagesVal")
    ap.add_argument("--labels-root", required=True, help="WORD labelsVal")
    ap.add_argument("--stride-xy", type=int, default=64)
    ap.add_argument("--stride-z", type=int, default=64)
    ap.add_argument("--patch", type=int, nargs=3, default=[128, 128, 96])
    ap.add_argument("--num-classes", type=int, default=17)
    args = ap.parse_args()

    import SimpleITK as sitk
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    ckpt_path = out_dir / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    if (out_dir / "val_dice.json").exists():
        print(f"already evaluated: {args.arm}")
        return 0

    params = {
        "in_chns": 1,
        "class_num": args.num_classes,
        "feature_chns": [16, 32, 64, 128],
        "dropout": [0, 0, 0.1, 0.2],
        "trilinear": True,
    }
    model = UNet3D_ProgCon(params).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state)
    model.eval()

    images = sorted(Path(args.images_root).glob("*.nii.gz"))
    labels = sorted(Path(args.labels_root).glob("*.nii.gz"))
    names = {p.name for p in labels}
    cases = [p for p in images if p.name in names]
    if not cases:
        raise RuntimeError(f"no matched image/label pairs under {args.images_root}")

    per_case = []
    per_class_sum = np.zeros(args.num_classes, dtype=np.float64)
    for i, img_p in enumerate(cases):
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_p))).astype(np.float32)
        mn, mx = float(img.min()), float(img.max())
        if mx > mn:
            img = (img - mn) / (mx - mn)
        image = torch.from_numpy(img).to(device)
        with torch.no_grad():
            pred = test_single_case_fourpre(
                model, image, args.stride_xy, args.stride_z,
                list(args.patch), num_classes=args.num_classes,
            )
        gt = sitk.GetArrayFromImage(
            sitk.ReadImage(str(Path(args.labels_root) / img_p.name))
        ).astype(np.uint8)
        dice = np.asarray(
            get_multi_class_evaluation_score(
                pred, gt, list(range(args.num_classes)), False, "dice"
            ),
            dtype=np.float64,
        )
        per_class_sum += dice
        per_case.append(
            {"case": img_p.name, "mean_all17": float(dice.mean()),
             "mean_fg16": float(dice[1:].mean()), "per_class": dice.tolist()}
        )
        print(f"[{i + 1}/{len(cases)}] {img_p.name} fg16={dice[1:].mean():.4f}",
              flush=True)

    per_class_mean = (per_class_sum / len(cases)).tolist()
    result = {
        "arm": args.arm,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": hashlib.sha256(ckpt_path.read_bytes()).hexdigest(),
        "n_cases": len(cases),
        "stride_xy": args.stride_xy,
        "stride_z": args.stride_z,
        "patch": list(args.patch),
        "num_classes": args.num_classes,
        "organs": ORGANS[: args.num_classes],
        "per_class_mean_dice": per_class_mean,
        "mean_all17": float(np.mean(per_class_mean)),
        "mean_fg16": float(np.mean(per_class_mean[1:])),
        "per_case": per_case,
        "torch": torch.__version__,
        "device": str(device),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "val_dice.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"WROTE {out_dir / 'val_dice.json'} mean_fg16={result['mean_fg16']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
