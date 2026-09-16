"""Anchor alignment: recompute the W3 PL-Seg record from its own checkpoint
on both validation-case roots, under both metric conventions. No retraining.

Outputs mean_all17 (BG-inclusive, as train_PLSeg validate_slice reports
torch.mean over 17 classes) and mean_fg16 per (checkpoint, case root), plus
per-case/per-organ detail, checkpoint SHA and runtime module SHAs.

Decides: (1) is the ~0.7598 anchor reproducible from best_model.pth at
stride 64/64; (2) do the W3 data_view cases and our locked share cases
agree (ids + label bytes); (3) how much metric convention alone explains.

Usage (GPU):
  python calib/w3_eval_pair.py --out calib_out/w3_anchor.json
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

from models.Proposed.unet3d_ProgCon import UNet3D_ProgCon  # noqa: E402
from utils.val3D import test_single_case_fourpre  # noqa: E402
from utils.evaluation_seg import get_multi_class_evaluation_score  # noqa: E402

PARAMS = {"in_chns": 1, "class_num": 17,
          "feature_chns": [16, 32, 64, 128], "dropout": [0, 0, 0.1, 0.2],
          "trilinear": True}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_eval(ckpt_path, device):
    model = UNet3D_ProgCon(PARAMS).to(device)
    ck = torch.load(ckpt_path, map_location=device)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state)
    model.eval()
    return model


def eval_case(model, sitk, img_p, lab_p, stride, patch, device):
    img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_p))).astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) / (mx - mn)
    with torch.no_grad():
        pred = test_single_case_fourpre(model, torch.from_numpy(img).to(device),
                                        stride_xy=stride, stride_z=stride,
                                        patch_size=patch, num_classes=17)
    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(lab_p))).astype(np.uint8)
    dice = np.asarray(get_multi_class_evaluation_score(
        pred, gt, list(range(17)), False, "dice"), dtype=np.float64)
    return dice


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w3-dir",
                    default="/public/home/heyecheng/word_paper_registry_20260904/"
                            "exp/w3_plseg_2of16")
    ap.add_argument("--w3-data-view",
                    default="/public/home/heyecheng/word_paper_registry_20260904/"
                            "data_views/WORD_2of16")
    ap.add_argument("--share-root",
                    default="/public/share/td20230405/WORD")
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--patch", type=int, nargs=3, default=[128, 128, 96])
    ap.add_argument("--out", default="calib_out/w3_anchor.json")
    args = ap.parse_args()

    import SimpleITK as sitk
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"stride": args.stride, "patch": args.patch,
              "runtime_imports": {}, "runs": [], "case_identity": {}}
    for mod in (UNet3D_ProgCon, test_single_case_fourpre,
                get_multi_class_evaluation_score):
        f = sys.modules[mod.__module__].__file__
        report["runtime_imports"][mod.__name__] = {"file": f,
                                                   "sha256": sha256_file(f)}

    roots = {
        "w3_data_view": (Path(args.w3_data_view) / "imagesVal",
                         Path(args.w3_data_view) / "labelsVal"),
        "locked_share": (Path(args.share_root) / "imagesVal",
                         Path(args.share_root) / "labelsVal"),
    }
    ids_by_root, bytes_mismatch = {}, []
    for root, (ir, lr) in roots.items():
        if not ir.exists():
            report.setdefault("skipped_roots", []).append(str(ir))
            continue
        cases = sorted(p.name for p in ir.glob("*.nii.gz"))
        ids_by_root[root] = cases
        for name in cases:
            lp = lr / name
            if lp.exists() and sha256_file(lp) != sha256_file(
                    roots["locked_share"][1] / name if (
                        roots["locked_share"][1] / name).exists() else lp):
                bytes_mismatch.append(name)
    common = set.intersection(*[set(v) for v in ids_by_root.values()]) \
        if ids_by_root else set()
    report["case_identity"] = {
        "counts": {k: len(v) for k, v in ids_by_root.items()},
        "common_ids": len(common),
        "id_sets_equal": bool(len(ids_by_root) > 1 and all(
            set(v) == set(next(iter(ids_by_root.values())))
            for v in ids_by_root.values())),
        "label_bytes_mismatching": bytes_mismatch,
    }

    ckpts = {"best": Path(args.w3_dir) / "checkpoint/models/best_model.pth",
             "final": Path(args.w3_dir) / "checkpoint/models/final_model.pth"}
    for cname, path in ckpts.items():
        if not path.exists():
            continue
        model = load_eval(path, device)
        for root, (ir, lr) in roots.items():
            if not ir.exists():
                continue
            cases = sorted(p.name for p in ir.glob("*.nii.gz"))
            per_case = []
            for name in cases:
                dice = eval_case(model, sitk, ir / name, lr / name,
                                 args.stride, list(args.patch), device)
                per_case.append({"case": name, "all17": float(dice.mean()),
                                 "fg16": float(dice[1:].mean()),
                                 "per_class": dice.tolist()})
            m_all = float(np.mean([c["all17"] for c in per_case]))
            m_fg = float(np.mean([c["fg16"] for c in per_case]))
            report["runs"].append({
                "checkpoint": cname, "path": str(path),
                "sha256": sha256_file(path), "case_root": root,
                "n_cases": len(cases),
                "mean_all17": m_all, "mean_fg16": m_fg,
                "per_case": per_case,
            })
            print(f"{cname}@{root}: mean_all17={m_all:.6f} mean_fg16={m_fg:.6f}",
                  flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
