"""Diagnose the "silent collapse" (train healthy, eval all-background).

Loads a COLLAPSED checkpoint (e.g. runs/R1_v1/e1_random_s0/final_model.pth)
and a healthy twin for contrast, then on one locked validation case:
  (a) plain model.eval()            -> official eval path (running stats)
  (b) BN train-mode, dropout eval   -> batch-stats inference
and dumps per-layer BatchNorm running-stat health (mean/var extremes).

If (a) predicts all-background while (b) predicts organs, the root cause is
degenerate BN running statistics (train/eval divergence), not the weights.

Usage (GPU):
  python calib/diag_collapse.py \
    --collapsed runs/R1_v1/e1_random_s0/final_model.pth \
    --healthy runs/R1_v1/e1_random_s1/final_model.pth \
    --val-images-root ... --val-labels-root ... \
    --out calib_out/collapse_diag.json
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
import torch.nn as nn  # noqa: E402

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
from utils.val3D import test_single_case_fourpre  # noqa: E402

PARAMS = {"in_chns": 1, "class_num": 17,
          "feature_chns": [16, 32, 64, 128], "dropout": [0, 0, 0.1, 0.2],
          "trilinear": True}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bn_report(model):
    rows = []
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm3d):
            rm, rv, nb = m.running_mean, m.running_var, m.num_batches_tracked
            rows.append({
                "layer": name,
                "mean_min": float(rm.min()), "mean_max": float(rm.max()),
                "var_min": float(rv.min()), "var_max": float(rv.max()),
                "var_median": float(rv.median()),
                "finite": bool(rm.isfinite().all() and rv.isfinite().all()),
                "batches_tracked": int(nb),
            })
    return rows


def predict(model, img_t, patch):
    with torch.no_grad():
        pred = test_single_case_fourpre(model, img_t, stride_xy=64,
                                        stride_z=64, patch_size=patch,
                                        num_classes=17)
    return np.asarray(pred)


def set_bn_train_drop_eval(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            m.train()          # inference with batch statistics
        elif isinstance(m, nn.Dropout):
            m.eval()           # keep dropout off


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collapsed", required=True)
    ap.add_argument("--healthy", default="")
    ap.add_argument("--val-images-root", required=True)
    ap.add_argument("--val-labels-root", required=True)
    ap.add_argument("--case", default="word_0003.nii.gz")
    ap.add_argument("--patch", type=int, nargs=3, default=[128, 128, 96])
    ap.add_argument("--out", default="calib_out/collapse_diag.json")
    args = ap.parse_args()

    import SimpleITK as sitk
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = sitk.GetArrayFromImage(sitk.ReadImage(
        str(Path(args.val_images_root) / args.case))).astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) / (mx - mn)
    gt = sitk.GetArrayFromImage(sitk.ReadImage(
        str(Path(args.val_labels_root) / args.case))).astype(np.uint8)
    img_t = torch.from_numpy(img).to(device)

    report = {"case": args.case, "models": {}}
    for tag, ckpt in (("collapsed", args.collapsed), ("healthy", args.healthy)):
        if not ckpt:
            continue
        model = UNet3D_ProgCon(PARAMS).to(device)
        sd = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(sd["model"] if "model" in sd else sd)
        entry = {"sha256": sha256_file(Path(ckpt)), "bn_stats": bn_report(model)}

        model.eval()
        pe = predict(model, img_t, list(args.patch))
        set_bn_train_drop_eval(model)
        pb = predict(model, img_t, list(args.patch))

        entry["eval_mode"] = {
            "bg_fraction": float((pe == 0).mean()),
            "pred_classes": sorted(int(c) for c in np.unique(pe)),
        }
        entry["bn_batchstats_mode"] = {
            "bg_fraction": float((pb == 0).mean()),
            "pred_classes": sorted(int(c) for c in np.unique(pb)),
        }
        entry["gt_classes"] = sorted(int(c) for c in np.unique(gt))
        entry["verdict"] = (
            "BN running-stats divergence" if
            entry["eval_mode"]["bg_fraction"] > 0.999
            and entry["bn_batchstats_mode"]["bg_fraction"] < 0.95
            else "inconclusive")
        report["models"][tag] = {k: entry[k] for k in
                                 ("sha256", "eval_mode",
                                  "bn_batchstats_mode", "gt_classes", "verdict")}
        report["models"][tag]["bn_stats_summary"] = {
            "n_layers": len(entry["bn_stats"]),
            "any_nonfinite": any(not r["finite"] for r in entry["bn_stats"]),
            "var_min_min": min(r["var_min"] for r in entry["bn_stats"]),
            "var_max_max": max(r["var_max"] for r in entry["bn_stats"]),
        }
        report["models"][tag]["bn_stats"] = entry["bn_stats"]
        print(tag, json.dumps({k: report["models"][tag][k] for k in
              ("eval_mode", "bn_batchstats_mode", "verdict",
               "bn_stats_summary")}, indent=2), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
