"""Contract/QC verifier for frozen WORD E1-v1 mask packs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-root", required=True)
    ap.add_argument("--positivity-floor", type=float, default=0.05)
    args = ap.parse_args()
    root = Path(args.pack_root)
    man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    if man.get("protocol") != "WORD_E1_v1":
        errors.append(f"protocol={man.get('protocol')}")
    if man.get("num_volumes") != 100 or man.get("num_classes") != 16:
        errors.append("N/C mismatch")
    if man.get("labels_per_volume") != 2:
        errors.append("k != 2")
    if man.get("severity") != "moderate":
        errors.append("severity != moderate")
    if man.get("mnar_included") or man.get("severe_included"):
        errors.append("mnar/severe must be excluded from E1-v1")
    if not man.get("conditional_requires_word_images", False):
        errors.append("conditional pack must be image-only WORD-based")
    if "no_labels" not in man.get("conditional_feature_method", ""):
        errors.append("conditional feature method must declare no_labels")

    packs = man.get("packs", {})
    if len(packs) != 12:
        errors.append(f"expected 12 packs, got {len(packs)}")

    qc_rows = json.loads((root / "qc_report.json").read_text(encoding="utf-8"))
    if len(qc_rows) != 12:
        errors.append("qc_report rows != 12")

    for fname, meta in packs.items():
        path = root / fname
        if not path.exists():
            errors.append(f"missing {fname}")
            continue
        digest = sha256_file(path)
        if digest != meta.get("sha256"):
            errors.append(f"sha mismatch {fname}")
        z = np.load(path)
        mask = z["mask"]
        score = z["selection_score"]
        pi = z["inclusion_prob"]
        if mask.shape != (100, 16):
            errors.append(f"{fname} mask shape {mask.shape}")
        if not np.all(mask.sum(axis=1) == 2):
            errors.append(f"{fname} budget violated")
        if score.shape != (100, 16) or pi.shape != (100, 16):
            errors.append(f"{fname} score/pi shape")
        if float(pi.min()) < args.positivity_floor - 0.008:
            errors.append(f"{fname} pi_min={float(pi.min()):.4f}")
        if np.allclose(score, pi):
            errors.append(f"{fname} selection_score equals inclusion_prob")

    expected = {
        f"{p}_moderate_s{s}.npz"
        for p in ("random", "longtail", "sitelike", "conditional")
        for s in (0, 1, 2)
    }
    if set(packs) != expected:
        errors.append(f"pack name set mismatch: {set(packs) ^ expected}")

    report = {
        "ok": not errors,
        "errors": errors,
        "phase0_git_commit": man.get("phase0_git_commit"),
        "phase0_tag": man.get("phase0_tag"),
        "num_packs": len(packs),
        "min_pi_overall": min(
            float(np.load(root / f)["inclusion_prob"].min()) for f in packs
        ),
    }
    out = root / "contract_verify.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
