"""Generate immutable WORD E1-v1 annotation mask packs.

Protocol (locked):
  N=100, C=16, k=2, severity=moderate
  patterns: random | longtail | clustered | conditional
  mask_seeds: 0,1,2
  MNAR / severe excluded

Conditional features are image-only descriptors (no segmentation labels).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from code.dataloader.missingness_sampler import MissingnessSampler
from code.utils.annotation_matrix import class_coverage, coverage_entropy, coverage_gini
from code.utils.inclusion_prob import summarize_weights

PATTERN_MAP = {
    "random": "random",
    "longtail": "longtail",
    "sitelike": "clustered",
    "conditional": "conditional",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_lines(lines: list[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def image_only_features(images_root: Path, volume_ids: list[str], dim: int = 32) -> tuple[np.ndarray, str]:
    """Non-label image descriptor: fixed random projection of coarse intensity stats.

    Never reads labels. Checkpoint-free and segmentation-free by construction.
    """
    # Deterministic projection matrix from fixed seed (independent of mask seeds).
    rng = np.random.default_rng(20260914)
    stats = []
    for vid in volume_ids:
        # Prefer nii.gz via nibabel if available; else fail closed.
        path = None
        for ext in (".nii.gz", ".nii"):
            cand = images_root / f"{vid}{ext}"
            if cand.exists():
                path = cand
                break
        if path is None:
            raise FileNotFoundError(f"missing image for {vid} under {images_root}")
        try:
            import nibabel as nib
        except ImportError as e:
            raise RuntimeError("nibabel required for conditional features") from e
        data = np.asanyarray(nib.load(str(path)).dataobj, dtype=np.float32)
        # Coarse, label-free summary: quantiles + spatial moments.
        flat = data.reshape(-1)
        qs = np.quantile(flat, [0.05, 0.25, 0.5, 0.75, 0.95]).astype(np.float64)
        mean = float(flat.mean())
        std = float(flat.std() + 1e-6)
        zz, yy, xx = np.meshgrid(
            np.linspace(-1, 1, data.shape[0], dtype=np.float32),
            np.linspace(-1, 1, data.shape[1], dtype=np.float32),
            np.linspace(-1, 1, data.shape[2], dtype=np.float32),
            indexing="ij",
        )
        w = flat / (np.abs(flat).sum() + 1e-6)
        # subsample for speed
        idx = np.linspace(0, flat.size - 1, num=min(flat.size, 200000), dtype=np.int64)
        stats.append(
            np.concatenate(
                [
                    qs,
                    [mean, std, float(data.min()), float(data.max())],
                    [
                        float((zz.reshape(-1)[idx] * flat[idx]).sum()),
                        float((yy.reshape(-1)[idx] * flat[idx]).sum()),
                        float((xx.reshape(-1)[idx] * flat[idx]).sum()),
                    ],
                ]
            )
        )
    raw = np.stack(stats, axis=0)
    raw = (raw - raw.mean(axis=0)) / (raw.std(axis=0) + 1e-6)
    proj = rng.normal(0.0, 1.0 / np.sqrt(raw.shape[1]), size=(raw.shape[1], dim))
    feats = np.tanh(raw @ proj)
    method = (
        "image_only_quantile_moment_randomproj_v1; "
        "no_labels; no_segmentation_encoder; seed=20260914"
    )
    return feats.astype(np.float64), method


def default_volume_ids(n: int = 100) -> list[str]:
    return [f"word_{i:04d}" for i in range(1, n + 1)]


def discover_volume_ids(images_root: Path) -> list[str]:
    ids = []
    for p in sorted(images_root.glob("word_*.nii*")):
        name = p.name
        if name.endswith(".nii.gz"):
            vid = name[: -len(".nii.gz")]
        else:
            vid = name[: -len(".nii")]
        if vid not in ids:
            ids.append(vid)
    if len(ids) != 100:
        raise RuntimeError(f"expected 100 WORD train volumes, found {len(ids)}")
    return ids


def qc_row(pattern: str, seed: int, out) -> dict:
    mask = out.mask
    cov = class_coverage(mask)
    wstats = summarize_weights(out.inclusion_prob)
    return {
        "pattern": pattern,
        "mask_seed": seed,
        "edges": int(mask.sum()),
        "row_min": int(mask.sum(axis=1).min()),
        "row_max": int(mask.sum(axis=1).max()),
        "pi_min": float(out.inclusion_prob.min()),
        "pi_max": float(out.inclusion_prob.max()),
        "weight_max": wstats["weight_max"],
        "weight_p95": wstats["weight_p95"],
        "ess": wstats["ess"],
        "gini": coverage_gini(mask),
        "entropy": coverage_entropy(mask),
        "coverage_min": float(cov.min()),
        "coverage_max": float(cov.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--images-root", default=None, help="WORD imagesTr for conditional features")
    ap.add_argument("--phase0-commit", default="UNKNOWN")
    ap.add_argument("--phase0-tag", default="phase0-v1.1")
    ap.add_argument("--n-mc-draws", type=int, default=8000)
    ap.add_argument("--positivity-floor", type=float, default=0.05)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.images_root:
        images_root = Path(args.images_root)
        volume_ids = discover_volume_ids(images_root)
        feats, feat_method = image_only_features(images_root, volume_ids)
        feat_sha = hashlib.sha256(feats.tobytes()).hexdigest()
    else:
        volume_ids = default_volume_ids(100)
        # Synthetic image-only stand-in ONLY when WORD images are unavailable.
        # Documented as non-clinical; conditional packs generated this way must
        # be regenerated on a host with WORD before E1 formal training.
        rng = np.random.default_rng(20260914)
        feats = rng.normal(size=(100, 32))
        feat_method = "SYNTHETIC_image_only_stub; regenerate_with_WORD_before_E1"
        feat_sha = hashlib.sha256(feats.tobytes()).hexdigest()

    site_ids = np.arange(len(volume_ids)) % 3
    class_order = [f"organ_{i+1:02d}" for i in range(16)]
    volume_order_path = out_root / "volume_order.txt"
    volume_order_path.write_text("\n".join(volume_ids) + "\n", encoding="utf-8")
    class_order_path = out_root / "class_order.json"
    class_order_path.write_text(json.dumps(class_order, indent=2), encoding="utf-8")

    packs = {}
    qc = []
    for e1_name, mode in PATTERN_MAP.items():
        for seed in (0, 1, 2):
            sampler = MissingnessSampler(
                num_classes=16,
                labels_per_volume=2,
                mode=mode,  # type: ignore[arg-type]
                severity="moderate",
                seed=seed,
                positivity_floor=args.positivity_floor,
                n_mc_draws=args.n_mc_draws,
            )
            out = sampler.generate(
                volume_ids,
                features=feats if mode == "conditional" else None,
                site_ids=site_ids if mode == "clustered" else None,
            )
            assert out.mask.shape == (100, 16)
            assert np.all(out.mask.sum(axis=1) == 2)
            fname = f"{e1_name}_moderate_s{seed}.npz"
            path = out_root / fname
            np.savez_compressed(
                path,
                mask=out.mask.astype(np.uint8),
                selection_score=out.selection_score.astype(np.float64),
                inclusion_prob=out.inclusion_prob.astype(np.float64),
            )
            packs[fname] = {
                "sha256": sha256_file(path),
                "mode": mode,
                "e1_pattern": e1_name,
                "mask_seed": seed,
            }
            qc.append(qc_row(e1_name, seed, out))

    # positivity contract
    bad = [r for r in qc if r["pi_min"] < args.positivity_floor - 1e-3]
    if bad:
        raise RuntimeError(f"positivity violated: {bad}")

    manifest = {
        "protocol": "WORD_E1_v1",
        "dataset": "WORD",
        "num_volumes": len(volume_ids),
        "num_classes": 16,
        "labels_per_volume": 2,
        "total_edges": len(volume_ids) * 2,
        "patterns": list(PATTERN_MAP.keys()),
        "severity": "moderate",
        "mask_seeds": [0, 1, 2],
        "optimization_seed_fixed": 42,
        "mnar_included": False,
        "severe_included": False,
        "phase0_git_commit": args.phase0_commit,
        "phase0_tag": args.phase0_tag,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "volume_order_sha256": sha256_lines(volume_ids),
        "class_order_sha256": hashlib.sha256(json.dumps(class_order).encode()).hexdigest(),
        "positivity_threshold": args.positivity_floor,
        "mc_pi_config": {"num_draws": args.n_mc_draws, "note": "Gumbel-topk MC"},
        "conditional_feature_method": feat_method,
        "conditional_feature_sha256": feat_sha,
        "conditional_requires_word_images": args.images_root is not None,
        "packs": packs,
        "paired_analysis_unit": "same mask_seed across patterns",
    }
    man_path = out_root / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    qc_path = out_root / "qc_report.json"
    qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    # markdown QC table
    md = [
        "# E1-v1 mask pack QC",
        "",
        f"Generated: {manifest['generated_utc']}",
        f"Phase0: `{args.phase0_commit}` / `{args.phase0_tag}`",
        f"Conditional features: `{feat_method}`",
        "",
        "| Pattern | Seed | edges | row min/max | min π | P95 w | ESS | Gini | entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in qc:
        md.append(
            f"| {r['pattern']} | s{r['mask_seed']} | {r['edges']} | "
            f"{r['row_min']}/{r['row_max']} | {r['pi_min']:.4f} | "
            f"{r['weight_p95']:.2f} | {r['ess']:.0f} | {r['gini']:.3f} | {r['entropy']:.3f} |"
        )
    (out_root / "QC.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "out_root": str(out_root), "packs": len(packs)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
