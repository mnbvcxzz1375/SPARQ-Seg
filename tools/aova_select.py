#!/usr/bin/env python3
"""AOVA Phase A: build an augmented mask pack from a frozen base pack.

Applies one acquisition strategy to a base annotation mask (e.g. E1_v1
random_moderate_s0 at 2/16) adding `budget` new (v,c) annotations, and emits a
pack directory compatible with the existing fail-closed training pipeline:
<out>/augmented.npz (mask, selection_score, inclusion_prob, pattern_meta) +
manifest.json (sha256 per pack) + volume_order.txt copied from the base.

Never reads hidden full labels: strategies see only the base mask and, for
entropy strategies, a precomputed teacher-entropy file over images.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparq.active.strategies import (  # noqa: E402
    select_random,
    select_class_balanced,
    select_top_scores,
    score_entropy_x_coverage,
)

STRATEGIES = ("random", "class_balanced", "entropy", "entropy_coverage")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-pack-root", required=True)
    ap.add_argument("--base-arm", required=True, help="e.g. random_moderate_s0")
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--budget", type=int, required=True, help="new (v,c) pairs")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--entropy-file", default="",
                    help="npz keys 'entropy' [N,C-1] + 'volume_ids'; required for "
                         "entropy strategies")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting an existing asset (discouraged)")
    args = ap.parse_args()

    base_dir = Path(args.base_pack_root)
    base_npz = base_dir / f"{args.base_arm}.npz"
    base_man = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    exp = base_man["packs"][base_npz.name]["sha256"]
    got = sha256_file(base_npz)
    if got != exp:
        raise RuntimeError(f"base pack sha mismatch {got} != {exp}")

    with np.load(base_npz) as z:
        base_mask = z["mask"].astype(np.uint8)
    n, c = base_mask.shape
    vids_path = base_dir / "volume_order.txt"
    if not vids_path.exists():
        raise RuntimeError("base pack missing volume_order.txt")
    vids = [ln.strip() for ln in
            vids_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(vids) != n:
        raise RuntimeError(f"volume_order rows {len(vids)} != mask rows {n}")

    entropy = None
    entropy_sha = ""
    if args.strategy in ("entropy", "entropy_coverage"):
        if not args.entropy_file:
            raise RuntimeError(f"--entropy-file required for {args.strategy}")
        entropy_sha = sha256_file(Path(args.entropy_file))
        with np.load(args.entropy_file) as zf:
            entropy = zf["entropy"].astype(np.float64)
            if "volume_ids" not in zf.files:
                raise RuntimeError("entropy file missing volume_ids; refuse to "
                                   "assume row order (audit F8)")
            ent_ids = [str(x) for x in zf["volume_ids"]]
        # identity + order gate: same set required; permutation reindexed;
        # anything else rejected.
        if len(set(ent_ids)) != len(ent_ids):
            raise RuntimeError("duplicate volume_ids in entropy file")
        if set(ent_ids) != set(vids):
            raise RuntimeError(
                f"entropy volume id set mismatch: only_base="
                f"{sorted(set(vids) - set(ent_ids))[:3]} "
                f"only_ent={sorted(set(ent_ids) - set(vids))[:3]}")
        if ent_ids != vids:
            perm = np.array([ent_ids.index(v) for v in vids])
            entropy = entropy[perm]
            print("entropy rows reindexed to base volume_order", flush=True)
        if entropy.shape[0] != len(ent_ids):
            raise RuntimeError("entropy row count != volume_ids count")
        if entropy.shape[1] != c:
            raise RuntimeError(f"entropy shape {entropy.shape} != mask {(n, c)}")
        entropy = np.where(base_mask == 1, -np.inf, entropy)  # mask annotated cells

    rng = np.random.default_rng(args.rng_seed)
    if args.strategy == "random":
        add = select_random(base_mask, args.budget, rng)
        score = np.zeros((n, c), dtype=np.float32)
    elif args.strategy == "class_balanced":
        add = select_class_balanced(base_mask, args.budget, rng)
        score = np.zeros((n, c), dtype=np.float32)
    elif args.strategy == "entropy":
        add = select_top_scores(base_mask, entropy, args.budget)
        score = entropy.astype(np.float32)
    else:  # entropy_coverage
        prod = score_entropy_x_coverage(np.where(np.isneginf(entropy), 0.0, entropy),
                                        base_mask)
        prod = np.where(base_mask == 1, -np.inf, prod)
        add = select_top_scores(base_mask, prod, args.budget)
        score = prod.astype(np.float32)

    if add[base_mask == 1].any():
        raise RuntimeError("acquisition selected already-annotated cells (bug)")

    mask = np.where(add, np.uint8(1), base_mask).astype(np.uint8)
    new_budget = int(mask.sum(axis=1).min())
    if mask.sum(axis=1).min() != mask.sum(axis=1).max():
        # Phase A keeps per-volume budgets uneven only if base was even;
        # document actual per-volume counts.
        new_budget = -1

    meta = {
        "design": "acquisition",
        "strategy": args.strategy,
        "rng_seed": args.rng_seed,
        "requested_budget": args.budget,
        "acquired": int(add.sum()),
        "base_arm": args.base_arm,
        "base_pack_sha256": got,
        "entropy_file_sha256": entropy_sha or None,
        "volume_order_sha256": sha256_file(vids_path),
        "per_volume_counts_minmax": [int(mask.sum(axis=1).min()),
                                     int(mask.sum(axis=1).max())],
        "inclusion_prob_semantics": "realized per-class coverage (uniform "
                                    "approximation; not used for IPW here)",
    }
    inc = np.tile(coverage_vec(mask), (n, 1)).astype(np.float32)

    out = Path(args.out_root)
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.strategy}_b{args.budget}_r{args.rng_seed}.npz"
    if (out / name).exists() and not args.force:
        raise RuntimeError(
            f"asset exists: {out / name}; AOVA assets are immutable - "
            f"bump budget/seed or pass --force explicitly (audit F9)")
    np.savez_compressed(
        out / name,
        mask=mask,
        selection_score=score,
        inclusion_prob=inc,
        pattern_meta=np.array(json.dumps(meta)),
    )
    man_path = out / "manifest.json"
    entry = {
        name: {
            "sha256": sha256_file(out / name),
            "mask_seed": args.rng_seed,
            "pattern": args.strategy,
            "severity": "moderate",
            "budget": new_budget if new_budget >= 0 else "variable",
            "meta": meta,
        }
    }
    if man_path.exists():
        # merge so multiple strategies can share one pack root
        manifest = json.loads(man_path.read_text(encoding="utf-8"))
        manifest.setdefault("packs", {}).update(entry)
        manifest["design"] = "acquisition"
    else:
        manifest = {
            "design": "acquisition",
            "base": {"pack_root": str(base_dir), "arm": args.base_arm, "sha256": got},
            "packs": entry,
        }
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "volume_order.txt").write_bytes((base_dir / "volume_order.txt").read_bytes())
    print(f"WROTE {out / name} acquired={meta['acquired']} "
          f"counts={meta['per_volume_counts_minmax']}")
    return 0


def coverage_vec(mask: np.ndarray) -> np.ndarray:
    return mask.mean(axis=0, dtype=np.float64)


if __name__ == "__main__":
    raise SystemExit(main())
