#!/usr/bin/env python3
"""Collect R1_v1 results: convert each arm's final_eval_stride64.json into
the val_dice.json schema compatible with tools/analyze_e1.py, then print a
summary table (final-model primary, best as sensitivity).

Usage (any host with the runs dir):
  python tools/r1_collect.py --runs-root runs/R1_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs/R1_v1")
    args = ap.parse_args()
    root = Path(args.runs_root)

    rows = []
    for arm_dir in sorted(root.iterdir()):
        if not arm_dir.is_dir():
            continue
        fe = arm_dir / "final_eval_stride64.json"
        if not fe.exists():
            rows.append({"arm": arm_dir.name, "state": "incomplete"})
            continue
        r = json.loads(fe.read_text(encoding="utf-8"))
        # primary checkpoint: final (pre-declared for R1 gate); best = sensitivity
        final, best = r["final_model.pth"], r["best_model.pth"]
        out = {
            "arm": arm_dir.name,
            "state": "done",
            "final_fg16": final["mean_fg16"], "final_all17": final["mean_all17"],
            "best_fg16": best["mean_fg16"], "best_all17": best["mean_all17"],
            "per_class_mean_dice_final": [
                sum(c["per_class"][i] for c in final["per_case"])
                / len(final["per_case"]) for i in range(17)],
            "n_cases": len(final["per_case"]),
        }
        # analyze_e1-compatible val_dice.json (final checkpoint = primary)
        (arm_dir / "val_dice.json").write_text(
            json.dumps({
                "arm": arm_dir.name, "checkpoint": "final_model.pth",
                "n_cases": out["n_cases"], "organs": [
                    "bg", "Liver", "Spleen", "L.Kidney", "R.Kidney", "Stomach",
                    "Gallbladder", "Esophagus", "Pancreas", "Duodenum", "Colon",
                    "Intestine", "Adrenal", "Rectum", "Bladder", "L.Hip", "R.Hip"],
                "per_class_mean_dice": out["per_class_mean_dice_final"],
                "mean_all17": out["final_all17"], "mean_fg16": out["final_fg16"],
                "per_case": final["per_case"],
            }, indent=2), encoding="utf-8")
        rows.append(out)

    print(f"{'arm':34s} {'state':10s} {'final_fg16':>10s} {'best_fg16':>10s}")
    for r in rows:
        if r["state"] == "done":
            print(f"{r['arm']:34s} {'done':10s} {r['final_fg16']:10.4f} "
                  f"{r['best_fg16']:10.4f}")
        else:
            print(f"{r['arm']:34s} {r['state']}")
    print("\nval_dice.json (final primary) written per arm; run "
          "tools/analyze_e1.py --runs-root <R1 root> for the paired gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
