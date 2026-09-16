#!/usr/bin/env python3
"""E1 phenomenon-gate paired analysis.

Reads runs/E1_v1/<arm>/val_dice.json for the 12 arms, pairs every
structured-pattern arm with the same-mask-seed random arm, and reports:
  - paired mean-fg16 Dice delta per pattern (mean over seeds, per-seed deltas)
  - mask-seed variance of the delta
  - per-organ deltas with focus on hard/tail organs
  - gate verdict per DESIGN_LOCKS: GO if paired dDSC >= 2pp and hard organs
    drop more; NO-GO (stop propensity) if paired dDSC < 1.0-1.5pp.

Writes runs/E1_v1/E1_ANALYSIS.json and E1_ANALYSIS.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PATTERNS = ["random", "longtail", "sitelike", "conditional"]
SEEDS = [0, 1, 2]


def arm_name(pattern: str, seed: int) -> str:
    return f"{pattern}_moderate_s{seed}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True, help="runs/E1_v1")
    ap.add_argument("--metric", default="mean_fg16",
                    choices=["mean_fg16", "mean_all17"])
    ap.add_argument("--go-pp", type=float, default=2.0)
    ap.add_argument("--nogo-pp", type=float, default=1.0)
    args = ap.parse_args()

    root = Path(args.runs_root)
    data: dict[str, dict] = {}
    missing = []
    for p in PATTERNS:
        for s in SEEDS:
            a = arm_name(p, s)
            f = root / a / "val_dice.json"
            if f.exists():
                data[a] = json.loads(f.read_text(encoding="utf-8"))
            else:
                missing.append(a)

    if "random_moderate_s0" not in data or not data:
        print("no eval results yet")
        return 1

    organs = data[next(iter(data))]["organs"]
    metric = args.metric

    def val(a: str) -> float:
        return float(data[a][metric])

    lines = [
        "# E1 phenomenon-gate paired analysis",
        "",
        f"metric: `{metric}` over {data[next(iter(data))]['n_cases']} locked "
        f"imagesVal cases; missing arms: {missing or 'none'}",
        "",
        "| arm | dice |",
        "|---|---|",
    ]
    for p in PATTERNS:
        for s in SEEDS:
            a = arm_name(p, s)
            if a in data:
                lines.append(f"| {a} | {val(a):.4f} |")
    lines.append("")

    report = {
        "metric": metric,
        "missing_arms": missing,
        "arms": {a: {"dice": val(a)} for a in data},
        "patterns": {},
    }

    hard_ref = None
    if len(data) >= 3:
        ref = "random_moderate_s0"
        pc = data[ref]["per_class_mean_dice"]
        hard_ref = sorted(range(1, len(organs)), key=lambda c: pc[c])[:5]

    for p in PATTERNS[1:]:
        deltas = []
        for s in SEEDS:
            a = arm_name(p, s)
            r = arm_name("random", s)
            if a in data and r in data:
                deltas.append(val(a) - val(r))
        if not deltas:
            continue
        mean_d = sum(deltas) / len(deltas)
        var = (
            sum((d - mean_d) ** 2 for d in deltas) / len(deltas)
            if len(deltas) > 1 else 0.0
        )
        entry = {
            "paired_deltas_pp": [round(d * 100, 3) for d in deltas],
            "mean_delta_pp": round(mean_d * 100, 3),
            "seed_std_pp": round(var ** 0.5 * 100, 3),
            "n_pairs": len(deltas),
        }
        if hard_ref and all(arm_name(p, s) in data for s in SEEDS if arm_name(p, s) in data):
            pd = [
                data[arm_name(p, s)]["per_class_mean_dice"]
                for s in SEEDS if arm_name(p, s) in data
            ]
            rd = [
                data[arm_name("random", s)]["per_class_mean_dice"]
                for s in SEEDS if arm_name("random", s) in data
            ]
            if pd and rd and len(pd) == len(rd):
                pm = [sum(c[i] for c in pd) / len(pd) for i in range(len(organs))]
                rm = [sum(c[i] for c in rd) / len(rd) for i in range(len(organs))]
                per_organ = {
                    organs[i]: round((pm[i] - rm[i]) * 100, 3)
                    for i in range(1, len(organs))
                }
                entry["per_organ_delta_pp"] = per_organ
                entry["hard5_delta_pp"] = {
                    organs[i]: per_organ[organs[i]] for i in hard_ref
                }
        report["patterns"][p] = entry

        lines.append(f"## {p} vs random (paired by mask seed)")
        lines.append(f"- paired deltas (pp): {entry['paired_deltas_pp']}")
        lines.append(f"- mean delta: **{entry['mean_delta_pp']:+.2f} pp** "
                     f"(seed std {entry['seed_std_pp']:.2f} pp, n={entry['n_pairs']})")
        if "hard5_delta_pp" in entry:
            lines.append(f"- hard-5 organ deltas: {entry['hard5_delta_pp']}")
        if entry["mean_delta_pp"] <= -args.go_pp:
            verdict = (f"PHENOMENON ({-entry['mean_delta_pp']:.2f}pp drop >= "
                       f"{args.go_pp}pp gate)")
            if entry["seed_std_pp"] > args.go_pp:
                verdict += " BUT seed std exceeds gate - unreliable, extend seeds"
        elif -entry["mean_delta_pp"] < args.nogo_pp:
            verdict = "NO-GO: |delta| < 1pp, no usable phenomenon"
        else:
            verdict = (f"GRAY ZONE: {-entry['mean_delta_pp']:.2f}pp drop in "
                       f"{args.nogo_pp}-{args.go_pp}pp band, direction "
                       f"{'consistent' if entry['seed_std_pp'] < 0.5 else 'mixed'}; "
                       f"extend seeds before deciding")
        lines.append(f"- verdict: {verdict}")
        lines.append("")
        entry["verdict"] = verdict

    (root / "E1_ANALYSIS.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (root / "E1_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
