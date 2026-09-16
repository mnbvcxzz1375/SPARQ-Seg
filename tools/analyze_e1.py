#!/usr/bin/env python3
"""E1 phenomenon-gate paired analysis.

Reads runs/E1_v1/<arm>/val_dice.json for the 12 arms, pairs every
structured-pattern arm with the same-mask-seed random arm.

Primary metric (pre-specified per PL-Seg/WORD official difficulty groups,
decided 2026-09-16 to avoid post-hoc organ selection):
    DSC_difficult8  = mean Dice over Gallbladder, Esophagus, Pancreas,
                      Duodenum, Colon, Intestine, Adrenal, Rectum
Secondary: DSC_easy8, mean_fg16 (global mean), per-organ deltas.
The random_s0-worst-5 "hard-5" view is kept ONLY as an explicitly labeled
exploratory, post-hoc breakdown.

Gate (DESIGN_LOCKS, applied to DSC_difficult8 paired mean):
  GO    >= 2pp drop, consistent sign across seeds
  NO-GO < 1pp drop
  GRAY  in between / inconsistent

Writes runs/E1_v1/E1_ANALYSIS.json and E1_ANALYSIS.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PATTERNS = ["random", "longtail", "sitelike", "conditional"]
SEEDS = [0, 1, 2]

# ORGANS index -> name mapping matches tools/eval_e1_val.py
ORGANS = [
    "bg", "Liver", "Spleen", "L.Kidney", "R.Kidney", "Stomach", "Gallbladder",
    "Esophagus", "Pancreas", "Duodenum", "Colon", "Intestine", "Adrenal",
    "Rectum", "Bladder", "L.Hip", "R.Hip",
]
# WORD / PL-Seg official difficulty groups (pre-specified).
DIFFICULT_IDX = [6, 7, 8, 9, 10, 11, 12, 13]  # Gallbladder..Rectum
EASY_IDX = [1, 2, 3, 4, 5, 14, 15, 16]


def arm_name(pattern: str, seed: int) -> str:
    return f"{pattern}_moderate_s{seed}"


def group_mean(per_class, idx):
    return sum(per_class[i] for i in idx) / len(idx)


def paired_deltas(data, arms_by_seed, getter):
    deltas = []
    for s in SEEDS:
        a, r = arms_by_seed.get(s), None
        ra = arm_name("random", s)
        if a in data and ra in data:
            deltas.append(getter(data[a]) - getter(data[ra]))
    return deltas


def verdict_for(deltas, go_pp, nogo_pp):
    if not deltas:
        return "NO DATA", None
    mean = sum(deltas) / len(deltas)
    std = (sum((d - mean) ** 2 for d in deltas) / len(deltas)) ** 0.5 if len(deltas) > 1 else 0.0
    drop = -mean * 100  # positive = structured worse
    signs = [1 if d < 0 else 0 for d in deltas]
    consistent = all(s == signs[0] for s in signs)
    if drop >= go_pp and consistent:
        v = f"GO: -{drop:.2f}pp drop >= {go_pp}pp, sign-consistent"
    elif abs(drop) < nogo_pp:
        v = f"NO-GO: |drop| {abs(drop):.2f}pp < {nogo_pp}pp"
    elif drop <= -1.0 and consistent:
        v = (f"NO PHENOMENON: structured BETTER by {-drop:.2f}pp, "
             f"sign-consistent (std {std*100:.2f}pp)")
    elif not consistent:
        v = (f"UNRELIABLE: mean drop {drop:+.2f}pp but signs flip across seeds "
             f"(std {std*100:.2f}pp)")
    else:
        v = f"GRAY: drop {drop:+.2f}pp in {-go_pp:.1f}..-{nogo_pp:.1f}pp band, consistent"
    return v, {"mean_drop_pp": round(drop, 3), "seed_std_pp": round(std * 100, 3),
               "deltas_pp": [round(d * 100, 3) for d in deltas], "n_pairs": len(deltas),
               "sign_consistent": consistent}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", required=True, help="runs/E1_v1")
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
    if not data:
        print("no eval results yet")
        return 1

    g_diff = lambda d: group_mean(d["per_class_mean_dice"], DIFFICULT_IDX)
    g_easy = lambda d: group_mean(d["per_class_mean_dice"], EASY_IDX)
    g_fg = lambda d: float(d["mean_fg16"])

    lines = [
        "# E1 phenomenon-gate paired analysis (v2, pre-specified groups)",
        "",
        f"metric primary: DSC_difficult8 (WORD official difficult group, "
        f"pre-specified 2026-09-16); cases={data[next(iter(data))]['n_cases']} "
        f"locked imagesVal; missing arms: {missing or 'none'}",
        "",
        "| arm | difficult8 | easy8 | fg16 |",
        "|---|---|---|---|",
    ]
    for p in PATTERNS:
        for s in SEEDS:
            a = arm_name(p, s)
            if a in data:
                d = data[a]
                lines.append(f"| {a} | {g_diff(d):.4f} | {g_easy(d):.4f} | "
                             f"{float(d['mean_fg16']):.4f} |")
    lines.append("")

    report = {"metric_primary": "difficult8", "missing_arms": missing,
              "gate": {"go_pp": args.go_pp, "nogo_pp": args.nogo_pp},
              "arms": {a: {"difficult8": g_diff(d), "easy8": g_easy(d),
                           "mean_fg16": float(d["mean_fg16"])}
                       for a, d in data.items()},
              "patterns": {}}

    for p in PATTERNS[1:]:
        by_seed = {s: arm_name(p, s) for s in SEEDS}
        entry = {}
        lines.append(f"## {p} vs random (paired by mask seed)")
        for key, getter in (("difficult8", g_diff), ("easy8", g_easy),
                            ("mean_fg16", g_fg)):
            v, st = verdict_for(paired_deltas(data, by_seed, getter),
                                args.go_pp, args.nogo_pp)
            if key == "difficult8":
                entry["primary"] = {"verdict": v, **(st or {})}
                lines.append(f"- **DSC_difficult8: {v}** {st or {}}")
            else:
                entry[key] = {"verdict": v, **(st or {})}
                lines.append(f"- {key}: {v} {st or {}}")

        # per-organ deltas (difficult group first, pre-specified)
        pd = [data[by_seed[s]]["per_class_mean_dice"] for s in SEEDS
              if by_seed[s] in data]
        rd = [data[arm_name("random", s)]["per_class_mean_dice"] for s in SEEDS
              if arm_name("random", s) in data]
        if pd and rd and len(pd) == len(rd):
            pm = [sum(c[i] for c in pd) / len(pd) for i in range(len(ORGANS))]
            rm = [sum(c[i] for c in rd) / len(rd) for i in range(len(ORGANS))]
            per_organ = {ORGANS[i]: round((pm[i] - rm[i]) * 100, 3)
                         for i in DIFFICULT_IDX + EASY_IDX}
            entry["per_organ_delta_pp"] = {k: per_organ[k]
                                           for k in (ORGANS[i] for i in DIFFICULT_IDX)}
            entry["per_organ_delta_pp_easy"] = {k: per_organ[k]
                                                for k in (ORGANS[i] for i in EASY_IDX)}
            lines.append(f"- per-organ delta (difficult8, pp): {entry['per_organ_delta_pp']}")

            # post-hoc explorer: worst single-seed arm breakdown
            worst = min(SEEDS, key=lambda s: (g_diff(data[by_seed[s]]) -
                          g_diff(data[arm_name('random', s)])) if s in by_seed
                          and by_seed[s] in data else 0)
            ws = by_seed[worst]
            if ws in data:
                delta1 = {ORGANS[i]: round((data[ws]["per_class_mean_dice"][i] -
                              data[arm_name("random", worst)]["per_class_mean_dice"][i])
                             * 100, 3) for i in DIFFICULT_IDX}
                lines.append(f"- largest drop on seed{worst} ({ws}), difficult8 per-organ (pp): "
                             f"{delta1}")
        report["patterns"][p] = entry
        lines.append("")

    lines.append("## Exploratory (post-hoc, NOT for gate claims)")
    ref = "random_moderate_s0"
    if ref in data:
        pc = data[ref]["per_class_mean_dice"]
        hard5 = sorted(DIFFICULT_IDX + EASY_IDX, key=lambda c: pc[c])[:5]
        for p in PATTERNS[1:]:
            pd = [data[arm_name(p, s)]["per_class_mean_dice"] for s in SEEDS
                  if arm_name(p, s) in data]
            rd = [data[arm_name("random", s)]["per_class_mean_dice"] for s in SEEDS
                  if arm_name("random", s) in data]
            if len(pd) == len(rd):
                pm = [sum(c[i] for c in pd) / len(pd) for i in range(len(ORGANS))]
                rm = [sum(c[i] for c in rd) / len(rd) for i in range(len(ORGANS))]
                h5 = {ORGANS[i]: round((pm[i] - rm[i]) * 100, 3) for i in hard5}
                lines.append(f"- {p} random_s0-worst5 {h5}")
    report["hard5_exploratory_posthoc"] = True

    (root / "E1_ANALYSIS.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (root / "E1_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
