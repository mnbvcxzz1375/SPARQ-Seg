# SPARQ-Seg design locks

Frozen at roadmap v1.1 (2026-09-14). Do not silently relax.

## Scope

| Stage | Name | Change |
|---|---|---|
| I | SM-PLSeg | annotation protocol + HADFL → PA-HADFL only |
| II | AOVA | query unit `(volume, organ)` |

Unchanged in Stage I: UNet3D_ProgCon backbone, COCL, PSD.

## Simulator contract

```python
{
  "mask": [N, C],             # realized 0/1
  "selection_score": [N, C],  # generator score, NOT a probability
  "inclusion_prob": [N, C],   # true π for IPW / oracle
  "pattern_meta": {...},      # coverage, gini, entropy, seed, severity
}
```

- Fixed-size k-of-C sampling couples class choices; inclusion_prob ≠ selection_score.
- Oracle IPW uses inclusion_prob only (analytic design or Monte-Carlo estimate).

## Positivity

Main experiments: `π_min ≥ 0.05` (prefer `0.1`). Report weight max, P95, ESS.
True zero coverage only as explicit MNAR/failure stress.

## Masks

```text
full_label              # simulator / oracle only
annotation_mask [C]     # volume-level human annotation availability
presence_mask   [C]     # patch-level organ presence
partial_label           # training-visible labels only
```

Do not introduce new negative supervision just because volume-level M exists.

## Loss / theory

- Standard IPW unbiasedness is motivation only.
- Deployed PA-HADFL: `(π̄ / max(π̂, ε))^ρ` with ε=0.1, ρ=0.5, × hardness, batch-normalized.
- Write as stabilized bias–variance trade-off, not unbiased full-risk.

## E1 protocol

- 3 **annotation-mask seeds**, optimization seed fixed.
- Paired structured-vs-random comparison.
- Gate uses paired ΔDSC + hard/worst-class gap + mask-seed variance.
- Stop propensity if paired ΔDSC < 1.0–1.5 pp and patterns are similar.

## Stage II

- MVP: entropy × class-coverage sanity check first.
- Final score: Fisher (head only) + distribution repair.
- Repair v1: product kernel `k_x(z_n,z_m)·1[c=d]` — no learned organ embedding yet.
- Dual model: sequential acquisition model + fixed-init evaluation model at key budgets.
- Metric: learning curve, AUC, B_{90%FSL} (not dataset-specific 75% Dice).

## Data boundary

- Corrected `labelsTr_2` for 2/16; never fake partial with full labelsTr.
- Locked imagesVal for selection; official test after locked candidates only.
