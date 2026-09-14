# WORD E1-v1 mask pack freeze note

Date: 2026-09-14  
Protocol: `WORD_E1_v1`  
Phase0 commit: `ea4921f41a70c559fe3808872d4f497226ee11c7`  
Phase0 tag: `phase0-v1.1` (tag remains on first commit `f99bfc7`; pack generated from `ea4921f`)

## Locked design

| Field | Value |
|---|---|
| Dataset | WORD train |
| N / C / k | 100 / 16 / 2 |
| Total edges | 200 |
| Patterns | random, longtail, sitelike, conditional |
| Severity | moderate only |
| Mask seeds | 0,1,2 (paired across patterns) |
| Optimization seed | fixed 42 (not in mask pack) |
| MNAR / severe | excluded |
| Positivity | π_min ≥ 0.05 (observed overall min ≈ 0.076) |
| Conditional features | image-only quantile/moment random projection; no labels |

## Assets

- Generation host: `server-40902`
- WORD images: `/data/hyc/PLS4MIS/code/datasets/WORD/imagesTr`
- Pack dir: `annotation_masks/WORD/E1_v1/`
- Each `.npz`: `mask`, `selection_score`, `inclusion_prob`
- `manifest.json` records pack SHA256s, volume/class order hashes, feature method
- `contract_verify.json` from `tools/verify_e1_v1_pack.py`

## Copies

| Host | Path |
|---|---|
| Local | `E:\VScodeProject\SPARQ-Seg\annotation_masks\WORD\E1_v1` |
| 40902 | `/data/hyc/SPARQ-Seg/annotation_masks/WORD/E1_v1` |
| School | `/public/home/heyecheng/SPARQ-Seg/annotation_masks/WORD/E1_v1` |

## Immutability

This pack is the E1 experimental asset. Do not overwrite after E1 training starts.
Any sampler change → generate `E1_v2`, never mutate `E1_v1`.

## Residual risk

Moderate patterns keep substantial MCAR overlap for positivity. LongTail shows
the clearest class-coverage skew; SiteLike/Conditional are weaker but still
non-uniform by design. E1 phenomenon gate may still fail — that is a valid
scientific outcome, not a reason to regenerate packs post hoc.

## Next (user-locked sequence)

1. 40902 GPU1 smoke: Random-s0 + Conditional-s0 dataloader→crop→HADFL leak check
2. Smoke pass → 12 E1 long trainings on school A800
3. Do not start propensity before E1 gate
