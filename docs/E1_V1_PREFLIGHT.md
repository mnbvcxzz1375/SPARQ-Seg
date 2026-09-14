# E1-v1 Preflight Gate

Date: 2026-09-14  
Host: server-40902 GPU1 (`CUDA_VISIBLE_DEVICES=1`)  
Code commit: `65a8e8bb2776e4a6a05c5a7f6fc095a3743e3e16`  
Pack source commit: `ea4921f41a70c559fe3808872d4f497226ee11c7`  
Artifact: `artifacts/e1_v1_preflight.json`

## Verdict

**GO = true** — Random-s0 and Conditional-s0 both pass the full leak-check contract.
No performance/Dice gate. E1-v1 mask pack is not modified.

## Presence leak fix

`presence_mask` is now `presence &= annotation_mask`.
Computing presence from full GT without this intersection would leak hidden organs
even when `full_label` is omitted from the train payload.

## Seven checks (both arms PASS)

| Check | Random-s0 | Conditional-s0 |
|---|---|---|
| Pack identity (SHA256 vs manifest) | PASS | PASS |
| Volume-mask alignment (N=100) | PASS | PASS |
| Partial target purity | PASS | PASS |
| Presence purity (`presence ≤ annotation`) | PASS | PASS |
| Multi-scale purity (s1/s2/s3) | PASS | PASS |
| HADFL state isolation | PASS | PASS |
| Counterfactual invariance | PASS | PASS |
| Train smoke finite (20 steps) | PASS | PASS |

## Counterfactual (hidden vs visible)

| Arm | ΔL hidden | Δ∇ hidden | ΔL visible |
|---|---:|---:|---:|
| random_s0 | 0.0 | 0.0 | 1.314 |
| conditional_s0 | 0.0 | 0.0 | 1.284 |

Hidden-organ GT perturbation leaves loss and gradients identical;
visible-organ perturbation changes loss substantially. Positive control passed.

## Seeds logged separately

- `mask_seed = 0`
- `optimization_seed = 42`
- `dataloader_generator_seed = 42`
- first pass `num_workers = 0`

## Notes

- Smoke uses TinySeg + MiniHADFL (hardness-queue semantics matching PL-Seg HADFLoss)
  for cheap counterfactual gradients, not the full UNet3D_ProgCon paper model.
- Package renamed `code/` → `sparq/` to avoid Python stdlib clash.
- Next allowed action: school A800 E1 long trainings (12 arms). Still no propensity.

## Immutability

This document and `artifacts/e1_v1_preflight.json` are the E1 preflight freeze.
Do not regenerate E1-v1 masks. Do not start propensity before E1 phenomenon gate.
