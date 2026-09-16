# Audit evidence snapshot — 2026-09-16

Collected for the AUDIT HOLD calibration step (i) of `docs/E1_V1_VERDICT.md`.
Source: external source-code audit REVIEW.md @ snapshot `0a6d573` / E1 training
commit `8a4169d`.

## Contents

- `school/` — 9 E1 arms (random/longtail/sitelike × seeds 0–2) trained on
  school gpu_4090 (RTX 4090D): `run_manifest.json`, `train.log`,
  `val_dice.json` per arm (flattened `<arm>__<file>`), plus the frozen pack
  manifest `E1_v1__pack_manifest.json`. Conditional arms' `val_dice.json`
  copies here were aggregated from the local hosts for the gate analysis.
- `40901/`, `3080/` — conditional arms trained on local hosts (audit F4:
  different environment from the school nine arms). `conditional_moderate_s0`
  `run_manifest.json` records `resumed_from_epoch: 350` (non-equivalent
  resume after a 20:36 reboot; HADFL history / sampler order / plateau
  counter not restored).
- `VENDOR.sha256` — SHA-256 of the external PL-Seg tree
  (`/public/home/heyecheng/word_paper_registry_20260904/code`) that the SPARQ
  commit does not bind (audit F5). md5 cross-check: unet3d_ProgCon d27330d1…,
  sing_class_loss 3ae8ab05…, val3D 2a47683e…, transforms 3608c510….
- `E1_ANALYSIS.md` / `E1_ANALYSIS.json` — gate analysis (regenerated with the
  corrected verdict logic: full-seed requirement + seed-std gate).

## Confirmed facts relevant to interpretation

- F3: vendor `test_single_case_fourpre` slices D by `patch_size[2]` (=96):
  eval window is D96×H128×W128; E1 training crops D128×H128×W96. WORD spacing
  xy≈0.98mm, z=2.5mm → physical FOV differs (z +33%, W −25%).
- F8: current AOVA packs verified row-aligned (entropy volume_ids == base
  volume_order, 100 unique); selector now hard-gates on ids and reindexes or
  rejects permutations.
