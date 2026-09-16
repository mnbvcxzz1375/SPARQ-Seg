# AOVA pilot evidence — 2026-09-16

Phase A 4/16 arms (B=200 global) launched under AUDIT HOLD; kept as pilot.

- `ASSETS.sha256` — SHA-256 of the four augmented packs, the base-derived
  volume order binding, and the teacher entropy file
  (`runs/AOVA_v1/entropy_random_s0.npz`, teacher = E1
  `random_moderate_s0/final_model.pth`). Row alignment verified
  (entropy volume_ids == base volume_order, 100 unique) before submission.
- `WORD/AOVA_v1/manifest.json` — pack manifest (merge-updated, per-pack
  meta incl. base/entropy/order SHAs).
- `AOVA_v1/<arm>/run_manifest.json` — per-arm training manifests (running).

Restrictions inherited from `docs/E1_V1_VERDICT.md` audit table:
F2 recipe drift and F3 patch-coordinate mismatch apply to these arms;
interpretation limited to pilot level; budget expansion frozen pending the
MCAR bridge experiment.
