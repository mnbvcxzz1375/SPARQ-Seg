# SPARQ-Seg

**S**tructured-missingness-aware **P**artial-label segmentation with
**A**ctive or**g**an–volume **R**epair and **Q**uery.

Implementation workspace for the continuous research route:

**Stage I SM-PLSeg** (debias structured missingness) →
**Stage II AOVA** (actively acquire organ–volume annotations).

Baseline: PL-Seg — Li et al., *Medical Image Analysis* 108 (2026) 103885
(online 21 Nov 2025). Benchmark: WORD partial labels (`labelsTr_2/4/6`).

## Related knowledge

| Item | Location |
|---|---|
| Research roadmap v1.1 | `E:\VScodeProject\JEPA_Pseudorgb\research\plseg-structured-missingness\index.html` |
| Obsidian project | `word-jepa-plseg` vault (`E:\Obsidian\llm-wiki-lab\word-jepa-plseg`) |
| Route wiki | `wiki/SM-PLSeg to AOVA two-stage research route.md` |
| Workstream | `WS-sm-plseg-aova` |
| Phase-0 task | `TASK-20260914-sm-plseg-phase0-simulator` |

## Design locks (do not violate)

1. Do **not** change backbone / COCL / PSD in Stage I. Only HADFL → PA-HADFL.
2. Simulator returns `mask + selection_score + inclusion_prob` — never treat sampling score as propensity.
3. Main experiments keep positivity `π_min ≥ 0.05–0.1`; true zeros only as stress.
4. `annotation_mask` (volume-level) ≠ `presence_mask` (patch-level).
5. Training path must never read hidden organs from `full_label`.
6. Standard IPW is theory motivation only; deployed PA-HADFL is tempered/stabilized.
7. E1 uses **annotation-mask seeds** (fixed optimization seed) and paired stats.

## Phase-0 layout (target)

```text
code/
  dataloader/
    missingness_sampler.py
    partial_dataset.py
  utils/
    annotation_matrix.py
    inclusion_prob.py
  models/Proposed/
    propensity_head.py      # later
    propensity_loss.py      # later
  active/                   # Stage II
annotation_masks/WORD/
configs/
tests/
```

## Status

- [x] Remote repo linked
- [x] Local clone at `E:\VScodeProject\SPARQ-Seg`
- [x] Phase-0 data layer + review
- [x] Phase-0 commits (`f99bfc7` tag `phase0-v1.1`; pack source `ea4921f`)
- [x] WORD E1-v1 mask pack frozen + contract verify PASS
- [ ] 40902 GPU1 dataloader→HADFL leak smoke
- [ ] E1 phenomenon trainings (school A800)
- [ ] Stage I PA-HADFL
- [ ] Stage II AOVA

## Data boundary

- WORD root / labels: use corrected `labelsTr_2` view for 2/16 experiments.
- `full_label` is simulator/oracle only.
- Locked `imagesVal` for selection; official test only after locked candidates.
