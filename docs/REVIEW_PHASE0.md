# Phase-0 code review (v1.1 locks)

Date: 2026-09-14  
Reviewer: mimo  
Scope: `code/utils/*`, `code/dataloader/*`, `tests/test_phase0_contracts.py`

## Verdict

**PASS after fixes.** Local and server-40902 `vllmenv` both report 18/18.

## Findings fixed before freeze

1. **`inclusion_prob` shape**  
   Was `[C]` for random and `[N,C]` for MC modes.  
   Fix: always emit `[N,C]` (analytic uniform broadcast).

2. **Oracle π must not be clipped**  
   Early draft called `enforce_positivity` on the MC estimate, which would
   destroy Oracle-IPW (clipped π ≠ design π).  
   Fix: positivity is enforced in **score construction** (uniform blend);
   generate() fail-closes if estimated min π still falls below floor.
   `enforce_positivity` remains a diagnostics-only helper with a warning.

3. **Fixed-k coupling vs naive score floor**  
   Gumbel-topk with k=2/C=16 drove tail-class π below 0.05 even with
   score floors.  
   Fix: row-normalize + strong uniform blend
   (`α=0.72` moderate / `0.55` severe) so main modes satisfy
   π_min ≥ 0.05 by design; MC verifies.

4. **`apply_annotation_mask` class inference**  
   Inferring C from a single volume's max label is unsafe if that volume
   omits high class ids.  
   Fix: explicit `num_foreground`; PartialLabelView passes `amask.size`.

5. **`get()` method break during edit**  
   Repaired `_apply` / `get` / `train_sample` structure.

## Contract checklist

| Lock | Status |
|---|---|
| mask / selection_score / inclusion_prob / pattern_meta | PASS |
| selection_score ≠ propensity | PASS (test) |
| fixed budget every row | PASS |
| positivity π_min≥0.05 main modes | PASS (score blend + fail-closed) |
| annotation_mask ≠ presence_mask | PASS |
| train payload never includes full_label | PASS |
| allow_full_label_in_train forbidden | PASS |
| seed reproducible | PASS |
| mask-seed changes mask | PASS |
| severity metadata (coverage/gini/entropy/co-annotation) | PASS |

## Residual risks (accepted for Phase 0)

- MC inclusion is statistical (default 4000 draws); for paper Oracle rows
  consider 20k+ draws or analytic designs where available.
- Uniform blend may under-represent “severe” clinical skew; keep `severe`
  as stress appendix, `moderate` as headline (FLARE-calibrated later).
- `partial_dataset` is a visibility contract, not a NIfTI loader; WORD I/O
  comes in the E1 adapter.
- School default `python3` is 3.6 — use
  `odin2026_local_baseline/.venv` (3.11) for SPARQ-Seg.

## Test matrix

| Host | Interpreter | Result |
|---|---|---|
| Local Windows | MIMO_PYTHON 3.12 | 18/18 OK |
| server-40902 | `/home/ubuntu/anaconda3/envs/vllmenv/bin/python` | 18/18 OK |
| school-platform | `odin2026_local_baseline/.venv` (3.11.15) | 18/18 OK |

## Next

- E1 adapter: load WORD imagesTr + labelsTr_2, bind mask pack, train PL-Seg unchanged.
- Do not start propensity training before E1 phenomenon gate.
