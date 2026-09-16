"""Calibration module (AUDIT HOLD response, 2026-09-16).

Versioned SEPARATELY from E1-v1: nothing here may modify frozen packs, the
in-flight AOVA pilot, or the vendor tree. Shared train_step lives here and is
extracted from train_e1_arm @8a4169d WITHOUT behavior change; equivalence is
proven by calib/test_step_equivalence.py before any recipe change.

Files:
- legacy_step.py          verbatim copy of the current inline step (oracle)
- train_step.py           shared step function (single source of truth)
- test_step_equivalence.py  step-by-step equality (losses/grads/state/update)
- real_chain_checks.py    true-model counterfactual leak check + runtime
                          module fingerprints
- w3_eval_pair.py         recompute the W3 anchor from its checkpoint on the
                          locked 20 cases (metric-convention alignment)
"""
