"""Regression tests for the analysis gate (verdict_for) — kept separate from
the AOVA strategy tests so '12/12' never implies the gate is covered.

Known limitation pinned by test_single_seed_leverage (audit F6): the 3-seed
engineering gate (mean>=2pp, all seeds, std<=2pp) still passes
[-0.70,-0.70,-4.60] where two seeds alone give 0.70pp. Reports therefore
also expose median and leave-one-out means; do NOT describe the gate as a
statistical reliability proof.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "analyze_e1", Path(__file__).resolve().parents[1] / "tools" / "analyze_e1.py")
ae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ae)


class TestVerdict(unittest.TestCase):
    def setUp(self):
        self.f = ae.verdict_for

    def test_partial_single_pair(self):
        v, st = self.f([-0.03], 2.0, 1.0, require_full_seeds=3)
        self.assertTrue(v.startswith("PARTIAL"), v)

    def test_single_seed_driven_unreliable(self):
        v, _ = self.f([-0.0001, -0.0001, -0.06], 2.0, 1.0)
        self.assertTrue(v.startswith("UNRELIABLE"), v)

    def test_single_seed_leverage_pinned_limitation(self):
        # mean exactly 2.0pp with std 1.84pp <= gate -> still GO by design;
        # two of three seeds alone would give only 0.70pp.
        v, st = self.f([-0.007, -0.007, -0.046], 2.0, 1.0)
        self.assertTrue(v.startswith("GO"), v)
        self.assertAlmostEqual(st["mean_drop_pp"], 2.0, places=1)
        self.assertLess(st["seed_std_pp"], 2.0)
        # mitigations exposed:
        self.assertLess(st["median_drop_pp"], 1.0)
        self.assertEqual(len(st["leave_one_out_mean_drop_pp"]), 3)
        self.assertTrue(any(x < 1.0 for x in st["leave_one_out_mean_drop_pp"]))

    def test_gray_zone(self):
        v, _ = self.f([-0.0167, -0.01265, -0.00929], 2.0, 1.0)
        self.assertTrue(v.startswith("GRAY"), v)

    def test_no_go_small(self):
        v, _ = self.f([-0.005, -0.008, -0.003], 2.0, 1.0)
        self.assertTrue(v.startswith("NO-GO"), v)

    def test_structured_better(self):
        v, _ = self.f([0.02, 0.025, 0.03], 2.0, 1.0)
        self.assertTrue(v.startswith("NO PHENOMENON"), v)

    def test_sign_flip_unreliable(self):
        v, _ = self.f([-0.0166, -0.0132, 0.0527], 2.0, 1.0)
        self.assertTrue(v.startswith("UNRELIABLE"), v)

    def test_clear_go(self):
        v, st = self.f([-0.03, -0.028, -0.032], 2.0, 1.0)
        self.assertTrue(v.startswith("GO"), v)
        self.assertTrue(st["sign_consistent"])


if __name__ == "__main__":
    unittest.main()
