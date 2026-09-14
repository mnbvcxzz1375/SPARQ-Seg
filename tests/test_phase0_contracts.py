"""Unit tests for Phase-0 missingness simulator and partial-label contracts."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

# Allow `python tests/test_phase0_contracts.py` from repo root.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from code.dataloader.missingness_sampler import MissingnessSampler, generate_word_masks
from code.dataloader.partial_dataset import (
    PartialLabelView,
    apply_annotation_mask,
    load_mask_pack,
    patch_presence_mask,
    save_mask_pack,
)
from code.utils.annotation_matrix import class_coverage, coverage_gini, pattern_meta
from code.utils.inclusion_prob import (
    monte_carlo_inclusion_prob,
    summarize_weights,
    uniform_inclusion_prob,
)


class TestAnnotationMatrix(unittest.TestCase):
    def test_coverage_and_gini(self):
        mask = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]], dtype=np.uint8)
        cov = class_coverage(mask)
        self.assertTrue(np.allclose(cov, [1.0, 1 / 3, 1 / 3, 1 / 3]))
        g = coverage_gini(mask)
        self.assertGreater(g, 0.0)
        self.assertLessEqual(g, 1.0)

    def test_pattern_meta_fixed_budget(self):
        sampler = MissingnessSampler(16, 2, mode="random", seed=0)
        out = sampler.generate([f"v{i}" for i in range(20)])
        self.assertEqual(out.pattern_meta["rows_with_budget"], 20)
        self.assertEqual(out.pattern_meta["labels_per_volume"], 2)


class TestSamplerContracts(unittest.TestCase):
    def test_fixed_budget_all_modes(self):
        feats = np.random.default_rng(0).normal(size=(30, 8))
        sites = np.arange(30) % 3
        for mode in ("random", "longtail", "clustered", "conditional", "mnar_stress"):
            kwargs = {}
            if mode == "conditional":
                kwargs["features"] = feats
            if mode == "clustered":
                kwargs["site_ids"] = sites
            s = MissingnessSampler(16, 2, mode=mode, seed=1, n_mc_draws=400)
            out = s.generate([f"v{i}" for i in range(30)], **kwargs)
            self.assertEqual(out.mask.shape, (30, 16))
            self.assertTrue(np.all(out.mask.sum(axis=1) == 2), mode)
            self.assertEqual(out.inclusion_prob.shape, (30, 16), mode)
            self.assertEqual(out.selection_score.shape, (30, 16))

    def test_selection_score_is_not_inclusion_prob(self):
        feats = np.random.default_rng(1).normal(size=(40, 8))
        s = MissingnessSampler(16, 2, mode="conditional", seed=2, n_mc_draws=800)
        out = s.generate([f"v{i}" for i in range(40)], features=feats)
        # They must be different objects / generally not equal.
        self.assertFalse(np.allclose(out.selection_score.mean(axis=0), out.inclusion_prob))

    def test_random_inclusion_analytic(self):
        s = MissingnessSampler(16, 2, mode="random", seed=3)
        out = s.generate([f"v{i}" for i in range(10)])
        self.assertEqual(out.inclusion_prob.shape, (10, 16))
        self.assertTrue(np.allclose(out.inclusion_prob, 2 / 16))
        self.assertTrue(np.allclose(uniform_inclusion_prob(16, 2), 2 / 16))

    def test_positivity_floor_main_modes(self):
        feats = np.random.default_rng(2).normal(size=(50, 8))
        sites = np.arange(50) % 3
        for mode, kwargs in (
            ("longtail", {}),
            ("clustered", {"site_ids": sites}),
            ("conditional", {"features": feats}),
        ):
            s = MissingnessSampler(16, 2, mode=mode, seed=4, n_mc_draws=600, positivity_floor=0.05)
            out = s.generate([f"v{i}" for i in range(50)], **kwargs)
            self.assertGreaterEqual(out.inclusion_prob.min(), 0.05 - 1e-9, mode)
            # selection scores also floored
            self.assertGreater(out.selection_score.min(), 0.0, mode)

    def test_seed_reproducibility(self):
        a = MissingnessSampler(16, 2, mode="longtail", seed=7).generate([f"v{i}" for i in range(15)])
        b = MissingnessSampler(16, 2, mode="longtail", seed=7).generate([f"v{i}" for i in range(15)])
        self.assertTrue(np.array_equal(a.mask, b.mask))
        self.assertTrue(np.allclose(a.inclusion_prob, b.inclusion_prob))

    def test_mask_seed_changes_mask(self):
        a = MissingnessSampler(16, 2, mode="random", seed=0).generate([f"v{i}" for i in range(40)])
        b = MissingnessSampler(16, 2, mode="random", seed=1).generate([f"v{i}" for i in range(40)])
        self.assertFalse(np.array_equal(a.mask, b.mask))

    def test_mc_uniform_close_to_analytic(self):
        scores = np.ones((20, 8))

        def sample_fn(rng):
            keys = rng.uniform(size=scores.shape)
            topk = np.argpartition(-keys, 1, axis=1)[:, :2]
            m = np.zeros_like(scores, dtype=np.uint8)
            rows = np.arange(20)[:, None]
            m[rows, topk] = 1
            return m

        pi = monte_carlo_inclusion_prob(sample_fn, num_volumes=20, num_classes=8, n_draws=3000, seed=1)
        self.assertTrue(np.allclose(pi.mean(axis=0), 2 / 8, atol=0.03))

    def test_weight_diagnostics(self):
        pi = uniform_inclusion_prob(16, 2)
        stats = summarize_weights(pi)
        self.assertAlmostEqual(stats["pi_min"], 0.125)
        self.assertGreater(stats["ess"], 0)

    def test_oracle_inclusion_not_silently_clipped(self):
        """Main modes must fail closed rather than clip Oracle π."""
        feats = np.random.default_rng(3).normal(size=(30, 8))
        s = MissingnessSampler(
            16, 2, mode="conditional", seed=6, n_mc_draws=8000, positivity_floor=0.05,
            uniform_blend=0.55, conditional_strength=1.2,
        )
        out = s.generate([f"v{i}" for i in range(30)], features=feats)
        # Returned π is the MC estimate under floored scores, not a post-clip.
        self.assertGreaterEqual(out.inclusion_prob.min(), 0.05 - 1e-3)
        # selection_score still not equal to inclusion
        self.assertFalse(np.allclose(out.selection_score, out.inclusion_prob))

    def test_structured_coverage_distinguishable_from_random(self):
        """E1 needs pattern signal, not only MCAR after strong blending."""
        n = 80
        rand = MissingnessSampler(16, 2, mode="random", seed=0).generate([f"v{i}" for i in range(n)])
        longtail = MissingnessSampler(16, 2, mode="longtail", seed=0).generate([f"v{i}" for i in range(n)])
        self.assertGreater(coverage_gini(longtail.mask), coverage_gini(rand.mask) + 0.02)
        self.assertGreaterEqual(longtail.inclusion_prob.min(), 0.05 - 1e-3)
        packs = generate_word_masks(
            num_volumes=12,
            num_classes=16,
            labels_per_volume=2,
            modes=("random", "longtail"),
            mask_seeds=(0, 1),
            n_mc_draws=200,
        )
        self.assertEqual(len(packs), 4)
        for key, out in packs.items():
            self.assertEqual(out.mask.shape, (12, 16))
            self.assertIn("maskseed", key)


class TestPartialDataset(unittest.TestCase):
    def _toy(self):
        # 4x4x4 volume, classes 1..3
        full = np.zeros((4, 4, 4), dtype=np.uint8)
        full[0, 0, 0] = 1
        full[1, 1, 1] = 2
        full[2, 2, 2] = 3
        image = np.zeros((1, 4, 4, 4), dtype=np.float32)
        return image, full

    def test_apply_annotation_mask_hides_unannotated(self):
        _, full = self._toy()
        amask = np.array([1, 0, 1], dtype=np.uint8)
        partial = apply_annotation_mask(full, amask)
        self.assertTrue(np.any(partial == 1))
        self.assertFalse(np.any(partial == 2))
        self.assertTrue(np.any(partial == 3))

    def test_presence_not_leaked_from_hidden_gt(self):
        """presence from full GT must be intersected with annotation_mask."""
        _, full = self._toy()
        # full contains organs 1,2,3; only 1 and 3 annotated
        amask = np.array([1, 0, 1], dtype=np.uint8)
        # If someone mistakenly computes presence on full_label:
        raw = patch_presence_mask(full, 3)  # would be [1,1,1]
        safe = patch_presence_mask(full, 3, amask)
        self.assertTrue(np.array_equal(raw, np.array([1, 1, 1], dtype=np.uint8)))
        self.assertTrue(np.array_equal(safe, np.array([1, 0, 1], dtype=np.uint8)))
        self.assertTrue(np.all(safe <= amask))

    def test_presence_vs_annotation(self):
        _, full = self._toy()
        amask = np.array([1, 1, 1], dtype=np.uint8)
        partial = apply_annotation_mask(full, amask)
        # Crop that only contains organ 1
        patch = partial[0:1, 0:1, 0:1]
        pres = patch_presence_mask(patch, 3)
        self.assertTrue(np.array_equal(pres, np.array([1, 0, 0], dtype=np.uint8)))
        # Annotation says all three are annotated in this volume.
        self.assertTrue(np.array_equal(amask, np.array([1, 1, 1], dtype=np.uint8)))

    def test_train_view_no_hidden_leak(self):
        image, full = self._toy()
        view = PartialLabelView(
            ["v0"],
            [image],
            [full],
            np.array([[1, 0, 1]], dtype=np.uint8),
        )
        view.assert_no_hidden_leak(0)
        payload = view.train_sample(0)
        self.assertNotIn("full_label", payload)
        self.assertFalse(np.any(payload["partial_label"] == 2))

    def test_forbid_full_label_in_train_flag(self):
        image, full = self._toy()
        with self.assertRaises(ValueError):
            PartialLabelView(
                ["v0"], [image], [full], np.array([[1, 1, 1]]), allow_full_label_in_train=True
            )

    def test_mask_pack_roundtrip(self):
        s = MissingnessSampler(16, 2, mode="longtail", seed=9, n_mc_draws=300)
        out = s.generate([f"v{i}" for i in range(10)]).as_dict()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "pack.npz")
            save_mask_pack(path, out)
            loaded = load_mask_pack(path)
            self.assertTrue(np.array_equal(loaded["mask"], out["mask"]))
            self.assertTrue(np.allclose(loaded["inclusion_prob"], out["inclusion_prob"]))
            self.assertEqual(loaded["pattern_meta"]["mode"], "longtail")


class TestOracleVsTrainBoundary(unittest.TestCase):
    def test_oracle_dict_includes_full_train_does_not(self):
        image, full = np.zeros((1, 2, 2, 2), np.float32), np.zeros((2, 2, 2), np.uint8)
        full[0, 0, 0] = 1
        view = PartialLabelView(["v"], [image], [full], np.array([[1]]))
        sample = view.get(0, oracle=True)
        self.assertIsNotNone(sample.as_oracle_dict()["full_label"])
        self.assertIsNone(sample.as_train_dict().get("full_label", None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
