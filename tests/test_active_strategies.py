"""AOVA Phase A: acquisition strategies + pack builder contracts (no torch)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparq.active.strategies import (
    candidate_cells,
    coverage,
    score_entropy_x_coverage,
    select_class_balanced,
    select_random,
    select_top_scores,
)

REPO = Path(__file__).resolve().parents[1]


def _load_aova_select():
    spec = importlib.util.spec_from_file_location(
        "aova_select", REPO / "tools" / "aova_select.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def base_mask(n=20, c=6, k=2, seed=3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = np.zeros((n, c), dtype=np.uint8)
    for v in range(n):
        m[v, rng.choice(c, size=k, replace=False)] = 1
    return m


class TestStrategies(unittest.TestCase):
    def setUp(self):
        self.mask = base_mask()

    def test_budget_and_disjoint(self):
        for strat, kw in (
            (select_random, {"rng": np.random.default_rng(0)}),
            (select_class_balanced, {"rng": np.random.default_rng(0)}),
        ):
            add = strat(self.mask, 5, **kw)
            self.assertEqual(int(add.sum()), 5)
            self.assertFalse((add & (self.mask == 1)).any(), "picked annotated cell")

    def test_top_scores_exact(self):
        scores = np.arange(self.mask.size).reshape(self.mask.shape).astype(float)
        add = select_top_scores(self.mask, scores, 4)
        got = sorted(map(tuple, np.argwhere(add)))
        cands = [(v, c) for v, c in candidate_cells(self.mask)]
        want = sorted(map(tuple, max(cands, key=lambda p: 0) and sorted(
            cands, key=lambda p: -scores[p[0], p[1]])[:4]))
        self.assertEqual(got, want)

    def test_determinism(self):
        a = select_random(self.mask, 8, np.random.default_rng(7))
        b = select_random(self.mask, 8, np.random.default_rng(7))
        self.assertTrue(np.array_equal(a, b))

    def test_class_balanced_prefers_underannotated(self):
        mask = np.zeros((40, 4), dtype=np.uint8)
        mask[:, 0] = 1          # fully annotated
        mask[:8, 1] = 1         # 20%
        # classes 2,3: 0%
        add = select_class_balanced(mask, 30, np.random.default_rng(1))
        by_class = add.sum(axis=0)
        self.assertEqual(int(by_class[0]), 0, "annotated class picked")
        self.assertGreater(int(by_class[2]) + int(by_class[3]), int(by_class[1]),
                           "under-annotated classes should dominate")

    def test_entropy_coverage_boosts_rare(self):
        mask = base_mask(n=30, c=4, k=1)
        ent = np.ones_like(mask, dtype=float)
        pri = score_entropy_x_coverage(ent, mask)
        cov = coverage(mask)
        rare = int(np.argmin(cov))
        common = int(np.argmax(cov))
        col_r = pri[:, rare][mask[:, rare] == 0]
        col_c = pri[:, common][mask[:, common] == 0]
        self.assertGreater(float(col_r.mean()), float(col_c.mean()))


class TestAovaSelect(unittest.TestCase):
    def _make_base(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        mask = base_mask(n=12, c=4, k=1)
        np.savez_compressed(root / "random_moderate_s0.npz", mask=mask,
                            selection_score=np.zeros_like(mask, dtype=np.float32),
                            inclusion_prob=np.ones_like(mask, dtype=np.float32) * 0.25)
        h = hashlib.sha256((root / "random_moderate_s0.npz").read_bytes()).hexdigest()
        (root / "manifest.json").write_text(json.dumps(
            {"packs": {"random_moderate_s0.npz": {"sha256": h}}}))
        (root / "volume_order.txt").write_text("v0\nv1\nv2\n")
        return root

    def test_random_augment_contract(self):
        mod = _load_aova_select()
        with tempfile.TemporaryDirectory() as td:
            base = self._make_base(Path(td) / "base")
            out = Path(td) / "aug"
            argv = ["aova_select", "--base-pack-root", str(base),
                    "--base-arm", "random_moderate_s0", "--strategy", "random",
                    "--budget", "6", "--rng-seed", "0", "--out-root", str(out)]
            old = sys.argv
            sys.argv = argv
            try:
                rc = mod.main()
            finally:
                sys.argv = old
            self.assertEqual(rc, 0)
            man = json.loads((out / "manifest.json").read_text())
            name = next(iter(man["packs"]))
            with np.load(out / name) as z:
                mask = z["mask"].copy()
            with np.load(base / "random_moderate_s0.npz") as zf:
                bmask = zf["mask"].copy()
            self.assertTrue(((mask == 1) | (bmask == 0)).all(), "base annotations lost")
            self.assertEqual(int(mask.sum()) - int(bmask.sum()), 6)
            self.assertEqual(man["packs"][name]["sha256"],
                             hashlib.sha256((out / name).read_bytes()).hexdigest())
            self.assertEqual(man["packs"][name]["meta"]["design"], "acquisition")
            self.assertTrue((out / "volume_order.txt").exists())

    def test_manifest_merges_multiple_strategies(self):
        mod = _load_aova_select()
        with tempfile.TemporaryDirectory() as td:
            base = self._make_base(Path(td) / "base")
            out = Path(td) / "aug"
            for strat in ("random", "class_balanced"):
                old = sys.argv
                sys.argv = ["aova_select", "--base-pack-root", str(base),
                            "--base-arm", "random_moderate_s0", "--strategy", strat,
                            "--budget", "3", "--rng-seed", "0", "--out-root", str(out)]
                try:
                    mod.main()
                finally:
                    sys.argv = old
            man = json.loads((out / "manifest.json").read_text())
            self.assertEqual(len(man["packs"]), 2, "second run clobbered manifest")
            for name, entry in man["packs"].items():
                self.assertEqual(entry["sha256"],
                                 hashlib.sha256((out / name).read_bytes()).hexdigest())

    def test_entropy_picks_max_cells(self):
        mod = _load_aova_select()
        with tempfile.TemporaryDirectory() as td:
            base = self._make_base(Path(td) / "base")
            ent = np.zeros((12, 4), dtype=np.float32)
            with np.load(base / "random_moderate_s0.npz") as zf:
                bmask = zf["mask"].copy()
            free = [tuple(p) for p in np.argwhere(bmask == 0)]
            target = free[0]
            ent[target] = 10.0
            ef = Path(td) / "ent.npz"
            np.savez_compressed(ef, entropy=ent)
            out = Path(td) / "aug"
            old = sys.argv
            sys.argv = ["aova_select", "--base-pack-root", str(base),
                        "--base-arm", "random_moderate_s0", "--strategy", "entropy",
                        "--budget", "1", "--entropy-file", str(ef), "--out-root", str(out)]
            try:
                mod.main()
            finally:
                sys.argv = old
            man = json.loads((out / "manifest.json").read_text())
            with np.load(out / next(iter(man["packs"]))) as z:
                added = z["mask"] - bmask
            self.assertEqual(tuple(np.argwhere(added)[0]), target)

    def test_corrupt_base_rejected(self):
        mod = _load_aova_select()
        with tempfile.TemporaryDirectory() as td:
            base = self._make_base(Path(td) / "base")
            np.savez_compressed(base / "random_moderate_s0.npz",
                                mask=np.ones((12, 4), dtype=np.uint8))  # tampered
            old = sys.argv
            sys.argv = ["aova_select", "--base-pack-root", str(base),
                        "--base-arm", "random_moderate_s0", "--strategy", "random",
                        "--budget", "2", "--out-root", str(Path(td) / "aug")]
            try:
                with self.assertRaises(RuntimeError):
                    mod.main()
            finally:
                sys.argv = old


if __name__ == "__main__":
    unittest.main()
