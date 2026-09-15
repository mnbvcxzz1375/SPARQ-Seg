#!/usr/bin/env python3
"""Dry-run: validate 12 E1 arms map 1:1 to E1-v1 pack without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--expected-code-sha", default="8a4169dc2d15fd5163102f65279f66e07fa5b88a")
    args = ap.parse_args()
    root = Path(args.root)
    pack_root = root / "annotation_masks" / "WORD" / "E1_v1"
    arms_tsv = root / "scripts" / "e1_v1" / "arms.tsv"
    man = json.loads((pack_root / "manifest.json").read_text(encoding="utf-8"))

    rows = []
    for line in arms_tsv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        i, arm = line.split("\t")
        rows.append((int(i), arm))
    if len(rows) != 12:
        print("FAIL arms.tsv count", len(rows))
        return 1
    if len({a for _, a in rows}) != 12:
        print("FAIL duplicate arms")
        return 1

    manifest_arms = {Path(k).stem for k in man["packs"]}
    launcher_arms = {a for _, a in rows}
    if launcher_arms != manifest_arms:
        print("FAIL set mismatch", launcher_arms ^ manifest_arms)
        return 1

    ok = True
    for i, arm in rows:
        npz = pack_root / f"{arm}.npz"
        if not npz.exists():
            print("FAIL missing", npz)
            ok = False
            continue
        digest = sha256_file(npz)
        exp = man["packs"][npz.name]["sha256"]
        match = digest == exp
        ok = ok and match
        print(
            json.dumps(
                {
                    "array_id": i,
                    "arm": arm,
                    "npz_path": str(npz),
                    "npz_sha256": digest,
                    "sha_ok": match,
                    "output_dir": f"runs/E1_v1/{arm}",
                    "code_sha": args.expected_code_sha,
                    "opt_seed": 42,
                }
            )
        )
    print(json.dumps({"ok": ok, "n_tasks": len(rows), "unique_arms": len(launcher_arms)}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
