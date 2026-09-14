#!/bin/bash
# Standalone E1 arm launcher for 3080 / 40901 / 40902 (no Slurm).
set -euo pipefail

HOST_ROOT=${HOST_ROOT:-/data/hyc/SPARQ-Seg}
ARM=${1:?arm name e.g. random_moderate_s0}
GPU=${2:-0}
PLSEG_CODE_ROOT=${PLSEG_CODE_ROOT:?set PLSEG_CODE_ROOT}
PY=${PY:-/home/ubuntu/anaconda3/envs/vllmenv/bin/python}
WORD=${WORD:-/data/hyc/PLS4MIS/code/datasets/WORD}
PACK_ROOT=${PACK_ROOT:-$HOST_ROOT/annotation_masks/WORD/E1_v1}
OPT_SEED=42
EPOCHES=${EPOCHES:-500}
BATCH=${BATCH:-1}

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=$HOST_ROOT:${PYTHONPATH:-}
export PLSEG_CODE_ROOT

OUT=$HOST_ROOT/runs/E1_v1/$ARM
mkdir -p "$OUT" "$HOST_ROOT/logs"

if [ -e "$OUT/DONE" ]; then
  echo "Already completed: $ARM"
  exit 0
fi
if [ -d "$OUT" ] && [ "$(ls -A "$OUT" 2>/dev/null)" ]; then
  # allow recovery from FAILED nan runs
  if [ -e "$OUT/FAILED" ]; then
    echo "Cleaning FAILED run: $OUT"
    rm -rf "$OUT"
    mkdir -p "$OUT"
  else
    echo "Non-empty unfinished run directory: $OUT"
    exit 2
  fi
fi

test -f "$PACK_ROOT/${ARM}.npz"
test -f "$PACK_ROOT/manifest.json"
test -f "$HOST_ROOT/sparq/e1/train_e1_arm.py"

echo "host=$(hostname) arm=$ARM gpu=$GPU pack=$PACK_ROOT/${ARM}.npz out=$OUT opt_seed=$OPT_SEED"

"$PY" - <<PY
import hashlib, json
from pathlib import Path
pack = Path("$PACK_ROOT") / f"$ARM.npz"
man = json.loads(Path("$PACK_ROOT/manifest.json").read_text())
h = hashlib.sha256(pack.read_bytes()).hexdigest()
exp = man["packs"][pack.name]["sha256"]
assert h == exp, (h, exp)
print("pack_sha_ok", h)
PY

"$PY" "$HOST_ROOT/sparq/e1/train_e1_arm.py" \
  --arm "$ARM" \
  --pack-root "$PACK_ROOT" \
  --images-root "$WORD/imagesTr" \
  --labels-full-root "$WORD/labelsTr_All" \
  --out-dir "$OUT" \
  --epoches "$EPOCHES" \
  --batch-size "$BATCH" \
  --lr 0.01 \
  --samples-per-epoch 200 \
  --optimization-seed "$OPT_SEED" \
  2>&1 | tee "$HOST_ROOT/logs/${ARM}_gpu${GPU}.log"

echo "FINISHED $ARM"
