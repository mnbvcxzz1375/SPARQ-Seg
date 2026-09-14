#!/bin/bash
#SBATCH --job-name=e1_v1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=slurm_logs/e1_%A_%a.out
#SBATCH --error=slurm_logs/e1_%A_%a.err
set -euo pipefail

ROOT=/public/home/heyecheng/SPARQ-Seg
PACK_ROOT=$ROOT/annotation_masks/WORD/E1_v1
ARMS=${E1_ARMS_FILE:-$ROOT/scripts/e1_v1/arms.tsv}
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
WORD=/public/share/td20230405/WORD
EXPECTED_SHA=${EXPECTED_SHA:-b9b4e44b83316160a884d4fb63e226a23f1dda9c}
OPT_SEED=42

IDX=${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}
ARM=$(awk -v i="$IDX" '$1==i {print $2}' "$ARMS")
test -n "${ARM}"
test -f "$PACK_ROOT/${ARM}.npz"
test -f "$PACK_ROOT/manifest.json"
test -f "$ROOT/sparq/e1/train_e1_arm.py"

OUT=$ROOT/runs/E1_v1/${ARM}
mkdir -p "$OUT" "$ROOT/slurm_logs"

if [ -e "$OUT/DONE" ]; then
  echo "Already completed: $ARM"
  exit 0
fi
if [ -d "$OUT" ] && [ "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "Non-empty unfinished run directory: $OUT"
  exit 2
fi

# code pin (best-effort: SPARQ-Seg may not be a git repo on school)
if [ -d "$ROOT/.git" ]; then
  ACTUAL_SHA=$(git -C "$ROOT" rev-parse HEAD)
  if [ -n "$EXPECTED_SHA" ] && [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
    echo "ERROR: code drift: $ACTUAL_SHA != $EXPECTED_SHA"
    exit 3
  fi
fi

echo "array_id=$IDX arm=$ARM pack=$PACK_ROOT/${ARM}.npz out=$OUT"
echo "hostname=$(hostname) opt_seed=$OPT_SEED"
echo "python=$PY torch_will_import"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
cd "$ROOT"

$PY - <<PY
import hashlib, json
from pathlib import Path
pack = Path("$PACK_ROOT") / f"$ARM.npz"
man = json.loads(Path("$PACK_ROOT/manifest.json").read_text())
h = hashlib.sha256(pack.read_bytes()).hexdigest()
exp = man["packs"][pack.name]["sha256"]
assert h == exp, (h, exp)
print("pack_sha_ok", h)
PY

$PY "$ROOT/sparq/e1/train_e1_arm.py" \
  --arm "$ARM" \
  --pack-root "$PACK_ROOT" \
  --images-root "$WORD/imagesTr" \
  --labels-full-root "$WORD/labelsTr_All" \
  --out-dir "$OUT" \
  --epoches 500 \
  --batch-size 1 \
  --lr 0.01 \
  --samples-per-epoch 200 \
  --optimization-seed "$OPT_SEED" \
  --expected-code-sha "$EXPECTED_SHA"

echo "FINISHED $ARM"
