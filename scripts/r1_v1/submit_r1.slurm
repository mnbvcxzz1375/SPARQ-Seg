#!/bin/bash
#SBATCH --job-name=r1_v1
#SBATCH --partition=gpu_4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=slurm_logs/r1_%A_%a.out
#SBATCH --error=slurm_logs/r1_%A_%a.err
# R1_v1: re-run the route-deciding arms under the CERTIFIED reference recipe
# (calib/train_bridge.py + shared train_step; equivalence/counterfactual/
# integration all PASS; anchor 0.759786 reproduced). Fail-closed entry.
set -euo pipefail

ROOT=/public/home/heyecheng/SPARQ-Seg
ARMS=$ROOT/scripts/r1_v1/arms.tsv
PY=/public/home/heyecheng/odin2026_local_baseline/.venv/bin/python
WORD=/public/share/td20230405/WORD
IDX=${SLURM_ARRAY_TASK_ID:?missing SLURM_ARRAY_TASK_ID}

ARM=$(awk -v i="$IDX" '$1==i {print $2}' "$ARMS")
PROOT=$(awk -v i="$IDX" '$1==i {print $3}' "$ARMS")
PNAME=$(awk -v i="$IDX" '$1==i {print $4}' "$ARMS")
test -n "$ARM" && test -n "$PROOT" && test -n "$PNAME"
test -f "$PROOT/$PNAME"
test -f "$PROOT/manifest.json"

OUT=$ROOT/runs/R1_v1/$ARM
mkdir -p "$OUT" "$ROOT/slurm_logs"
if [ -e "$OUT/DONE" ]; then
  echo "Already completed: $ARM"; exit 0
fi

echo "r1 arm=$ARM pack=$PROOT/$PNAME out=$OUT host=$(hostname)"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

"$PY" calib/train_bridge.py \
  --pack-root "$PROOT" --pack-name "$PNAME" \
  --images-root "$WORD/imagesTr" \
  --labels-full-root "$WORD/labelsTr_All" \
  --val-images-root "$WORD/imagesVal" \
  --val-labels-root "$WORD/labelsVal" \
  --out-dir "$OUT" \
  --epoches 500 --batch-size 4 --lr 0.01 --opt-seed 42

echo "FINISHED $ARM"
