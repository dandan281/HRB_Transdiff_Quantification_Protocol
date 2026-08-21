#!/bin/bash
# Learning-rate sweep. The 300-epoch run collapsed to a constant output: the
# model emits a flat -3.804 distance field on its own training data, which is
# roughly the background value of a target that is ~94% background. It learned
# the mean and never learned to discriminate.
#
# `diagnose_target.py` already cleared the data: labels carry 5-7% foreground,
# links resolve, and the target has correct distance and flow channels. So this
# sweeps the one knob most likely responsible — lr=0.1 constant with SGD is
# Cellpose's FROM-SCRATCH default, not a fine-tuning rate.
#
# 30 epochs each is enough: the failed run had already flatlined by epoch 20, so
# a rate that is going to work will visibly out-descend 0.1 well inside 30.
#
#   sbatch model_labs/omnipose/tillicum_lr_sweep.slurm
#
#SBATCH --job-name=omni-lr
#SBATCH --account=hrbomics
#SBATCH --partition=gpu-h200
#SBATCH --gpus=h200:1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH --time=1:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

PROJECT="${PROJECT:-/gpfs/scrubbed/danlovuw/precision_myotube}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT/envs/pm-omnipose}"
REPO="${REPO:-$PROJECT/repo}"
CORPUS="${CORPUS:-$REPO/PrecisionMyotube/annotation_work/plate32_dense_v1}"
EPOCHS="${EPOCHS:-30}"
RATES="${RATES:-0.1 0.01 0.001 0.0001}"

PY="$ENV_PREFIX/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: no interpreter at $PY" >&2; exit 1; }
export CELLPOSE_LOCAL_MODELS_PATH="$PROJECT/cellpose_models"
cd "$REPO"

echo "node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "rates: $RATES   epochs each: $EPOCHS"
echo

for LR in $RATES; do
  echo "================================================================"
  echo "=== lr = $LR ==="
  echo "================================================================"
  OUT="$PROJECT/runs/lr_sweep/lr_$LR"
  srun "$PY" model_labs/omnipose_lab/train_fold.py \
    --corpus "$CORPUS" \
    --held-out B02 \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --window-px 1280 --overlap 0.25 \
    --dataloader --num-workers 6 \
    --out "$OUT" 2>&1 | grep -E "Train epoch|fold data|trained " || true
  echo
done

echo "================================================================"
echo "=== loss at epochs 0 / 10 / 20 / 29, per rate ==="
echo "================================================================"
for LR in $RATES; do
  echo "--- lr = $LR"
  grep -h "Train epoch" "$SLURM_JOB_NAME-$SLURM_JOB_ID.out" 2>/dev/null \
    | awk -v lr="$LR" 'BEGIN{p=0} /lr = /{p=0} {print}' >/dev/null 2>&1 || true
done
echo "(read the per-rate blocks above; a working rate keeps descending past"
echo " epoch 20, a collapsing one flatlines by epoch 5 like the 300-epoch run)"
