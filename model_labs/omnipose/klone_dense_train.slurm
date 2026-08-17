#!/bin/bash
# Full training run on the dense operator-traced corpus.
#
# Measured on the 5-epoch smoke run (job 38613379, L40S):
#   ~12 s/epoch, peak GPU 27.86 GB of 46, data prep 79 s
# so 300 epochs is ~1 hour. The preemptible partition is fine at this scale:
# cellpose_omni checkpoints EVERY epoch, so a preemption costs ~12 s of work,
# not the run. That is why this does not need a rented GPU.
#
# One model, not twelve. The six-fold leave-one-well-out design existed because
# 375 sparse instances could not spare a test set; this corpus has ~6,800 whole
# fibres and PLATE_23 is a genuinely independent test set -- different plate,
# different session, different annotation method -- so leakage is structurally
# impossible rather than merely audited.
#
#   sbatch model_labs/omnipose/klone_dense_train.slurm
#   sbatch --export=ALL,EPOCHS=500 model_labs/omnipose/klone_dense_train.slurm
#
#SBATCH --job-name=omni-dense
#SBATCH --account=iscrm
#SBATCH --partition=ckpt-g2
#SBATCH --gpus=l40s:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

PROJECT="${PROJECT:-/gscratch/iscrm/danlovuw/precision_myotube}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT/envs/pm-omnipose}"
REPO="${REPO:-$PROJECT/repo}"
CORPUS="${CORPUS:-$REPO/PrecisionMyotube/annotation_work/plate32_dense_v1}"
EPOCHS="${EPOCHS:-300}"
HELD_OUT="${HELD_OUT:-B02}"
OUT="${OUT:-$PROJECT/runs/dense_v1}"

set +u
module load conda/Miniforge3-25.9.1-0
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u

export CELLPOSE_LOCAL_MODELS_PATH="$PROJECT/cellpose_models"
cd "$REPO"

echo "node $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "requeue count: ${SLURM_RESTART_COUNT:-0}"
echo "corpus  $CORPUS"
echo "epochs  $EPOCHS   held out  $HELD_OUT"
echo "out     $OUT"
[[ -d "$CORPUS" ]] || { echo "ERROR: corpus not found" >&2; exit 1; }

# The smoke run proved this build accepts AND uses links. Re-assert it here so a
# silent upstream change can never turn 13,619 links into zero without saying so.
python - <<'PY'
import inspect, sys
from cellpose_omni import models
if "train_links" not in inspect.signature(models.CellposeModel.train).parameters:
    print("FAIL: train_links vanished from the API", file=sys.stderr); sys.exit(1)
print("train_links: present")
PY

srun python model_labs/omnipose_lab/train_fold.py \
  --corpus "$CORPUS" \
  --held-out "$HELD_OUT" \
  --epochs "$EPOCHS" \
  --window-px 1280 --overlap 0.25 \
  --dataloader --num-workers 6 \
  --out "$OUT"

echo
echo "=== done. next: inference on the PLATE_23 test set ==="
