#!/bin/bash
# Dense-corpus training on UW Tillicum (H200). The point of moving here from
# klone is preemption: `iscrm` owns no GPUs, so every klone GPU job runs on the
# checkpoint partition and is killed whenever the node's owner wants it back --
# four attempts, four restarts from epoch 0, nothing finished. `hrbomics` has no
# checkpoint QOS and gpu-h200 is not a preemptible partition, so this run gets
# an uninterrupted hour.
#
# Use gpu-h200, NOT gpu-h200-mig: the MIG slices are 18 GB and the measured peak
# is 27.86 GB, so a MIG job would OOM.
#
#   sbatch model_labs/omnipose/tillicum_dense_train.slurm
#   sbatch --export=ALL,EPOCHS=60 model_labs/omnipose/tillicum_dense_train.slurm
#
#SBATCH --job-name=omni-dense
#SBATCH --account=hrbomics
#SBATCH --partition=gpu-h200
#SBATCH --gpus=h200:1
#SBATCH -c 8
#SBATCH --mem=96G
# Billed on hours REQUESTED, not used. 300 epochs at the measured ~11.5 s/epoch
# is ~1 h plus ~5 min startup, so 1.5 h is margin without paying for idle
# headroom. Raise it if you raise --epochs.
#SBATCH --time=1:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

PROJECT="${PROJECT:-/gpfs/scrubbed/danlovuw/precision_myotube}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT/envs/pm-omnipose}"
REPO="${REPO:-$PROJECT/repo}"
CORPUS="${CORPUS:-$REPO/PrecisionMyotube/annotation_work/plate32_dense_v1}"
EPOCHS="${EPOCHS:-300}"
HELD_OUT="${HELD_OUT:-B02}"
OUT="${OUT:-$PROJECT/runs/dense_v1}"

# Call the environment's interpreter by absolute path -- do NOT `conda activate`.
# Submitting from an already-activated shell exports CONDA_PREFIX into the job;
# re-activating on top of that leaves `python` resolving to the wrong
# interpreter, and because `conda activate` still returns 0 the script sails past
# it and dies later with ModuleNotFoundError. An absolute path cannot do that.
PY="$ENV_PREFIX/bin/python"
[[ -x "$PY" ]] || { echo "ERROR: no interpreter at $PY" >&2; exit 1; }
"$PY" -c "import cellpose_omni, torch; print('env OK:', torch.__version__)" \
  || { echo "ERROR: $PY cannot import cellpose_omni" >&2; exit 1; }

# Weights live beside the project, not in $HOME. Compute nodes may be firewalled
# even where the login node is not, so these are prefetched before submitting.
export CELLPOSE_LOCAL_MODELS_PATH="$PROJECT/cellpose_models"

cd "$REPO"
echo "node $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "corpus  $CORPUS"
echo "epochs  $EPOCHS   held out  $HELD_OUT"
echo "out     $OUT"
[[ -d "$CORPUS" ]] || { echo "ERROR: corpus not found at $CORPUS" >&2; exit 1; }

# 13,619 links carry 59.5% of the fibres. If an upstream change ever drops the
# parameter, a run would silently train false splits at every crossing.
"$PY" - <<'PY'
import inspect, sys
from cellpose_omni import models
if "train_links" not in inspect.signature(models.CellposeModel.train).parameters:
    print("FAIL: train_links missing from this build", file=sys.stderr); sys.exit(1)
print("train_links: present")
PY

srun "$PY" model_labs/omnipose_lab/train_fold.py \
  --corpus "$CORPUS" \
  --held-out "$HELD_OUT" \
  --epochs "$EPOCHS" \
  --window-px 1280 --overlap 0.25 \
  --dataloader --num-workers 6 \
  --out "$OUT"

echo
echo "=== done ==="
echo "checkpoint: $OUT/checkpoints/"
echo "COPY IT OFF /gpfs/scrubbed -- that filesystem is purged on a schedule."
