#!/bin/bash
# Zero-shot Omnipose on the PLATE_23 bootstrap fields. NO training.
#
# This is minutes of GPU, not hours, so the preemption problem that killed both
# Stage-2 attempts barely applies -- a preempted job here loses a few minutes and
# `--requeue` restarts it. That is the whole point of running this first.
#
#   sbatch model_labs/omnipose/klone_zeroshot.slurm
#   sbatch --export=ALL,CROP=0 model_labs/omnipose/klone_zeroshot.slurm   # full field
#
#SBATCH --job-name=omni-zeroshot
#SBATCH --account=iscrm
#SBATCH --partition=ckpt-g2
#SBATCH --gpus=l40s:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

PROJECT="${PROJECT:-/gscratch/iscrm/danlovuw/precision_myotube}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT/envs/pm-omnipose}"
REPO="${REPO:-$PROJECT/repo}"
CROP="${CROP:-1024}"
WELLS="${WELLS:-23_B02_ctrl 32_C08_br223_igf1r}"
MODELS="${MODELS:-bact_phase_affinity}"

# Lmod and conda are not `set -u` clean; guard our own code only, after they load.
set +u
module load conda/Miniforge3-25.9.1-0
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u

# Weights live on /gscratch, not in $HOME. Without this, cellpose_omni downloads
# to ~/.cellpose/models -- which blows the 10 GB $HOME quota and, on a compute
# node with no outbound network, simply fails. Same line as klone_folds.slurm:76
# and klone_probe.slurm:76; keep the three in agreement.
export CELLPOSE_LOCAL_MODELS_PATH="$PROJECT/cellpose_models"

cd "$REPO"
echo "node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "requeue count: ${SLURM_RESTART_COUNT:-0}"
echo "model cache: $CELLPOSE_LOCAL_MODELS_PATH"
ls -1 "$CELLPOSE_LOCAL_MODELS_PATH" 2>/dev/null || echo "  (cache dir missing)"

# Fail loudly HERE rather than mid-run: a compute node has no outbound network,
# so any weight not already in the cache cannot be fetched. Pre-download it from
# a LOGIN node (which does have network) before submitting.
for m in $MODELS; do
  if ! ls "$CELLPOSE_LOCAL_MODELS_PATH" 2>/dev/null | grep -q "^${m}"; then
    echo "ERROR: '$m' is not in the model cache and this node cannot download it." >&2
    echo "  On a LOGIN node run:" >&2
    echo "    export CELLPOSE_LOCAL_MODELS_PATH=$CELLPOSE_LOCAL_MODELS_PATH" >&2
    echo "    python -c \"from cellpose_omni import models; models.CellposeModel(gpu=False, omni=True, model_type='$m')\"" >&2
    exit 1
  fi
done

# Discover what this install actually ships before assuming a name exists.
echo "=== available pretrained models ==="
python model_labs/omnipose_lab/zeroshot.py --list-models

echo
echo "=== zero-shot run ==="
# shellcheck disable=SC2086
srun python model_labs/omnipose_lab/zeroshot.py \
  --models $MODELS \
  --wells $WELLS \
  --polarity both \
  --crop "$CROP" \
  --out model_labs/omnipose_lab/_runs/zeroshot
