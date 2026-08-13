#!/bin/bash
# One fold, one arm, 100 epochs. The cheap probe before committing to twelve.
#
# What this is actually measuring
# -------------------------------
# On the Windows workstation Omnipose ran with `dataloader=False`, because
# `cellpose_omni/core.py` calls `set_start_method('fork')` in the worker branch
# and Windows has no fork. The resource note reports ~22 s/epoch and says CPU
# flow recompute dominates -- but at 256 tiles / batch 4 the GPU side alone is
# ~19 s of that, so how much the worker dataloader buys on Linux is UNKNOWN.
#
# So the first number to read off this job is **seconds per epoch**, not the
# metrics. If it is not meaningfully below the workstation's 22 s, the 20-hour
# estimate for the full run stands and the port bought correctness, not speed.
#
#   sbatch --export=ALL,ARM=paint_out model_labs/omnipose/klone_probe.slurm
#   sbatch --export=ALL,ARM=paint_out,GAPS=1 model_labs/omnipose/klone_probe.slurm
#
#SBATCH --job-name=pm-omni-probe
#SBATCH --account=iscrm
#SBATCH --partition=ckpt-g2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=l40s:1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --requeue
#SBATCH --output=%x-%j.out
set -eo pipefail

PROJECT="${PROJECT:-/gscratch/iscrm/danlovuw/precision_myotube}"
REPO="${REPO:-$PROJECT/repo}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT/envs/pm-omnipose}"
WELL="${WELL:-23_B02_ctrl}"          # smallest well: fastest honest probe
ARM="${ARM:-paint_out}"
EPOCHS="${EPOCHS:-100}"
# The checkpoint tag is built from well + policy + arm flags and does NOT include
# epochs or initialisation, so a second probe of the same fold overwrites the first
# one's checkpoint and manifest in place. Override this for any run whose results
# must not clobber the recorded Stage 1 numbers.
OUT_DIR="${OUT_DIR:-$PROJECT/runs/probe}"

# Lmod's init and conda's hook dereference variables that are unset on a fresh
# klone shell (LD_LIBRARY_PATH, then LD_PRELOAD, and conda has its own). Neither
# is `set -u` clean by design, and naming them one at a time is whack-a-mole. So
# `set -u` goes on AFTER the environment is up, where it guards our own code.
set +u

# `--export=ALL` -- Slurm's default, and what the submit lines above use -- copies
# the submitting shell's environment into the job. If pm-omnipose is active in
# that shell, and it usually is because you just finished verifying it, then
# CONDA_PREFIX comes with it and Lmod REFUSES to load the conda module:
#   "Conda is already active (CONDA_PREFIX is set). Module ... not loaded."
# `set -e` turns that into exit 1 two seconds in, before any of our code runs.
# Probe jobs 38247995 and 38248037 both died exactly here. Strip the inherited
# conda state so the job builds its environment from the module, not the shell.
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER \
      CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M

module load conda/Miniforge3-25.9.1-0
module load cuda/12.8.1
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u

cd "$REPO"
export PYTHONPATH="PrecisionMyotube:annotation_tools:model_labs"

# Where Omnipose looks for pretrained weights. It fetches them from GitHub at model
# construction and caches them under ~/.cellpose/models by default, which is wrong
# here twice over: $HOME is 10 GB and ~80% full, and a ckpt-g2 compute node may have
# no outbound network at all. The weights are staged on /gscratch ahead of time, so
# `cache_model_path` finds the file present and never touches the network.
export CELLPOSE_LOCAL_MODELS_PATH="$PROJECT/cellpose_models"

echo "node $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "cpus-per-task ${SLURM_CPUS_PER_TASK:-?} | arm ${ARM} | gaps ${GAPS:-0}"

# Linux HAS fork, so Omnipose's worker dataloader is available here and the flow
# recompute can leave the critical path. This is the one thing the port unlocks
# that the workstation could not do at all.
ARGS=(--held-out "$WELL" --policy "$ARM" --epochs "$EPOCHS"
      --out "$OUT_DIR"
      --dataloader --num-workers "$((${SLURM_CPUS_PER_TASK:-8} - 2))")
if [[ "${GAPS:-0}" == "1" ]]; then ARGS+=(--augment-gaps); fi

srun python model_labs/omnipose_lab/train_fold.py "${ARGS[@]}"

echo
echo "=== read seconds/epoch from timing.train_seconds / ${EPOCHS} above ==="
echo "workstation baseline was ~22 s/epoch at batch 4, tyx 224, dataloader OFF"
