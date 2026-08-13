#!/bin/bash
# The full run: six leave-one-well-out folds, gap augmentation off and on.
#
# Two array tasks, not twelve
# ---------------------------
# `run_folds.py` already loops the six wells and resumes per fold from its
# sidecar, so the array indexes the ABLATION (gaps off / gaps on) and lets the
# script own the fold loop. That keeps the paired comparison inside one process
# per arm, and it means a preemption costs one fold rather than one arm.
#
# Preemption is not an edge case here. `iscrm` owns no GPU allocation at all --
# hyakalloc reports GPUS: 0 on compute, compute-hugemem and cpu-g2 -- so every
# GPU job runs on checkpoint and WILL be stopped and requeued roughly every 8-9
# hours without notice. `--requeue` plus the per-fold sidecars is what makes that
# survivable; do not remove either.
#
#   sbatch model_labs/omnipose/klone_folds.slurm
#
#SBATCH --job-name=pm-omni-folds
#SBATCH --account=iscrm
#SBATCH --partition=ckpt-g2
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=l40s:1
#SBATCH --mem=64G
# `ckpt-g2` reports TIMELIMIT=infinite, so the old 8 h here was self-imposed and it was
# actively harmful: `--requeue` covers preemption and node failure but NOT `TIMEOUT`, so a
# job that hits its own wall dies permanently and needs manual resubmission. Attempt 1
# (38384273) died exactly that way with zero folds written. With a generous limit the only
# interruption left is preemption, which requeues automatically. 48 h is ~3x the slowest
# arm's projected 15.7 h, which is margin for a slower-than-expected fold, not a target.
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=%x-%A_%a.out
set -eo pipefail

PROJECT="${PROJECT:-/gscratch/iscrm/danlovuw/precision_myotube}"
REPO="${REPO:-$PROJECT/repo}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT/envs/pm-omnipose}"
EPOCHS="${EPOCHS:-300}"

# Lmod's init and conda's hook dereference variables that are unset on a fresh
# klone shell (LD_LIBRARY_PATH, then LD_PRELOAD, and conda has its own). Neither
# is `set -u` clean by design, and naming them one at a time is whack-a-mole. So
# `set -u` goes on AFTER the environment is up, where it guards our own code.
set +u

# `--export=ALL` -- Slurm's default -- copies the submitting shell's environment
# into the job. If pm-omnipose is active in that shell, and it usually is because
# you just finished verifying it, then CONDA_PREFIX comes with it and Lmod
# REFUSES to load the conda module:
#   "Conda is already active (CONDA_PREFIX is set). Module ... not loaded."
# `set -e` turns that into exit 1 two seconds in, before any of our code runs.
# This is what killed both probe jobs on 2026-08-06. Strip the inherited conda
# state so the job builds its environment from the module, not the shell.
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

# Both array tasks write into the SAME run directory on purpose: the gap arms
# differ only in their tile set, so a paired per-well comparison across them is
# the point. `run_folds.py` separates their sidecars by name, so they do not
# overwrite each other.
OUT="$PROJECT/runs/omnipose_v1"
# `--dataloader` is NOT optional here. Without it Omnipose recomputes flows in-process and
# the CPU becomes the critical path: the workstation measured ~57 s/epoch that way against
# 20.7 s/epoch with workers. The first Stage-2 attempt (38384273) omitted these two flags,
# ran the slow path, and timed out at 8 h having completed zero folds on both arms. The
# probe script always passed them, which is why its timing looked fine and did not
# describe this run. Keep the two scripts in agreement.
ARGS=(--out "$OUT" --arms paint_out --epochs "$EPOCHS"
      --dataloader --num-workers "$((${SLURM_CPUS_PER_TASK:-8} - 2))")
if [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
  ARGS+=(--augment-gaps)
  echo "=== arm: gap augmentation ON ==="
else
  echo "=== arm: gap augmentation OFF (baseline) ==="
fi

echo "node $(hostname) | $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "requeue count: ${SLURM_RESTART_COUNT:-0}"

# Resume is the default in run_folds.py; on a requeue it replays completed folds
# from their sidecars and picks up where it stopped.
srun python model_labs/omnipose_lab/run_folds.py "${ARGS[@]}"
