#!/bin/bash
# Build the pm-omnipose environment on UW Hyak (klone). Run ONCE, interactively.
#
# Why not `conda env create -f environment.yml`
# ---------------------------------------------
# Same reason as on the workstation: `pip install omnipose` pulls a CPU-only
# torch and will silently REPLACE a correct CUDA build (observed: 2.11.0+cu128 ->
# 2.13.0+cpu, exit code 0). Order matters, so this is a script and not a yaml.
#
# klone-specific facts this encodes
# ---------------------------------
# * $HOME is capped at 10 GB and is already ~80% full, so the environment MUST
#   live on /gscratch. Conda envs also preserve access times, which makes them
#   candidates for the scrubbed-storage erasure policy -- /gscratch/iscrm is
#   project storage and is not scrubbed.
# * `iscrm` owns no GPU allocation (hyakalloc shows GPUS: 0 on all three of its
#   partitions), so every GPU job goes through checkpoint. Build the environment
#   on a GPU node anyway: the torch install needs to be verified against a real
#   kernel launch, and that cannot be done on a login node.
# * cuda/12.8.1 is available as a module, which matches the cu128 wheel index the
#   workstation environment already pins. The L40S is sm_89 and inside that
#   build's architecture list, so the pinned versions carry over unchanged.
#
# Usage:
#   salloc -A iscrm -p ckpt-g2 --gpus=l40s:1 -c 16 --mem=64G -t 2:00:00
#   bash /gscratch/iscrm/danlovuw/precision_myotube/repo/model_labs/omnipose/klone_setup.sh
set -eo pipefail

PROJECT="${PROJECT:-/gscratch/iscrm/danlovuw/precision_myotube}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT/envs/pm-omnipose}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "ERROR: run this inside an salloc session on a GPU node." >&2
  echo "  salloc -A iscrm -p ckpt-g2 --gpus=l40s:1 -c 16 --mem=64G -t 2:00:00" >&2
  exit 1
fi

# Lmod's init and conda's hook dereference variables that are unset on a fresh
# klone shell (LD_LIBRARY_PATH, then LD_PRELOAD, and conda has its own). Neither
# is `set -u` clean by design, and naming them one at a time is whack-a-mole. So
# `set -u` goes on AFTER the environment is up, where it guards our own code.
set +u
module load conda/Miniforge3-25.9.1-0
module load cuda/12.8.1

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not on PATH after 'module load conda'." >&2
  echo "  check the module name with: module avail conda" >&2
  exit 1
fi

# Package and env caches default to $HOME and will blow the 10 GB quota.
export CONDA_PKGS_DIRS="$PROJECT/.conda_pkgs"
export PIP_CACHE_DIR="$PROJECT/.pip_cache"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$(dirname "$ENV_PREFIX")"

conda create --prefix "$ENV_PREFIX" python=3.10 -y
# `source activate` is the deprecated path and does not reliably exist under a
# module-provided conda; sourcing the hook defines `conda activate` properly.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u

# ORDER MATTERS -- omnipose first, then force the CUDA torch back over the top.
pip install omnipose==1.1.4
pip install --force-reinstall torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install tifffile==2023.2.28 scikit-image==0.25.2 scipy==1.15.3 numpy==2.2.6

echo
echo "=== verifying a REAL kernel launch, not just torch.cuda.is_available() ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda)
assert torch.cuda.is_available(), "no CUDA device visible"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"device {name} sm_{cap[0]}{cap[1]}")
# is_available() returns True on a GPU whose architecture the build does not
# cover; the failure only surfaces at the first kernel launch. So launch one.
x = torch.randn(512, 512, device="cuda")
torch.cuda.synchronize()
print("matmul ok:", float((x @ x).sum()) == float((x @ x).sum()))
print(f"arch list: {torch.cuda.get_arch_list()}")
assert f"sm_{cap[0]}{cap[1]}" in torch.cuda.get_arch_list(), (
    f"this torch build does not cover sm_{cap[0]}{cap[1]}")
PY

echo
echo "environment ready: $ENV_PREFIX"
echo "activate with:"
echo "  module load conda/Miniforge3-25.9.1-0 cuda/12.8.1"
echo "  source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate $ENV_PREFIX"
