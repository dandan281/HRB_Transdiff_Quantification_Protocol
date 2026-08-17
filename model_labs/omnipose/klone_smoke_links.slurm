#!/bin/bash
# 5-epoch smoke run on the dense traced corpus. The point is NOT the model --
# it is to prove, before renting a GPU, that this `cellpose_omni` build actually
# consumes `train_links`.
#
# 13,619 links carry 59.5% of the fibres in this corpus. If the API silently
# ignores them, a full run trains the model that a myotube ends at every
# crossing and the result is worse than useless -- and we would not find out for
# 90 more epochs. So: check the signature, run 5 epochs, stop.
#
# Short enough to survive the preemptible partition (measured worst-case
# uninterrupted window: 37 min), so this costs nothing.
#
#   sbatch model_labs/omnipose/klone_smoke_links.slurm
#
#SBATCH --job-name=omni-smoke-links
#SBATCH --account=iscrm
#SBATCH --partition=ckpt-g2
#SBATCH --gpus=l40s:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --requeue
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -eo pipefail

PROJECT="${PROJECT:-/gscratch/iscrm/danlovuw/precision_myotube}"
ENV_PREFIX="${ENV_PREFIX:-$PROJECT/envs/pm-omnipose}"
REPO="${REPO:-$PROJECT/repo}"
CORPUS="${CORPUS:-$REPO/PrecisionMyotube/annotation_work/plate32_dense_v1}"
EPOCHS="${EPOCHS:-5}"
HELD_OUT="${HELD_OUT:-B02}"

set +u
module load conda/Miniforge3-25.9.1-0
module load cuda/12.8.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"
set -u

export CELLPOSE_LOCAL_MODELS_PATH="$PROJECT/cellpose_models"
cd "$REPO"

echo "node $(hostname) | $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "corpus $CORPUS"
[[ -d "$CORPUS" ]] || { echo "ERROR: corpus not found. scp it up first." >&2; exit 1; }

# ---- gate 1: does this build even take train_links? ------------------------
echo
echo "=== gate 1: train_links in the API? ==="
python - <<'PY'
import inspect, sys
from cellpose_omni import models
sig = inspect.signature(models.CellposeModel.train)
has = "train_links" in sig.parameters
print("cellpose_omni train() parameters:")
print("  ", ", ".join(sig.parameters))
print(f"\ntrain_links present: {has}")
if not has:
    print("\nFAIL: this build cannot accept links. A dense-corpus run would "
          "teach false splits at 59.5% of fibres. Stop here.", file=sys.stderr)
    sys.exit(1)
# Does the flow builder actually use them, or are they accepted and dropped?
try:
    from omnipose import core
    src = inspect.getsource(core.labels_to_flows)
    print("labels_to_flows references links:", "links" in src)
except Exception as exc:
    print("could not introspect omnipose.core.labels_to_flows:", exc)
PY

# ---- gate 2: 5 epochs end to end -------------------------------------------
echo
echo "=== gate 2: $EPOCHS epochs on the dense corpus ==="
srun python model_labs/omnipose_lab/train_fold.py \
  --corpus "$CORPUS" \
  --held-out "$HELD_OUT" \
  --epochs "$EPOCHS" \
  --window-px 1280 --overlap 0.25 \
  --dataloader --num-workers 6 \
  --out "$PROJECT/runs/smoke_links"

echo
echo "=== done. check above for: ==="
echo "  * 'links' count in the fold line (expect ~13,600)"
echo "  * seconds/epoch -> multiply by 100 for the real run estimate"
echo "  * peak GPU GB   -> must fit the rented card"
