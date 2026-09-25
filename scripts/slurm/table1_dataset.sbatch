#!/usr/bin/env bash
# One Table 1 dataset across the fleet in a single 4-GPU allocation.
#   sbatch scripts/slurm/table1_dataset.sbatch insecure_code --n 10 --controls
# Extra arguments go to scripts/table1_dataset.py (see its --help).
#SBATCH --job-name=table1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=36:00:00
#SBATCH --output=%x-%j.out
set -euo pipefail
DATASET=$1; shift
cd "${EMERGE_REPO:-$(dirname "$(realpath "$0")")/../..}"
export EMERGE_ROOT=${EMERGE_ROOT:?set EMERGE_ROOT to the results root}
export HF_HUB_CACHE=${HF_HUB_CACHE:?set HF_HUB_CACHE to a cache the account can write}
export TOKENIZERS_PARALLELISM=false
echo "job $SLURM_JOB_ID dataset=$DATASET cards=${CUDA_VISIBLE_DEVICES:-unset} root=$EMERGE_ROOT"
uv run --no-sync python scripts/table1_dataset.py "$DATASET" "$@"
