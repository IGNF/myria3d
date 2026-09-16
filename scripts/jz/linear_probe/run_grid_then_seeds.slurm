#!/bin/bash

#SBATCH --output=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/%j.out
#SBATCH --error=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/%j.err
#SBATCH -A uhn@h100
#SBATCH -C h100

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread

#SBATCH --job-name=lp-grid-then-seeds
#SBATCH --time=14:00:00

# Chains the grid phase and the seed-ensemble phase in one job, requeue-safe: skips
# phase 1 if grid_search_results.json already exists, skips phase 2 if
# seed_ensemble_results.json already exists. Mirrors the requeue-safe pattern of
# Pointcept's sbatch_flair3d_lin_grid_then_seeds_h100.sh, adapted to myria3d's
# run.py/experiment= conventions.
#
# IMPORTANT: `@hydra.main` (run.py) changes the process's working directory to
# hydra.run.dir *before* running -- so a bare relative `output_path`/`results_path`
# in the callback configs is NOT relative to this script's $(pwd). Both phases below
# pin hydra.run.dir to a stable, dataset-specific path instead (same convention as
# run_grid.slurm / run_seeds.slurm) so this script's own `[ -f ... ]` checks and the
# python-side JSON writes agree on the same location.
#
# Usage:
#   sbatch run_grid_then_seeds.slurm <dales|h3d|eclair>
# Env overrides:
#   WEIGHT               path to the pretrained Flair3D+ multitask checkpoint (required)
#   DOWNSTREAM_DATA_ROOT root dir containing dales/h3d/eclair preprocessed subfolders
#   N_SEEDS               number of seed heads (default 10)
DATASET="${1:?Usage: sbatch run_grid_then_seeds.slurm <dales|h3d|eclair>}"
case "$DATASET" in
  dales|h3d|eclair) ;;
  *) echo "Unknown dataset '${DATASET}', expected dales|h3d|eclair" >&2; exit 1 ;;
esac
N_SEEDS="${N_SEEDS:-10}"

JOB_DIR=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}
ln -sf ../${SLURM_JOB_ID}.out ${JOB_DIR}/${SLURM_JOB_ID}.out
ln -sf ../${SLURM_JOB_ID}.err ${JOB_DIR}/${SLURM_JOB_ID}.err
cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Dataset: ${DATASET}"
    echo "N_SEEDS: ${N_SEEDS}"
    echo "Starting job at: $(date)"
    nvidia-smi
} > ${JOB_DIR}/job_info.log

module purge
module load arch/h100
module load cuda/12.1.0
module load miniforge/24.9.0

conda activate myria3d

set -euo pipefail

export LOGS_DIR=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs
export WANDB_MODE=offline
export FLAIR3D_CKPT_PATH="${WEIGHT:?set WEIGHT=/path/to/flair3d_plus_multitask.ckpt}"
export DOWNSTREAM_DATA_ROOT="${DOWNSTREAM_DATA_ROOT:-/lustre/fsn1/projects/rech/unv/usi32yh/data_pointcept}"
cd /lustre/fswork/projects/rech/unv/usi32yh/myria3d

GRID_DIR="${LOGS_DIR}/runs/linear_probe/${DATASET}/grid"
SEEDS_DIR="${LOGS_DIR}/runs/linear_probe/${DATASET}/seeds"
GRID_RESULTS="${GRID_DIR}/grid_search_results.json"
SEED_RESULTS="${SEEDS_DIR}/seed_ensemble_results.json"
mkdir -p "${GRID_DIR}" "${SEEDS_DIR}"

START_TIME=$(date +%s)

if [ -f "${GRID_RESULTS}" ]; then
    echo "Skipping grid phase: ${GRID_RESULTS} already exists."
else
    python run.py experiment=linear_probe/${DATASET}_grid \
      hydra.run.dir="${GRID_DIR}"
fi

if [ -f "${SEED_RESULTS}" ]; then
    echo "Skipping seed phase: ${SEED_RESULTS} already exists."
else
    WINNING_LR=$(python -c "import json; print(json.load(open('${GRID_RESULTS}'))['winner']['lr'])")
    PROBE_LRS="[$(python -c "print(','.join(['${WINNING_LR}'] * ${N_SEEDS}))")]"
    echo "Winning LR (from ${GRID_RESULTS}): ${WINNING_LR}"
    python run.py experiment=linear_probe/${DATASET}_seeds \
      model.probe_lrs="${PROBE_LRS}" \
      hydra.run.dir="${SEEDS_DIR}"
fi

END_TIME=$(date +%s)
{
    echo "Job finished at: $(date)"
    echo "Duration: $((END_TIME - START_TIME)) seconds"
    nvidia-smi
} >> ${JOB_DIR}/job_info.log
