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

#SBATCH --job-name=lp-seeds
#SBATCH --time=04:00:00

# Linear-probing seed-ensemble phase: N_SEEDS probe heads at the grid phase's winning
# LR (independent nn.Linear init via sequential construction order), trained and
# tested together in one job -- the same shared-backward multi-head trick as the grid
# phase, not N separate single-probe jobs (see myria3d/models/grid_probe_model.py and
# fact #1 in the linear-probing plan). Reads the winning LR from a prior
# run_grid.slurm's grid_search_results.json.
#
# Usage:
#   sbatch run_seeds.slurm <dales|h3d|eclair> [path/to/grid_search_results.json]
# Env overrides:
#   WEIGHT               path to the pretrained Flair3D+ multitask checkpoint (required)
#   DOWNSTREAM_DATA_ROOT root dir containing dales/h3d/eclair preprocessed subfolders
#   N_SEEDS               number of seed heads (default 10)
DATASET="${1:?Usage: sbatch run_seeds.slurm <dales|h3d|eclair> [grid_search_results.json]}"
case "$DATASET" in
  dales|h3d|eclair) ;;
  *) echo "Unknown dataset '${DATASET}', expected dales|h3d|eclair" >&2; exit 1 ;;
esac
LOGS_DIR_DEFAULT=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs
# Matches run_grid.slurm's stable RUN_DIR convention.
GRID_RESULTS="${2:-${LOGS_DIR_DEFAULT}/runs/linear_probe/${DATASET}/grid/grid_search_results.json}"
N_SEEDS="${N_SEEDS:-10}"

JOB_DIR=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}
ln -sf ../${SLURM_JOB_ID}.out ${JOB_DIR}/${SLURM_JOB_ID}.out
ln -sf ../${SLURM_JOB_ID}.err ${JOB_DIR}/${SLURM_JOB_ID}.err
cp $0 ${JOB_DIR}/script.slurm

module purge
module load arch/h100
module load cuda/12.1.0
module load miniforge/24.9.0

conda activate myria3d

set -euo pipefail

if [ ! -f "${GRID_RESULTS}" ]; then
    echo "Grid search results not found: ${GRID_RESULTS} (run run_grid.slurm first)" >&2
    exit 1
fi
WINNING_LR=$(python -c "import json; print(json.load(open('${GRID_RESULTS}'))['winner']['lr'])")
PROBE_LRS="[$(python -c "print(','.join(['${WINNING_LR}'] * ${N_SEEDS}))")]"

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Dataset: ${DATASET}"
    echo "Winning LR (from ${GRID_RESULTS}): ${WINNING_LR}"
    echo "N_SEEDS: ${N_SEEDS}"
    echo "Starting job at: $(date)"
    nvidia-smi
} > ${JOB_DIR}/job_info.log

export LOGS_DIR="${LOGS_DIR_DEFAULT}"
export WANDB_MODE=offline
export FLAIR3D_CKPT_PATH="${WEIGHT:?set WEIGHT=/path/to/flair3d_plus_multitask.ckpt}"
export DOWNSTREAM_DATA_ROOT="${DOWNSTREAM_DATA_ROOT:-/lustre/fsn1/projects/rech/unv/usi32yh/data_pointcept}"
cd /lustre/fswork/projects/rech/unv/usi32yh/myria3d

# Stable (non-timestamped) run dir -- mirrors run_grid.slurm's convention so
# seed_ensemble_results.json lands at a predictable path.
RUN_DIR="${LOGS_DIR}/runs/linear_probe/${DATASET}/seeds"
mkdir -p "${RUN_DIR}"

START_TIME=$(date +%s)

python run.py experiment=linear_probe/${DATASET}_seeds \
  model.probe_lrs="${PROBE_LRS}" \
  hydra.run.dir="${RUN_DIR}"

END_TIME=$(date +%s)
{
    echo "Job finished at: $(date)"
    echo "Duration: $((END_TIME - START_TIME)) seconds"
    nvidia-smi
} >> ${JOB_DIR}/job_info.log
