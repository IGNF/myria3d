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

#SBATCH --job-name=lp-grid
#SBATCH --time=10:00:00

# Grid phase only: sweep the 12-value LR grid, write grid_search_results.json.
# See readme_linear_probing.md and run_grid_then_seeds.slurm for the full protocol.
#
# Usage:
#   sbatch run_grid.slurm <dataset>          # dataset in {dales, h3d, eclair}
#   sbatch run_grid.slurm dales
#
# Env var overrides (export before sbatch, or edit here):
#   WEIGHT               path to the Flair3D+ multitask checkpoint being probed
#   DOWNSTREAM_DATA_ROOT parent dir of dales/h3d/eclair Pointcept-preprocessed data
DATASET="${1:?Usage: sbatch run_grid.slurm <dales|h3d|eclair>}"
WEIGHT="${WEIGHT:?Set WEIGHT=/path/to/flair3d_plus_multitask.ckpt (export before sbatch)}"
DOWNSTREAM_DATA_ROOT="${DOWNSTREAM_DATA_ROOT:-/lustre/fsn1/projects/rech/unv/usi32yh/data_pointcept}"

JOB_DIR=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}
ln -sf ../${SLURM_JOB_ID}.out ${JOB_DIR}/${SLURM_JOB_ID}.out
ln -sf ../${SLURM_JOB_ID}.err ${JOB_DIR}/${SLURM_JOB_ID}.err
cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Starting job at: $(date)"
    echo "Running on host: $(hostname)"
    echo "dataset: ${DATASET}"
    echo "weight: ${WEIGHT}"
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
export FLAIR3D_CKPT_PATH="${WEIGHT}"
export DOWNSTREAM_DATA_ROOT="${DOWNSTREAM_DATA_ROOT}"
cd /lustre/fswork/projects/rech/unv/usi32yh/myria3d

# Stable (non-timestamped) run dir: run_grid_then_seeds.slurm and run_seeds.slurm
# need to find grid_search_results.json deterministically across separate `python
# run.py` invocations, which Hydra's default ${now:...} run dir doesn't allow.
RUN_DIR="${LOGS_DIR}/runs/linear_probe/${DATASET}/grid"
mkdir -p "${RUN_DIR}"

START_TIME=$(date +%s)

python run.py experiment=linear_probe/${DATASET}_grid \
  hydra.run.dir="${RUN_DIR}"

END_TIME=$(date +%s)
{
    echo "Job finished at: $(date)"
    echo "Duration: $((END_TIME - START_TIME)) seconds"
    echo "grid_search_results.json: ${RUN_DIR}/grid_search_results.json"
    nvidia-smi
} >> ${JOB_DIR}/job_info.log
