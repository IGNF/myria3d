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

#SBATCH --job-name=lp-pureforest-extract
#SBATCH --time=04:00:00

# Extracts per-tile mean/max-pooled frozen-backbone hypercolumn features on
# PureForest (train/val/test), writing {split}.npz in Pointcept's own schema. Not a
# `python run.py` invocation: PureForest's actual linear probe is trained back in the
# sibling Pointcept repo on Hecate, via its `scripts/probe_pureforest_sklearn.py`
# (copy the output-dir's .npz files there, pass --channel-blocks 32 128 256 512
# explicitly). See readme_linear_probing.md.
#
# Usage:
#   sbatch run_extract_pureforest_features.slurm
# Env overrides:
#   WEIGHT               path to the pretrained Flair3D+ multitask checkpoint (required)
#   DOWNSTREAM_DATA_ROOT root dir containing the pureforest/ preprocessed subfolder
#   OUTPUT_DIR            where {split}.npz + meta.json are written
DOWNSTREAM_DATA_ROOT="${DOWNSTREAM_DATA_ROOT:-/lustre/fsn1/projects/rech/unv/usi32yh/data_pointcept}"
OUTPUT_DIR="${OUTPUT_DIR:-/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/pureforest_embeddings}"

JOB_DIR=/lustre/fswork/projects/rech/unv/usi32yh/myria3d/logs/slurm/${SLURM_JOB_ID}
mkdir -p ${JOB_DIR}
ln -sf ../${SLURM_JOB_ID}.out ${JOB_DIR}/${SLURM_JOB_ID}.out
ln -sf ../${SLURM_JOB_ID}.err ${JOB_DIR}/${SLURM_JOB_ID}.err
cp $0 ${JOB_DIR}/script.slurm

{
    echo "Job ID: $SLURM_JOB_ID"
    echo "Starting job at: $(date)"
    echo "weight: ${WEIGHT:-<unset>}"
    echo "output_dir: ${OUTPUT_DIR}"
    nvidia-smi
} > ${JOB_DIR}/job_info.log

module purge
module load arch/h100
module load cuda/12.1.0
module load miniforge/24.9.0
conda activate myria3d

set -euo pipefail

WEIGHT="${WEIGHT:?set WEIGHT=/path/to/flair3d_plus_multitask.ckpt}"
cd /lustre/fswork/projects/rech/unv/usi32yh/myria3d

START_TIME=$(date +%s)

python scripts/extract_pureforest_pooled_features.py \
  --ckpt-path "${WEIGHT}" \
  --data-root "${DOWNSTREAM_DATA_ROOT}/pureforest" \
  --output-dir "${OUTPUT_DIR}" \
  --splits train val test \
  --batch-size 8

END_TIME=$(date +%s)
{
    echo "Job finished at: $(date)"
    echo "Duration: $((END_TIME - START_TIME)) seconds"
    echo "output_dir: ${OUTPUT_DIR}"
    nvidia-smi
} >> ${JOB_DIR}/job_info.log
