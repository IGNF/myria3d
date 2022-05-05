set -e

# SETUP
# WARNING: you may need to adapt cudatoolkit version to your needs.
# Or remove it altogether in case you do not need CUDA support at all

# Install mamba to create the environment faster
conda install -y mamba -n base -c conda-forge
# Build it
mamba env create -f setup_env/requirements.yml
# activate it
conda activate lidar_multiclass