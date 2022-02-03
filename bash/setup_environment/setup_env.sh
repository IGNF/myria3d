set -e

# SETUP
conda install -y mamba -n base -c conda-forge
mamba env create -f bash/setup_environment/requirements.yml
conda activate lidar_multiclass_env

# INSTALL
export PYTORCHVERSION="1.10.1"
export TORCHVISIONVERSION="0.11.2"
export TORCH_CUDA="11.3"
mamba install -y pytorch==$PYTORCHVERSION torchvision==$TORCHVISIONVERSION cudatoolkit=$TORCH_CUDA -c pytorch -c conda-forge
mamba install -y pyg==2.0.3 -c pytorch -c pyg -c conda-forge
FORCE_CUDA=1 pip install torch-points-kernels --no-cache
pip install numba==0.55.1 numpy==1.20.0 # revert inconsistent torch-points-kernel dependencies
mamba install pytorch-lightning==1.5.9 -c conda-forge