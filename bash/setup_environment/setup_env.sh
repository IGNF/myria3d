#!/bin/bash
set -e
conda install mamba -n base -c conda-forge
# some mamba specific issue occurs as in https://github.com/conda-incubator/conda-lock/issues/101
# mamba env create -f bash/setup_environment/requirements.yml
mamba env create -f bash/setup_environment/requirements.yml
conda activate validation_module_gpu

conda install -y pytorch==1.8.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia
mamba install -y pytorch-lightning==1.3 -c conda-forge 
export CUDA="cu111"
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-geometric
