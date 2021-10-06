#!/bin/bash
set -e
conda install mamba -n base -c conda-forge
# some mamba specific issue occurs as in https://github.com/conda-incubator/conda-lock/issues/101
# mamba env create -f bash/setup_environment/requirements.yml  
conda env create -f bash/setup_environment/requirements.yml
conda activate validation_module

export CUDA="111"
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+${CUDA}.html
pip install torch-geometric