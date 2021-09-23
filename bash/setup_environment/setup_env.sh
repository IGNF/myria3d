#!/bin/bash
set -e
conda install mamba -n base -c conda-forge
mamba env create -f bash/setup_environment/requirements.yml
conda activate validation_module

mamba install pyg -c pyg -c conda-forge
pip install pytorch-lightning>=1.3.8
pip install torchvision>=0.9.1


# DEPRECATED
# mamba install pytorch==1.9.1+cu111 torchvision>=0.9.1 cudatoolkit=11.1 -c pytorch -c nvidia
# pip install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html  # if pytorch extensions are needed, use this torch install
# pip install pytorch-lightning
# pip install torchnet==0.0.4
# pip install -r bash/setup_environment/torch_extensions.txt --find-links https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
