#!/bin/bash
set -e
conda install mamba -n base -c conda-forge
mamba env create -f bash/setup_environment/requirements.yml
conda activate validation_module

# pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html  # if pytorch extensions are needed, use this torch install
# pip install pytorch-lightning
# pip install torchnet==0.0.4
# pip install -r setup_environment/torch_extensions.txt --find-links https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
