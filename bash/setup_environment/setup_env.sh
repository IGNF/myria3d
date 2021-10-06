#!/bin/bash
set -e
conda install mamba -n base -c conda-forge
mamba env create -f bash/setup_environment/requirements.yml
conda activate validation_module

pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html  # if pytorch extensions are needed, use this torch install
mamba install pyg==2.0.1==py39_torch_1.8.0_cu111 -c pyg -c conda-forge
pip install torchvision==0.9.0
pip install pytorch-lightning==1.3