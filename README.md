<div align="center">

# [Work in Progress] Lidar HD Semantic Segmentation with Deep Learning

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/ashleve/lightning-hydra-template)
</div>
<br><br>

## Description
### Context
The Lidar HD project ambitions to map France in 3D using 10 pulse/m² aerial Lidar. The data will be openly available, including a semantic segmentation with a minimal number of classes: ground, vegetation, buildings, vehicles, bridges, others.

Here we train multiclass segmentation models that can serve as base model for further segmentation tasks on French Lidar HD data. 

The goal is to be somewhat data-agnostic yet opiniated, with default configuration for different national Lidar data specifications. 

To kickstart these training, we will use data from the [SwissSurface3D](https://www.swisstopo.admin.ch/fr/geodata/height/surface3d.htm), a similar initiative from the Swiss geographical institute SwissTopo. Once labeled French Lidar HD data becomes available, we will switch to different datasets.

### Content

In this repository you will find two main components:

- `train.py`: Training and evaluation of the semantic segmentation neural network.
- `predict.py`: Applying model on unseen data.

### Process

...

## How to run

### Install dependencies

```yaml
# clone project
git clone https://github.com/CharlesGaydon/Lidar-Deep-Segmentation
cd Lidar-Deep-Segmentation

# [OPTIONAL] If you want to use a gpu make sure cuda toolkit is installed
sudo apt install nvidia-cuda-toolkit

# install conda
# see https://www.anaconda.com/products/individual

# create conda environment (you may need to run lines manually as conda may not activate properly from bash script)
source bash/setup_environment/setup_env.sh

# activate using
conda activate lidar_multiclass_env
```

Rename `.env_example` to `.env` and fill out the needed variables for looging and data directories.

### Install as a package
Once you have the environement setup, you can install as a package in the environment, to deploy inference in other codes.

```
# activate using
conda activate lidar_multiclass_env

# install from local source using 
pip install -e .
```

Then, simply copy and paste the `run.py` script and make inferences as usual.


In the future, conda packages should be supported instead of this basic pip env.
