<div align="center">

# Lidar Deep Segmentation


<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/ashleve/lightning-hydra-template)

[![Documentation Build](https://github.com/IGNF/lidar-deep-segmentation/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/IGNF/lidar-deep-segmentation/actions/workflows/gh-pages.yml)
</div>
<br><br>

> *Aerial Lidar HD Semantic Segmentation with Deep Learning*

## Context
The French Lidar HD project ambitions to map France in 3D using 10 pulse/m² aerial Lidar. The data will be openly available, including a semantic segmentation with a minimal number of classes: ground, vegetation, buildings, vehicles, bridges, others.

Considering the scale of this task, deep learning is leveraged to speed-up the production and give tools for quality control

## Content

Lidar-Deep-Segmentation is a deep learning library designed with a focused scope: the multiclass semantic segmentation of large scale, high density aerial Lidar points cloud.

The library implements the training of 3D Segmentation neural networks, with optimized data-processing and evaluation logics at fit time. Inference on unseen, large scale point cloud is also supported.
It allows for the evaluation of single-class IoU on the full point cloud, which results in reliable model evaluation.

Although the library can be easily extended with new neural network architectures or new data signatures, the library makes some opiniated choices in terms of neural network architecture, data processing logics, and inference logic.

Additionnaly, two data signatures are supported:
- [French Lidar HD](https://geoservices.ign.fr/lidarhd), produced by the French geographical Institute. The data is colorized with both RGB and Infrared. Therefore, data processing will include Infrared channel as well as NDVI.
- Swiss Lidar from [SwissSurface3D (en)](https://www.swisstopo.admin.ch/en/geodata/height/surface3d.html), a similar initiative from the Swiss geographical institute SwissTopo. The data comes from the SwissSurface3D Lidar database and is not colorized, so we have to join it with SwissImage10 orthoimages database. The procedure is described in this standalone [repository](https://github.com/CharlesGaydon/Colorize-SwissSURFACE3D-Lidar).

Lidar-Deep-Segmentation is built upon [PyTorch](https://pytorch.org/). It keeps the standard data format 
from [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/). 
Its structure was bootstraped from [this code template](https://github.com/ashleve/lightning-hydra-template),
which heavily relies on [Hydra](https://hydra.cc/) and [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to 
enable flexible and rapid iterations of deep learning experiments.

> -> For installation and usage, please refer to full [**Documentation**](https://ignf.github.io/lidar-deep-segmentation/).
