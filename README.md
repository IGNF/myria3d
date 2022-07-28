<div align="center">

# Myria3D: Aerial Lidar HD Semantic Segmentation with Deep Learning


<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/ashleve/lightning-hydra-template)

[![CICD](https://github.com/IGNF/myria3d/actions/workflows/cicd.yaml/badge.svg)](https://github.com/IGNF/myria3d/actions/workflows/cicd.yaml)
[![Documentation Build](https://github.com/IGNF/myria3d/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/IGNF/myria3d/actions/workflows/gh-pages.yml)
</div>
<br><br>


Myria3D is a deep learning library designed with a focused scope: the multiclass semantic segmentation of large scale, high density aerial Lidar points cloud.

The library implements the training of 3D Segmentation neural networks, with optimized data-processing and evaluation logics at fit time. Inference on unseen, large scale point cloud is also supported.
It allows for the evaluation of single-class IoU on the full point cloud, which results in reliable model evaluation.

Myria3D is built upon [PyTorch](https://pytorch.org/). It keeps the standard data format 
from [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/). 
Its structure was bootstraped from [this code template](https://github.com/ashleve/lightning-hydra-template),
which heavily relies on [Hydra](https://hydra.cc/) and [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to enable flexible and rapid iterations of deep learning experiments.

Although the library can be extended with new neural network architectures or new data signatures, it makes some opiniated choices in terms of neural network architecture, data processing logics, and inference logic. Indeed, it is initially built with the [French "Lidar HD" project](https://geoservices.ign.fr/lidarhd) in mind, with the ambition to map France in 3D with 10 pulse/m² aerial Lidar by 2025. The data will be openly available, including a semantic segmentation with a minimal number of classes: ground, vegetation, buildings, vehicles, bridges, others. 

> &rarr; For installation and usage, please refer to [**Documentation**](https://ignf.github.io/myria3d/).

> &rarr; A stable, production-ready version of Myria3D is tracked by a [Production Release](https://github.com/IGNF/myria3d/releases/tag/prod-release-tag). In the release's assets are a trained multiclass segmentation model as well as the necessary configuration file to perform inference on French "Lidar HD" data. Those assets are provided for convenience, and are subject to change in time to reflect latest model training.