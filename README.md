---
layout: default
---

<div align="center">

# Aerial Lidar HD Semantic Segmentation with Deep Learning

![Documentation Build](https://github.com/github/docs/actions/workflows/gh_pages.yml/badge.svg)


<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/ashleve/lightning-hydra-template)
</div>
<br><br>


## Context
The French Lidar HD project ambitions to map France in 3D using 10 pulse/m² aerial Lidar. The data will be openly available, including a semantic segmentation with a minimal number of classes: ground, vegetation, buildings, vehicles, bridges, others.

Here we train multiclass segmentation models that can serve as base model for further segmentation tasks on French Lidar HD data. 

The goal is to be somewhat data-agnostic yet opiniated, with default configuration for different national Lidar data specifications. 

{% include_relative docs/source/introduction.md %}

## Content

This repository tackles the following tasks:

- `train.py`: Training of the semantic segmentation neural network on aerial Lidar point clouds.
- `predict.py`: Applying model on unseen data.

Code is packaged for easy deployment (see below). Trained models are not public-hosted at the moment.

This Lidar Segmentation repository is heavily based on the following [template](https://github.com/ashleve/lightning-hydra-template). 

Please refer to **documentation**.



Currently, two sources are supported:

- [French Lidar HD](https://geoservices.ign.fr/lidarhd), produced by the French geographical Institute. The data is colorized with both RGB and Infrared. Therefore, data processing will include Infrared channel as well as NDVI.
- Swiss Lidar from [SwissSurface3D (en)](https://www.swisstopo.admin.ch/en/geodata/height/surface3d.html), a similar initiative from the Swiss geographical institute SwissTopo. The data comes from the SwissSurface3D Lidar database and is not colorized, so we have to join it with SwissImage10 orthoimages database. The procedure is described in this standalone [repository](https://github.com/CharlesGaydon/Colorize-SwissSURFACE3D-Lidar).
