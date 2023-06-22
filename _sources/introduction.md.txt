Myria3D is a deep learning library designed with a focused scope: the multiclass semantic segmentation of large scale, high density aerial Lidar points cloud.

The library implements the training of 3D Segmentation neural networks, with optimized data-processing and evaluation logics at fit time. Inference on unseen, large scale point cloud is also supported.
It allows for the evaluation of single-class IoU on the full point cloud, which results in reliable model evaluation.

Although the library can be easily extended with new neural network architectures or new data signatures, the library makes some opiniated choices in terms of neural network architecture, data processing logics, and inference logic.

Myria3D is built upon [PyTorch](https://pytorch.org/). It keeps the standard data format 
from [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/). 
Its structure was bootstraped from [this code template](https://github.com/ashleve/lightning-hydra-template),
which heavily relies on [Hydra](https://hydra.cc/) and [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to 
enable flexible and rapid iterations of deep learning experiments.