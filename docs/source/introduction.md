Lidar-Deep-Segmentation is a deep learning library designed with a focused scope: the multiclass semantic segmentation of large scale, high density aerial Lidar points cloud.

The library implements the training of 3D Segmentation neural networks, with optimized data-processing and evaluation logics at fit time.
It allows for the evaluation of single-class IoU on the full point cloud, which results in reliable model evaluation.

Although the library can be easily extended with new neural network architectures or new data signatures, the library makes some opiniated choices. In particular, the RandLa-Net neural network architecture is the go-to choice for its reprensation power, conceptual simplicity, and speed. 
Additionnaly, two data signatures are supported:
- [French Lidar HD](https://geoservices.ign.fr/lidarhd), produced by the French geographical Institute. The data is colorized with both RGB and Infrared. Therefore, data processing will include Infrared channel as well as NDVI.
- Swiss Lidar from [SwissSurface3D (en)](https://www.swisstopo.admin.ch/en/geodata/height/surface3d.html), a similar initiative from the Swiss geographical institute SwissTopo. The data comes from the SwissSurface3D Lidar database and is not colorized, so we have to join it with SwissImage10 orthoimages database. The procedure is described in this standalone [repository](https://github.com/CharlesGaydon/Colorize-SwissSURFACE3D-Lidar).

Lidar-Deep-Segmentation is built upon [PyTorch](https://pytorch.org/). It keeps the standard data format 
from [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/). 
Its structure was bootstraped from [this code template](https://github.com/ashleve/lightning-hydra-template),
which heavily relies on [Hydra](https://hydra.cc/) and [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning) to 
enable flexible and rapid iterations of deep learning experiments.