# General design of the package

Here are a few challenges relative to training 3D segmentation models for Aerial High Density Lidar, and the strategies we adopt in Myria3D in order to face them.

## Model should be fast, performant, and practical

**Situation**:
- Since the seminal PointCloud architecture [[Qi 2016](https://arxiv.org/abs/1612.00593)] and the later [[PointNet++](https://arxiv.org/abs/1706.02413)], there were many attempts to improve these architecture which operate direcly on point clouds in a hierarchical fashion. 
- Our main requirements are:
  1) Speed of inference, in order to operate at a national scale.
  2) performances in large-scale outdoor Lidar settings, on e.g. [SemanticKITTI](http://semantic-kitti.org/) and [Semantic 3D](http://semantic3d.net/) benchmarks, by opposition to e.g. the [S3DIS](https://ieeexplore.ieee.org/document/7780539/) benchmark.
  3) performances on specific, urban classes such as buildings.
  4) Conceptual simplicity - following a simple hierarchical structure was preferred to more complex architectures.

**Strategy**:
- We focused on the RandLA-Net architecture [Hu 2020](https://arxiv.org/abs/1911.11236). 
    - It was specifically conceived by authors with requirements (1) and (2) in mind. 
    - It meets requirement (2), by setting SOTA results on both Semantic3D and SemanticKITTI datasets. In particular, it takes the first place from the KPConv architecture by [Thomas et al. 2019](https://arxiv.org/abs/1904.08889) on Semantic3D, and is in particular superior on class `buildings`, meeting requirement (3). 
    - The official RandLA-Net implementation is ~4 times faster than KPConv on SemanticKITTI, meeting requirement (1).
    - It is conceptually simple, basically a PointNet++ like encoder-decoder architecture, with random samplings and explicit encoding of local geometric structures.


## Subsampling is important to improve point cloud structure

**Situation**:
- Point Cloud data, and aerial Lidar data, represent rich data with a level of detail that might hinder the detection of objects with generally simple structures such as buildings and ground. On the other hand, smaller, more intricate objects might benefit from denser point clouds.

- Until V3.0.*, we experimented with a RandLA-Net architecture implemented in pytorch by [aRI0U](https://github.com/aRI0U/RandLA-Net-pytorch/), which needed fixed size point cloud. Subsampling was thus required. This kind of implementation reduces flexibility and is suboptimal. There was no alternative RandLa-Net implementation in pytorch that can accept different-size point clouds within the same batch.

**Strategy**:
- We leverage torch_geometric [GridSampling](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.GridSampling) and [FixedPoints](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.FixedPoints) to (i) simplify local point structures with a 0.25m resolution, and (ii) get a fixed size point cloud that can be fed to the mmodel. Grid Sampling has the effect of reducing point cloud size by around a third, with most reductions expected to occur in vegetation.

**Next Steps**:
- Starting with V3 we implement our own version of RandLA-Net. Using the capabilities of pytorch-geometric, it accepts variable size point clouds, and also follows the authors paper more closely. We also contribute it to the pytorch-geometric community (see this [pull request](https://github.com/pyg-team/pytorch_geometric/pull/5117)) and benefit from their feedback. We are in the process of specifying a configuration and transforms to use fuller point clouds, taking into account memory limitations. If this works, this will become the defaut.

## Speed is of the essence
**Situation**:

- Short training time allow for faster iterations, more frequent feedback on architecture design, less time spent on doomed solutions. As a result, we want to be parcimonious in terms of the operation performed during a train forward pass of a model.

**Strategy**:
- During training we perform supervision and back-propagation on the subsampled point cloud directly. Our hypothesis is that the gain we may expect from interpolating predicted logits to the full point cloud (usually from N'=~12500 to N~30000 on a average for a 50mx50m sample) is tiny compared to the computational cost of such operation (time of a forward pass multiplied from x5 to x10 with a batch size of 32 on CPU).


## Evaluation is key to select the right approach

**Situation**:

- Evaluation of models must be reliable in order to compare solutions. For semantic segmentation models on point cloud, this means that performance metrics (i.e. mean and by-class Intersection-over-Union) should be computed based on a confusion matrix that is computed from all points in all point clouds in the test dataset.

**Strategy**:
- During test and validation phases, we **do** interpolate logits back to the each sample (point cloud) before computing performance metrics. Interestingly, this enable to compare different subsampling approaches and interpolation methods in a robust way. The interpolation step is triggered in `eval` mode only, and is of course also leveraged during inference.