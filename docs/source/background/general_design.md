# General design of the package

Here are a few challenges relative to training 3D segmentation models for Aerial High Density Lidar, and the strategies we adopt in Myria3D in order to face them.


## Subsampling is important to improve point cloud structure

**Situation**:
- Point Cloud data, and aerial Lidar data, represent rich data with a level of detail that might hinder the detection of  objects with generally simple structures such as buildings, ground, and trees. On the other hand, smaller, more variate objects might benefit from denser point clouds.
- Another point to consider is that some 3D semantic segmentation architectures - including the RandLa-Net architecture we leverage - need fixed size point cloud. This means that either subsampling or padding are required. This kind of implementation reduces flexibility and is suboptimal, but to our knowledge there are no alternative RandLa-Net implementation in pytorch that can accept different-size point clouds within the same batch. 

**Strategy**:
- We leverage torch_geometric [GridSampling](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.GridSampling) and [FixedPoints](https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html#torch_geometric.transforms.FixedPoints) to (i) simplify local point structures with a 0.25m resolution, and (ii) get a fixed size point cloud that can be fed to the mmodel. Grid Sampling has the effect of reducing point cloud size by around a third, with most reductions expected to occur in vegetation.

## Speed is of the essence
**Situation**:

- Short training time allow for faster iterations, more frequent feedback on architecture design, less time spent on doomed solutions. As a result, we want to be parcimonious in terms of the operation performed during a train forward pass of a model.

**Strategy**:
- During training and validation phases, we perform supervision and back-propagation on the subsampled point cloud directly. Our hypothesis is that the gain we may expect from interpolating predicted logits to the full point cloud (usually from N'=~12500 to N~30000 on a average for a 50mx50m sample) is tiny compared to the computational cost of such operation (time of a forward pass multiplied from x5 to x10 with a batch size of 32 on CPU).


## Evaluation is key to select the right approach

**Situation**:

- Evaluation of models must be reliable in order to compare solutions. For semantic segmentation models on point cloud, this means that performance metrics (i.e. mean and by-class Intersection-over-Union) should be computed based on a confusion matrix that is computed from all points in all point clouds in the test dataset.

**Strategy**:
- During test phase, we **do** interpolate logits back to the each sample (point cloud) before computing performance metrics. Interestingly, this enable to compare different subsampling approaches and interpolation methods in a robust way. The interpolation step is triggered in `eval` mode only, and is also leveraged during inference.
- This differentiated approach between `train` and `eval` modes has the drawback of requiring full (non-subsampled) positions and targets as well as subsampled, non-normalized positions to be copied and saved at data preparation time to allow for the interpolation. 
