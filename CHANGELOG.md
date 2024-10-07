# CHANGELOG

- Fix lidar_hd pre-transform to allow full 16-bits integer range for color/infra-red values
- Add a github action workflow to run a trained model on the lidar-prod thresholds optimisation dataset
(in order to automate thresholds optimization)

### 3.8.4
- fix: move IoU appropriately to fix wrong device error created by a breaking change in torch when using DDP.

### 3.8.3
- fix: prepare_data_per_node is a flag and was incorrectly used as a replacement for prepare_data.

### 3.8.2
- fix: points not dropped case in subsampling when the subtile contains only one point
- fix: type error in edge case when dropping points in DropPointsByClass (when there is only one remaining point)

### 3.8.1
- fix: propagate input las format to output las (in particular epsg which comes either from input or config)

## 3.8.0
- dev: log confusion matrices to Comet after each epoch.
- fix: do not mix the two way to log IoUs to avoid known lightning [Common Pitfalls](https://lightning.ai/docs/torchmetrics/stable/pages/lightning.html#common-pitfalls).

### 3.7.1
- fix: edge case when saving predictions under Classification channel, without saving entropy.

## 3.7.0
- Update all versions of Pytorch, Pytorch Lightning, and Pytorch Geometric.
  Changes are retrocompatible for models trained with older versions (with adjustment to the configuration file).
- Refactor logging of single-class IoUs to go from num_classes+1 torchmetrics instances to only 1.

### 3.6.1
- Set urllib3<2 for comet logging to function and add back seaborn for plotting optimal LR graph.

## 3.6.0
- Remove the "EPSG:2154" by default and use the metadata of the lidar file, unless a parameter is given.

### 3.5.2
- Track ./tests/data/ dir including single-point-cloud.laz.

### 3.5.1
- Run CICD operations for all branches prefixed with "staging-".

## 3.5.0
- Abandon of option to get circular patches since it was never used.

### 3.4.12
- Remove COPC datasets and dataloaders since they were abandonned and never used.

### 3.4.11
- Unification of max length of lines (99) by applying black everywhere.

### 3.4.10
- Migrate from setup.cfg to pyproject.toml and .flake8.

### 3.4.9
- Support edge-case where source LAZ has no valid subtile (i.e. pre_filter=False for all candidate subtiles) during hdf5 creation

### 3.4.8
- Raise an informative error in case of unexpected task_name

### 3.4.7
- Remove tqdm when splitting a lidar tile to avoid cluttered logs during data preparation

### 3.4.6
- Document the possible use of ign-pdal-tools for colorization

### 3.4.5
- Set a default task_name (fit) to avoid common error at lauch time

### 3.4.4
- Remove duplicated experiment configuration

### 3.4.3
- Remove outdated and incorrect hydra parameter in config.yaml

### 3.4.2
- Reconstruct absolute path of input LAS files explicitely, removing a costly glob operation

### 3.4.1
- Fix dataset description for pacasam: there was an unwanted int-to-int mapping in classification_dict

## 3.4.0
- Allow inference for the smallest possible patches (num_nodes=1) to have consistent inference behavior
