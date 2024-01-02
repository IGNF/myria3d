# Preparing data for training

## Peprocessing functions

The loading function is dataset dependant, and is `lidar_hd_pre_transform` by default. The function takes points loaded from a LAS file via pdal as input, and returns a `pytorch_geometric.Data` object following the standard naming convention of `pytorch_geometric`, plus a list of features names for later use in transforms. In the loading function, the return number and color information (RGBI) are scaled to 0-1 interval, a NDVI and an average color ((R+G+B)/3) dimension are created, and points that may be occluded (as indicated by higher return number) have their color set to 0.

Customization: You may want to implement your own logic (e.g. with custom, additional features) in directory `points_pre_transform`. It then needs to be referenced similarly to `lidar_hd_pre_transform`. 

The loading function is designed for the French Lidar HD data provided by IGN (see [the official page](https://geoservices.ign.fr/lidarhd) - link in French). Note that the clouds are shared without color information, and should be colorized (RGB+Infrared) to use myria3d. The [open-source ign-pdal-tools library](https://pypi.org/project/ign-pdal-tools/) is a convenient toolkit that can be used to colorize the raw clouds with IGN aerial imagery (see function 'pdaltools.color.color(...)').

Customization: If you use a different classification (e.g. additional classes), you will need to create a `dataset_description` configuration (similar to `configs/dataset_description/20220607_151_dalles_proto.yaml`).

Additionnaly, you can control cloud sampling parameters via two configurations:
- `configs/datamodule/transforms/preparations/points_budget.yaml`: (defaut) allows variable cloud size within lower and higher boundaries. 
- `configs/datamodule/transforms/preparations/fixed_num_points.yaml`: (alternative) samples all clouds to a fixed size, allowing for duplicated points.


## Preparing the dataset

To perform a training, you will need to specify these parameters in the datamodule config group:
- `data_dir`: path to a directory in which a set of LAS files are stored. Clouds must be nested in subdirectories named according to their spli: train, val, or test.
- `split_csv_path`: path to a CSV file with schema `basename,split`, specifying a train/val/test spit for your data.

Under the hood, the path of each LAS file will be reconstructed like this: '{data_dir}/{split}/{basename}'.

Large input point clouds need to be divided in smaller clouds that can be digested by segmentation models. We found that a receptive field of 50m x 50m was a good balance between context and memory intensity. The division is performed once, to avoid loading large file in memory multiple times during training.

To be able to read the lidar files, an EPSG is needed. If the files don't all specify an EPSG in their metadata, it should be given as a parameter with `datamodule.epsg=...` 

After division, the smaller clouds are preprocessed (i.e. selection of specific LAS dimensions, on-the-fly creation of dimensions) and regrouped into a single HDF5 file whose path is specified via the `datamodule.hdf5_file_path` parameter. 

The HDF5 dataset is created at training time. It should only happens once. Once this is done, you do not need sources anymore, and simply specifying the path to the HDF5 dataset is enough (there is no need for data_dir or split_csv_path parameters anymore).

It's also possible to create the hdf5 file without training any model: just fill the `datamodule.hdf5_file_path` parameter as before to specify the file path, but use `task=create_hdf5` instead of `task=fit`.


## Getting started quickly with a toy dataset

A LAS file is provided as part of the test suite. It can be turned into a small, training-ready dataset to get started with the package. 

To create a toy dataset run :
```
python myria3d/pctl/dataset/toy_dataset.py
```

You will see a new file: `/test/data/toy_dataset.hdf5`.