# Preparing data for training

### Peprocessing functions

The loading function is dataset dependant, and is `lidar_hd_pre_transform` by default. The function takes points loaded from a LAS file via pdal as input, and returns a `pytorch_geometric.Data` object following the standard naming convention of `pytorch_geometric`, plus a list of features names for later use in transforms.

It is adapted to the French Lidar HD data provided by IGN (see [the official page](https://geoservices.ign.fr/lidarhd) - link in French). Return number and color information (RGBI) are scaled to 0-1 interval, a NDVI and an average color ((R+G+B)/3) dimension are created, and points that may be occluded (as indicated by higher return number) have their color set to 0.

You may want to implement your own logic (e.g. with custom, additional features) in diretcory `points_pre_transform`. It then needs to be referenced similarly to `lidar_hd_pre_transform`.


### Using your own data

Input point clouds need to be splitted in subtiles that can be digested by segmentation models. We found that a receptive field of 50m*50m was a good balance between context and memory intensity. For faster training, this split can be done once, to avoid loading large file in memory multiple times.

To perform a training, you will need to specify these parameters of the datamodule config group:
- `data_dir`: path to a directory in which a set of LAS files are stored (can be nested in subdirectories).
- `split_csv_path`: path to a CSV file with schema `basename,split`, specifying a train/val/test spit for your data.

These will be composed into a single file dataset for which you can specify a path via the `datamodule.hdf5_file_path` parameter. This happens on the fly, therefore a first training might take some time, but this should only happens once.

Once this is done, you do not need sources anymore, and simply specifying the path to the HDF5 dataset is enough.


## Getting started quickly with a toy dataset

A LAS file is provided as part of the test suite. It can be turned into a small, training-ready dataset to get started with the package. 

To create a toy dataset run :
```
python myria3d/pctl/dataset/toy_dataset.py
```

You will see a new file: `/test/data/toy_dataset.hdf5`.