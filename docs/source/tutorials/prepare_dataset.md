# Preparing data for training

## Using your own data 

### Data signatures

In `myria3d/data/loading.py` is the logic for data preprocessing, both offline and online, i.e. saving preprocessed data objects for fast trainig vs. pre-processing at inference time.

The loading function is dataset dependant, and current one supports French Lidar HD IGN data. Variations in logics may be due to the availability of specific dimensions (e.g. Infrared information) or encodings. The function returns a `pytorch_geometric.Data` object following the standard naming convention of `pytorch_geometric`, plus a description of features names and the path to the input file.

> Legacy: Detailed instruction to create a compatible dataset from Swiss data is given in [this repo](https://github.com/CharlesGaydon/Colorize-SwissSURFACE3D-Lidar).

You may want to implement your own logic, which then needs to be specified in `_get_data_preparation_parser` and `main` so that it can be used via the CLI to prepare a new dataset (see section below). The loading logic must additionnaly be referenced by the hydra config parameter `datamodule.dataset_description.load_las_func`. 

## Data preparation

See the argument for data preparation in :

```
python myria3d/data/loading.py -h
```

Input point clouds need to be splitted in subtiles that can be digested by segmentation models. We found that a receptive field of 50m\*50m was a good balance between context and memory intensity. 

For train and split sets, the subtiles turned into `Data` objects are serialized with torch so that they can be quickly loaded at training time.

For test, input point clouds are simply copied so that the test logic might mirror the logic used at inference time, including the on-the-fly splitting of these large input point clouds.

The prepared dataset hence have the following structure.
```
prepared_dataset
└───train
│   │   fileA_part1.data
│   │   ...
│   │   fileA_partN.data
│   │   ...
│   │   fileB_part1.data
│   │   ...
│   │   fileB_partN.data
│   │   ...
└───val
│   │   fileC_part1.data
│   │   ...
│   │   fileC_partN.data
│   │   ...
│   │   fileD_part1.data
│   │   ...
│   │   fileD_partN.data
└───val
│   │   fileX.las
│   │   fileY.las
│   │   ...
│   │   fileZ.las
```

## Getting started quickly with a toy dataset

A LAS file is provided as part of the test suite. It can be turned into a small, training-ready dataset to get started with the package. 
This single file is copied 6 times, so that there are 2 copies in each split (train/val/test). The copies are then prepared for training. The French Lidar data signature is used.

To create a toy dataset in `./inputs/toy_dataset/`, simply run :
```
python myria3d/data/loading.py --origin FR_TOY --prepared_data_dir=./inputs/toy_dataset/
```