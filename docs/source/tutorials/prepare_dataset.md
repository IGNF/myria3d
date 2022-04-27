# Preparing data for training [WIP]

.. warning:: Work in progress


In `lidar_multiclass/datamodule/data.py` is the logic for data pre-processing, both offline and online, i.e. saving preprocessed data objects for fast trainig vs. pre-processing at inference time. 

The loading function is dataset dependant, and there are currently a logic for both SwissTopo data (withour infrared channel) and French IGN data (with infrared channel).

For help, run 

```
python lidar_multiclass/data/loading.py -h
```

## Using a toy dataset

Files used in unit tests can be turned into a small, training-ready dataset to get started with the package.
A single file is copied 6 times, so that there are 2 copies in each split (train/val/test). Data is then prepared for training. The French Lidar data signature is used.

To create a toy dataset in ./inputs/toy_dataset/, simply run :
```
python lidar_multiclass/data/loading.py --origin FR_TOY  --prepared_data_dir=./inputs/toy_dataset/
```