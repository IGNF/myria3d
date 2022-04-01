# How to train new models

Refer to [this tutorial](../tutorials/setup_install.md) for how to setup a virtual environment and install the library.
Refer to [this other tutorial](../tutorials/prepare_dataset.md) for how to prepare a dataset.


## Setup

Some environment variable are injected at runtime and need to be specified in a `.env` file. Rename `.env_example` to `.env` and fill out: 
- `LOG PATH`, where hydra logs and config are saved.
- `DATAMODULE` section, which specify where to look for training data.
- `LOGGER` section, which specify credentials needed for logging to [comet.ml](https://www.comet.ml/). Alternatively, logging can be disabled by setting `logger=null` ar runtime.

For training, input point clouds need to be splitted in chunks that can be digested by segmentation models. We found 50m\*50m to be a good balance between the model's receptive field and capacity. A specific preparation is needed that is described in section Data preparation

The expected file structure is summarized in `.env`.

A more detailed documentation on how to create a compatible, training-ready dataset from Swiss data is given in [this repo](https://github.com/CharlesGaydon/Colorize-SwissSURFACE3D-Lidar).

## Training
Once you have data, define your experiment setting in an experiment file in the `configs/experiment` folder. 

To try out your setting by overfitting on a single batch of a Swiss dataset, run

```
python run.py experiment=RandLaNetDebug.yaml
```

After training, you model best checkpoints and hydra config will be saved in a `DATE/TIME/` subfolder of the `LOG_PATH` you specified, with an associated hydra `config.yaml`.
