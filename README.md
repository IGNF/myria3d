<div align="center">

# Aerial Lidar HD Semantic Segmentation with Deep Learning

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/ashleve/lightning-hydra-template)
</div>
<br><br>

## Description
### Context
The French Lidar HD project ambitions to map France in 3D using 10 pulse/m² aerial Lidar. The data will be openly available, including a semantic segmentation with a minimal number of classes: ground, vegetation, buildings, vehicles, bridges, others.

Here we train multiclass segmentation models that can serve as base model for further segmentation tasks on French Lidar HD data. 

The goal is to be somewhat data-agnostic yet opiniated, with default configuration for different national Lidar data specifications. 

### Content

This repository provides scripts tackles the following tasks:

- `train.py`: Training of the semantic segmentation neural network on aerial Lidar point clouds.
- `predict.py`: Applying model on unseen data.

Code is packaged for easy deployment (see below). Trained models are not public-hosted at the moment.

This Lidar Segmentation repository is heavily based on the following [template](https://github.com/ashleve/lightning-hydra-template). Please refer to its README for documentation on its general logic.

## How to use

### Setup virtual environment

```yaml
# clone project
git clone https://github.com/CharlesGaydon/Lidar-Deep-Segmentation
cd Lidar-Deep-Segmentation

# [OPTIONAL] If you want to use a gpu make sure cuda toolkit is installed
sudo apt install nvidia-cuda-toolkit

# install anaconda
# see https://www.anaconda.com/products/individual

# create conda environment - adapt versions and use of cudatoolkit to your own infrastructure.
source bash/setup_environment/setup_env.sh

# activate using
conda activate lidar_multiclass_env
```

### Run inference from package
If you are interested in running inference from anywhere, you can install code as a package in a your virtual environment.

```
# activate an env matching ./bash/setup_env.sh requirements.
conda activate lidar_multiclass_env

# install the package
pip install --upgrade https://github.com/IGNF/lidar-deep-segmentation/tarball/main  # from github directly
pip install -e .  # from local sources
```

To run inference, you will need:
- A source cloud point in LAS format on which to infer new classes. and probabilites.
- A checkpoint of a trained lightning module implementing model logic (class `lidar_multiclass.models.model.Model`)
- A minimal yaml configuration specifying parameters. We use [hydra](https://hydra.cc/) to manage configurations, and this yaml results from the model training. The `datamodule` and `model` parameters groups must match datset characteristics and model training settings.  The `predict` parameters group specifies path to models and data as well as batch size (N=50 works well, the larger the faster) and use of gpu (optionnal).

Fill out the {missing parameters} and run: 
```
python -m lidar_multiclass.predict --config-path {/path/to/.hydra} --config-name {config.yaml} predict.src_las={/path/to/cloud.las} predict.output_dir={/path/to/out/dir/} predict.resume_from_checkpoint={/path/to/checkpoint.ckpt} predict.gpus={0 for none, [i] to use GPU number i} datamodule.batch_size={N}
```

To show you current inference config, simply add a `--help` flag 

```
python -m lidar_multiclass.predict --config-path {/path/to/.hydra} --config-name {config.yaml} --help
```

TODO: add a control to where hydra log files are saved.

### Training new models

#### Setup
Some environment variable are injected at runtime and need to be specified in a `.env` file. Rename `.env_example` to `.env` and fill out: 
- `LOG PATH`, where hydra logs and config are saved.
- `DATAMODULE` section, which specify where to look for training data.
- `LOGGER` section, which specify credentials needed for logging to [comet.ml](comet.ml). Alternatively, logging can be disabled by setting `logger=null` ar runtime.

For training, input point clouds need to be splitted in chunks that can be digested by segmentation models. We found 50m\*50m to be a good balance between the model's receptive field and capacity. A specific preparation is needed that is described in section Data preparation

The expected file structure is summarized in `.env`.

A more detailed documentation on how to create a compatible, training-ready dataset from Swiss data is given in [this repo](https://github.com/CharlesGaydon/Colorize-SwissSURFACE3D-Lidar).

#### Training
Once you have data, define your experiment setting in an experiment file in the `configs/experiment` folder. 

To try out your setting by overfitting on a single batch of a Swiss dataset, run

```
python run.py experiment=RandLaNetDebug.yaml
```

After training, you model best checkpoints and hydra config will be saved in a `DATE/TIME/` subfolder of the `LOG_PATH` you specified, with an associated hydra `config.yaml`.
#### Run inference from sources

From the line for package-based inference above, simply change `python -m lidar_multiclass.predict` to `python run.py` to run directly from sources.

In case you want to swicth to package-based inference, you will need to comment out the parameters that depends on local environment variables such as logger credentials and training data directory. You can do so by making a copy of the `config.yaml` file and commenting out the lines containing `oc.env` logic.

### Data preparation

In `lidar_multiclass/datamodule/data.py` is the logic for data pre-processing, both offline and online, i.e. saving preprocessed data objects for fast trainig vs. pre-processing at inference time. 

The loading function is dataset dependant, and there are currently a logic for both SwissTopo data (withour infrared channel) and French IGN data (with infrared channel).

For help, run 

```
python lidar_multiclass/datamodules/data.py -h
```
Currently, two sources are supported:

- [French Lidar HD](https://geoservices.ign.fr/lidarhd), produced by the French geographical Institute. The data is colorized with both RGB and Infrared. Therefore, data processing will include Infrared channel as well as NDVI.
- Swiss Lidar from [SwissSurface3D (en)](https://www.swisstopo.admin.ch/en/geodata/height/surface3d.html), a similar initiative from the Swiss geographical institute SwissTopo. The data comes from the SwissSurface3D Lidar database and is not colorized, so we have to join it with SwissImage10 orthoimages database. The procedure is described in this standalone [repository](https://github.com/CharlesGaydon/Colorize-SwissSURFACE3D-Lidar).