# Performing inference on new data

Refer to [this tutorial](./setup_install.md) for how to setup a virtual environment and install the library.

To run inference, you will need:
- A source cloud point in LAS format on which to infer classes and probabilites. Sample data from the French "Lidar HD" project can be downloaded at [this address](https://geoservices.ign.fr/lidarhd).
- A checkpoint of a trained lightning module implementing model logic (class `myria3d.models.model.Model`)
- A minimal yaml configuration specifying parameters. We use [hydra](https://hydra.cc/) to manage configurations, and this yaml results from the model training. The `datamodule` and `model` parameters groups must match dataset characteristics and model training settings.  The `predict` parameters group specifies path to models and data as well as batch size (N=50 works well, the larger the faster) and use of gpu (optionnal). For hints on what to modify, see the `experiment/predict.yaml` file.

## Run inference from installed package

From the package root, run `pip install -e .` to install the package locally and freeze its current version.

Then, fill out the {missing parameters} below and run: 

```bash
python run.py \
task.task_name=predict \
predict.src_las={/path/to/cloud.las} \
predict.output_dir={/path/to/out/dir/} \
predict.gpus={0 for none, [i] to use GPU number i} \
datamodule.batch_size={N}
```

To show you current inference config, simply add a `--help` flag:

```bash
python run.py task.task_name=predict --help
```

Note that `predict.src_las` may be any valid glob pattern (e.g. `/path/to/multiple_files/*.las`), in order to **predict on multiple files successively**.

## Run inference from sources

In case you want to swicth to package-based inference, you will need to comment out the parameters that depends on local environment variables such as logger credentials and training data directory. You can do so by making a copy of your configuration file and commenting out the lines containing `oc.env` logic.

## Run inference from within a docker image

Up to date docker images (named `myria3d`) are created via Github integration actions (see [Developer's guide](../guides/development.md).

A docker image encapsulating the virtual environment and application sources can also be built using the provided Dockerfile. At built time, the Dockerfile is not standalone and should be part of the repository - whose content is copied into the image - at the github reference you want to build from.

To run inference: 
- Mount the needed volumes with the `-v` option.
- Always set `--ipc=host` to allow multithreading (used in pytorch dataloader, as mentionned in [Pytorch's README](https://github.com/pytorch/pytorch#using-pre-built-images)). 
- Increase the shared memory with `--shm-size=2gb` (which should be enough for 1km*1km point French "Lidar HD" clouds).
- Set `--gpus=all` to make gpus visible to the container if available.

See [docker-pytorch README](https://github.com/anibali/docker-pytorch#running-pytorch-scripts) for more details plus an additional option to specify user id at runtime.

```bash
# specify your paths here as needed
docker run \
-v {local_inputs}:/inputs/ \
-v {local_output}:/outputs/ \
--ipc=host \
--gpus=all \
--shm-size=2gb \
myria3d.predict {...config paths & options...}
```

## Additional options for prediction


### Output dimensions

By default, the predicted classification is stored in a new `PredictedClassification` LAS dimension. The [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) of probabilities is also stored and used as a limited proxy for prediction uncertainty.

One can control for which classes to save the probabilities. This is achieved with a `predict.probas_to_save` config parameter, which can be either the `all` keyword (to save probabilities for all classes) or a list of specific classes (e.g. `predict.probas_to_save=[building,vegetation]` - note the absence of space between class names).

### Receptive field overlap at inference time

To improve spatial regularity of the predicted probabilities, one can make inference on square receptive fields that have a non-null overlap with each other. This has the effect of smoothing out irregular predictions. The resulting classification is better looking, with more homogeneous predictions at the object level.

To define an overlap between successive 50m*50m receptive fields, set `predict.subtile_overlap={value}`.
This, however, comes with a large computation price. For instance, `predict.subtile_overlap=25` means a 25m overlap on both x and y axes, which multiplies inference time by a factor of 4.