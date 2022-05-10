# Performing inference on new data

Refer to [this tutorial](./setup_install.md) for how to setup a virtual environment and install the library.

To run inference, you will need:
- A source cloud point in LAS format on which to infer classes and probabilites.
- A checkpoint of a trained lightning module implementing model logic (class `lidar_multiclass.models.model.Model`)
- A minimal yaml configuration specifying parameters. We use [hydra](https://hydra.cc/) to manage configurations, and this yaml results from the model training. The `datamodule` and `model` parameters groups must match datset characteristics and model training settings.  The `predict` parameters group specifies path to models and data as well as batch size (N=50 works well, the larger the faster) and use of gpu (optionnal).

## Run inference from installed package

Fill out the {missing parameters} below and run: 

```bash
python -m lidar_multiclass.predict \
--config-path {/path/to/.hydra} \
--config-name {config.yaml} \
predict.src_las={/path/to/cloud.las} \
predict.output_dir={/path/to/out/dir/} \
predict.resume_from_checkpoint={/path/to/checkpoint.ckpt} \
predict.gpus={0 for none, [i] to use GPU number i} \
datamodule.batch_size={N} \
hydra.run.dir={path/for/hydra/logs}
```

To show you current inference config, simply add a `--help` flag:

```bash
python -m lidar_multiclass.predict --config-path {/path/to/.hydra} --config-name {config.yaml} --help
```

Note that `predict.src_las` may be any valid glob pattern (e.g. `/path/to/multiple_files/*.las`), in order to **predict on multiple files successively**.

## Run inference from sources

From the line for package-based inference above, simply change `python -m lidar_multiclass.predict` to `python run.py` to run directly from sources.

In case you want to swicth to package-based inference, you will need to comment out the parameters that depends on local environment variables such as logger credentials and training data directory. You can do so by making a copy of the `config.yaml` file and commenting out the lines containing `oc.env` logic.

## Run inference from within a docker image

Up to date docker images (named `lidar_deep_im`) are created via Github integration actions (see [Developer's guide](../guides/development.md).

A docker image encapsulating the virtual environment and application sources can also be built using the provided Dockerfile. This Dockerfile is not standalone and should be part of the repository (whose content is copied into the image), on the github reference you want to build from.

To run, mount needed volumes and follow above syntaxes:
```bash
# specify your paths here as needed
docker run -v {local_inputs}:/inputs/ -v {local_output}:/outputs/ lidar_deep_im {...}
```