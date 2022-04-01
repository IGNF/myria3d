# Performing inference on new data [TODO]

Refer to [this tutorial](./setup_install.md) for how to setup a virtual environment and install the library.

## Run inference from installed package

To run inference, you will need:
- A source cloud point in LAS format on which to infer new classes. and probabilites.
- A checkpoint of a trained lightning module implementing model logic (class `lidar_multiclass.models.model.Model`)
- A minimal yaml configuration specifying parameters. We use [hydra](https://hydra.cc/) to manage configurations, and this yaml results from the model training. The `datamodule` and `model` parameters groups must match datset characteristics and model training settings.  The `predict` parameters group specifies path to models and data as well as batch size (N=50 works well, the larger the faster) and use of gpu (optionnal).

Fill out the {missing parameters} and run: 

```bash
python -m lidar_multiclass.predict --config-path {/path/to/.hydra} --config-name {config.yaml} predict.src_las={/path/to/cloud.las} predict.output_dir={/path/to/out/dir/} predict.resume_from_checkpoint={/path/to/checkpoint.ckpt} predict.gpus={0 for none, [i] to use GPU number i} datamodule.batch_size={N} hydra.run.dir={path/for/hydra/logs}
```

To show you current inference config, simply add a `--help` flag 

```bash
python -m lidar_multiclass.predict --config-path {/path/to/.hydra} --config-name {config.yaml} --help
```

#### Run inference from sources

From the line for package-based inference above, simply change `python -m lidar_multiclass.predict` to `python run.py` to run directly from sources.

In case you want to swicth to package-based inference, you will need to comment out the parameters that depends on local environment variables such as logger credentials and training data directory. You can do so by making a copy of the `config.yaml` file and commenting out the lines containing `oc.env` logic.