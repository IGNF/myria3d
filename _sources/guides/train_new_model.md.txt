# How to train new models

Refer to [this tutorial](../tutorials/setup_install.md) for how to setup a virtual environment and install the library.
Refer to [this other tutorial](../tutorials/prepare_dataset.md) for how to prepare a training-ready dataset.

Once your python environment is set up and your dataset ready for training, proceed to the next section.

Some environment variable to be injected at runtime can be specified in a `.env` file. Rename `.env_example` to `.env` and fill out: 
- `LOG PATH`, where hydra logs and config are saved.
- `PREPARED_DATA_DIR`, which specifies where to look for your prepared dataset.
- `LOGGER` section, which specifies credentials needed for logging to [comet.ml](https://www.comet.ml/). You will need to create an account first if you choose to use Comet. Alternatively, setting `logger=csv` at runtime will save results in a single, local csv file, and disable comet logging.

Define your experiment hyperparameters in an experiment file in the `configs/experiment` folder. You may also use one of the provided experiment file. 

To test your setup, logging capabilities, you may want beforehand to try overfitting on a single batch of a Swiss dataset, with

```bash
python run.py experiment=RandLaNetDebug.yaml
```

To run the full training and validation for e.g. French Lidar HD, run:

```bash
python run.py experiment=RandLaNet_base_run_FR.yaml
```

After training, you model best checkpoints and hydra config will be saved in a `DATE/TIME/` subfolder of the `LOG_PATH` you specified, with an associated hydra `config.yaml`.

To use the checkpointed model, refet to section [Performing inference on new data](../tutorials/make_predictions.md).