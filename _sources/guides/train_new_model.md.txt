# How to train new models

Refer to [this tutorial](../tutorials/install_on_linux.md) for how to setup a virtual environment and install the library.
Refer to [this other tutorial](../tutorials/prepare_dataset.md) for how to prepare a training-ready dataset.

## Setup

Once your python environment is set up and your dataset ready for training, proceed to the next section.

Some environment variable need to be injected at runtime can be specified in a `.env` file. Rename `.env_example` to `.env` and fill out: 
- `LOGS_DIR`, where hydra logs and config will be saved.
- `LOGGER` section, which specifies credentials needed for logging to [comet.ml](https://www.comet.ml/). You will need to create an account first if you choose to use Comet. Alternatively, setting `logger=csv` at runtime will save results in a single, local csv file, and disable comet logging.

## Quick run

To test your setup and logging capabilities, you can try overfitting on a single batch of data from the toy dataset, created following instructions in [this page](../tutorials/prepare_dataset.md)).
To overfit on a single batch for 30 epochs, run:

```bash
python run.py experiment=RandLaNet-Overfit
```

## Training

Define your experiment hyperparameters in an experiment file in the `configs/experiment` folder. You may stem from one of the provided experiment file (e.g. `RandLaNet_base_run_FR.yaml`). In particular, you will need to define `dataset_description` to specify your classification task - see config `20220607_151_dalles_proto.yaml` for an example.


To run the full training and validation for French Lidar HD, run:

```bash
python run.py experiment=RandLaNet_base_run_FR
```

After training, you model best checkpoints and hydra config will be saved in a `DATE/TIME/` subfolder of the `LOGS_DIR` you specified, with an associated hydra `config.yaml`.

### Optimized learning rate

Pytorch Lightning support au [automated learning rate finder](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#auto-lr-find), by means of an Learning Rate-range test (see section 3.3 in [this paper](https://arxiv.org/pdf/1506.01186.pdf) for reference). 
You can perfom this automatically before training by setting `trainer.auto_lr_find=true` when calling training on your dataset. The best learning rate will be logged and results saved as an image, so that you do not need to perform this test more than once.

### Multi-GPUs

Multi-GPUs training is supported. Refer to e.g. experiment file `RandLaNet_base_run_FR-MultiGPU.yaml` for pytorch lightning flags to activate it. 
Multi-GPUs training effectively reduces training time by the number of GPUs used. Batch size might need to be reduced to keep a constant number of steps per epoch in DDP.

## Testing the model

Test will be automatically performed after each training, using best checkpointeded model.

To manually evaluate per-class IoU on the test set, run:

```bash
python run.py \
--config-path {/path/to/.hydra} \
--config-name {config.yaml} \
task.task_name="test" \
model.ckpt_path={/path/to/checkpoint.ckpt} \
trainer.gpus={0 for none, [i] to use GPU number i} \
```
Arguments `config-path` and `config-name` means you are using the saved configuration from your training, which contains the path to the prepared HDF5 dataset. 

If you are using defaut configurations, you can call test using a custom experiment:

```bash
python run.py experiment=test
```

## Inference

To use the checkpointed model to make predictions on new data, refer to section [Performing inference on new data](../tutorials/make_predictions.md).