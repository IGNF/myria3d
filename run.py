import comet_ml  # noqa
import dotenv
import hydra
from omegaconf import DictConfig


# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):
    """Entrypoint to training-related logic and inference code.

    Run `python run.py -h` to print configuration and see parameters.

    Hydra configs can be overriden in CLI with `--config-path` and `--config-name` arguments.

    """

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from lidar_multiclass.utils import utils
    from lidar_multiclass.train import train
    from lidar_multiclass.predict import predict

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=False)

    task_name = config.task.get("task_name")
    if "fit" in task_name or "test" in task_name or "finetune" in task_name:
        """Training, eval, and test of a neural network."""
        return train(config)
    elif config.task.get("task_name") == "predict":
        """Infer probabilities and automate semantic segmentation decisions on unseen data."""
        return predict(config)


if __name__ == "__main__":
    main()
