import comet_ml
import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from semantic_val.optimize import optimize
    from semantic_val.utils import utils
    from semantic_val.train import train
    from semantic_val.predict import predict

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    if config.get("task") == "train":
        """Training, eval, and test of a neural network."""
        return train(config)
    elif config.get("task") == "optimize":
        """Optimization of decision threshold applied to predictions of the NN."""
        return optimize(config)
    elif config.get("task") == "predict":
        """Infer probabilities and automate semantic segmentation decisions on unseen data."""
        return predict(config)


if __name__ == "__main__":
    main()
