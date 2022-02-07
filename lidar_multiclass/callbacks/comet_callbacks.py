import os
import comet_ml
from pathlib import Path

from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import CometLogger, LoggerCollection
from pytorch_lightning.utilities import rank_zero_only

from lidar_multiclass.utils import utils

log = utils.get_logger(__name__)


def get_comet_logger(trainer: Trainer) -> CometLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if isinstance(trainer.logger, CometLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, CometLogger):
                return logger

    raise Exception(
        "You are using comet related callback, but CometLogger was not found for some reason..."
    )


class LogCode(Callback):
    """Upload all code files to comet, at the beginning of the run."""

    def __init__(self, code_dir: str):
        """

        Args:
            code_dir: the code directory
            use_git: if using git, then upload all files that are not ignored by git.
            if not using git, then upload all '*.py' file
        """
        self.code_dir = code_dir

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        logger = get_comet_logger(trainer=trainer)
        experiment = logger.experiment

        for path in Path(self.code_dir).resolve().rglob("*.py"):
            experiment.log_code(file_name=str(path))
        log.info("Logging all .py files to Comet.ml!")


class LogLogsPath(Callback):
    """Logs run working directory to comet.ml"""

    @rank_zero_only
    def on_init_end(self, trainer):
        experiment = get_comet_logger(trainer=trainer).experiment
        log_path = os.getcwd()
        log.info(f"----------------\n LOGS DIR is {log_path}\n ----------------")
        experiment.log_parameter("experiment_logs_dirpath", log_path)
