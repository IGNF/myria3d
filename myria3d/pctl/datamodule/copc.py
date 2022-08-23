from pytorch_lightning import LightningDataModule


class COPCLidarDataModule(LightningDataModule):
    """Datamodule to feed train and validation data to the model via COPC format.

    COPC might be valuable for data augmentation but comes with speed limitations.

    """

    def __init__(self):
        raise NotImplementedError()
