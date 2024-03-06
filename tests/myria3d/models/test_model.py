import hydra
from pytorch_lightning import LightningDataModule
from tests.conftest import make_default_hydra_cfg

from myria3d.models.model import Model


def test_model_get_batch_tensor_by_enumeration():
    config = make_default_hydra_cfg(
        overrides=[
            "predict.src_las=tests/data/toy_dataset_src/862000_6652000.classified_toy_dataset.100mx100m.las",
            "datamodule.epsg=2154",
            "work_dir=./../../..",
            "datamodule.subtile_width=1",
            "datamodule.hdf5_file_path=null",
        ]
    )

    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
    datamodule._set_predict_data(config.predict.src_las)

    model = Model(
        neural_net_class_name="PyGRandLANet",
        neural_net_hparams=dict(num_features=2, num_classes=7),
    )
    for batch in datamodule.predict_dataloader():
        # Check that no error is raised ("TypeError: object of type 'numpy.int64' has no len()")
        _ = model._get_batch_tensor_by_enumeration(batch.idx_in_original_cloud)
