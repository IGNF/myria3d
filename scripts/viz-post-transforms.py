"""
A simple loop to visualize transformed Data object right before they are fed into the model.
Random Sampling can be performed to evaluate the level of data-degradation that occurs in models that
need fixed size input point clouds (i.e. RandLa-Net).
TODO: use an argparser to avoid hardcoded paths.
"""

import os
import sys
from hydra.utils import instantiate
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule
import matplotlib.pyplot as plt
import torch
import math

# need to import myria3d for its hydra resolver to be loaded
import myria3d  # noqa


# Parameter to vary sampling strength
voxel_size = 0.5
NUM_NODES_SAMPLING = 20000  # grid_sampler = GridSampling(voxel_size)
# random_sampler = FixedPoints(NUM_NODES_SAMPLING, replace=False, allow_duplicates=True)
# SRC_LAS = "/home/cgaydon/repositories/lidar-deep-segmentation/tests/data/toy_dataset_src/870000_6618000.subset.50mx100m.las"
SRC_LAS = "/var/data/cgaydon/data/20220607_151_dalles_proto-prepared-V2.2.0/test/730000_6361000.las"
IMG_SAVE_PATH = f"/home/cgaydon/repositories/lidar-deep-segmentation/outputs/viz/post_transform_{voxel_size}_{NUM_NODES_SAMPLING}/"
os.makedirs(IMG_SAVE_PATH, exist_ok=True)


@hydra.main(
    config_path="/home/cgaydon/repositories/lidar-deep-segmentation/configs",
    config_name="config.yaml",
)
def visualize_dataset(cfg: DictConfig) -> None:
    instantiate(cfg)
    print(OmegaConf.to_yaml(cfg))
    cfg.datamodule.batch_size = 1
    cfg.datamodule.transforms.preparations.GridSampling._args_ = [voxel_size]
    cfg.datamodule.transforms.preparations.FixedPoints._args_ = [NUM_NODES_SAMPLING]
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule._set_predict_data(SRC_LAS)
    for idx, batch in enumerate(datamodule.predict_dataloader()):
        print(idx)
        print(batch.pos.shape)
        datamodule._visualize_graph(batch)
        plt.savefig(f"{IMG_SAVE_PATH}{str(idx).zfill(4)}")
        plt.close()


def random_sample(data):
    """Samples the data object pos, y, and x"""
    num_nodes = data.pos.shape[0]
    choice = torch.cat(
        [
            torch.randperm(num_nodes)
            for _ in range(math.ceil(NUM_NODES_SAMPLING / num_nodes))
        ],
        dim=0,
    )[:NUM_NODES_SAMPLING]
    data.pos = data.pos[choice]
    data.x = data.x[choice]
    data.y = data.y[choice]
    return data


sys.argv.append("experiment=predict")
visualize_dataset()
