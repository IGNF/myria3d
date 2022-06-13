import os
from typing import Union
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import hydra
from pytorch_lightning import LightningDataModule

# need to import myria3d for its hydra resolver to be loaded
import myria3d
from torch_geometric.data import Data, Batch
import networkx as nx
import matplotlib.pyplot as plt
import torch
import math

# Parameter to vary sampling strength
NUM_NODES_SAMPLING = 40000
SRC_LAS = "/home/cgaydon/repositories/lidar-deep-segmentation/tests/data/toy_dataset_src/870000_6618000.subset.50mx100m.las"
IMG_SAVE_PATH = f"/home/cgaydon/repositories/lidar-deep-segmentation/outputs/viz/post_transform_{NUM_NODES_SAMPLING}/"
os.makedirs(IMG_SAVE_PATH, exist_ok=True)


@hydra.main(
    config_path="/home/cgaydon/repositories/lidar-deep-segmentation/configs",
    config_name="config.yaml",
)
def visualize_dataset(cfg: DictConfig) -> None:
    instantiate(cfg)
    print(OmegaConf.to_yaml(cfg))
    cfg.datamodule.batch_size = 1
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    datamodule._set_predict_data(SRC_LAS)
    for idx, batch in enumerate(datamodule.predict_dataloader()):
        batch = random_sample(batch)
        datamodule._visualize_graph(batch)
        plt.savefig(f"{IMG_SAVE_PATH}{str(idx).zfill(4)}")


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


visualize_dataset()
