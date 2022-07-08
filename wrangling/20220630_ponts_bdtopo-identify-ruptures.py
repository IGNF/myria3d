import glob
import geopandas
import os
import numpy as np

import pandas as pd
import pdal
import torch
from torch_geometric.nn import knn
from torch_scatter import scatter_add
import tqdm

from myria3d.data.loading import _find_file_in_dir
from tests.myria3d.test_train_and_predict import pdal_read_las_array

INPUT_DATA_DIR = "/var/data/cgaydon/mnt/store-lidarhd/projet-LHD/IA/Multiclass-Segmentation/data/20220630_ponts_bdtopo/data/"
OUTPUT_DATA_DIR = "/var/data/cgaydon/mnt/store-lidarhd/projet-LHD/IA/Multiclass-Segmentation/data/20220630_ponts_bdtopo/data_with_rupture/"

BRIDGE = 17
GROUND = 2


phases = ["train", "val"]
# for p in tqdm.tqdm(phases):
#     for f in tqdm.tqdm(
#         glob.glob(os.path.join(INPUT_DATA_DIR, p, "*las"), recursive=True)
#     ):
#         print(f)

f = "/var/data/cgaydon/mnt/store-lidarhd/projet-LHD/IA/Multiclass-Segmentation/data/20220630_ponts_bdtopo/data/val/prioritaire_321.las"
f_out = "/var/data/cgaydon/mnt/store-lidarhd/projet-LHD/IA/Multiclass-Segmentation/data/20220630_ponts_bdtopo/data_with_rupture/val/prioritaire_321.las"
p1 = (
    pdal.Pipeline()
    | pdal.Reader.las(filename=f)
    | pdal.Filter.ferry(dimensions="=>rupture")
    | pdal.Filter.ferry(dimensions="=>rupture_heq_0_03")
)
p1.execute()
points = p1.arrays[0]
y = points["Classification"].copy()
y = np.vectorize(lambda code: code if code in (1, 2, 17) else 1)(y)
pos = np.asarray([points["X"], points["Y"], points["Z"]], dtype=np.float32).transpose()

y = torch.Tensor(y[:, None])
pos = torch.Tensor(pos)

# Interpolate like method
with torch.no_grad():
    assign_index = knn(pos, pos, 32, num_workers=4)
    y_idx, x_idx = assign_index
    diff = pos[x_idx] - pos[y_idx]
    squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
    class_x = y[x_idx]
    class_y = y[y_idx]
    is_rupture_ground_bridge = 1 * (torch.abs(class_x - class_y) == (17 - 2))
    weights = 1.0 / (1 + squared_distance)

    y = scatter_add(
        is_rupture_ground_bridge * weights, y_idx, dim=0, dim_size=pos.size(0)
    )
    y = y / scatter_add(weights, y_idx, dim=0, dim_size=pos.size(0))

points["rupture"][:] = y.cpu().numpy()[:, 0]
points["rupture_heq_0_03"][:] = 1 * (y.cpu().numpy()[:, 0] >= 0.03)

os.makedirs(os.path.dirname(f_out), exist_ok=True)

pipeline = pdal.Writer.las(
    filename=f_out, extra_dims="all", minor_version=4, dataformat_id=8
).pipeline(points)
n = pipeline.execute()
n
