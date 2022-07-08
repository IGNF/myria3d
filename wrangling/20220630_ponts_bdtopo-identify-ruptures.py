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
RUPTURE_INDEX_THRESHOLD = 0.03

for p in tqdm.tqdm(["train", "val"]):
    input_dir_phase = os.path.join(INPUT_DATA_DIR, p)
    output_dir_phase = os.path.join(OUTPUT_DATA_DIR, p)
    for f in tqdm.tqdm(
        glob.glob(os.path.join(input_dir_phase, "*las"), recursive=True)
    ):
        f_out = os.path.join(output_dir_phase, os.path.basename(f))
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
        pos = np.asarray(
            [points["X"], points["Y"], points["Z"]], dtype=np.float32
        ).transpose()

        y = torch.Tensor(y[:, None])
        pos = torch.Tensor(pos)

        with torch.no_grad():
            assign_index = knn(pos, pos, 32, num_workers=4)
            y_idx, x_idx = assign_index

            diff = pos[x_idx] - pos[y_idx]
            squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
            weights = 1.0 / (1 + squared_distance)

            is_rupture_ground_bridge = 1 * (
                torch.abs(y[x_idx] - y[y_idx]) == (BRIDGE - GROUND)
            )

            y = scatter_add(
                is_rupture_ground_bridge * weights, y_idx, dim=0, dim_size=pos.size(0)
            )
            y = y / scatter_add(weights, y_idx, dim=0, dim_size=pos.size(0))

        points["rupture"][:] = y.cpu().numpy()[:, 0]
        points["rupture_heq_0_03"][:] = 1 * (
            y.cpu().numpy()[:, 0] >= RUPTURE_INDEX_THRESHOLD
        )

        os.makedirs(os.path.dirname(f_out), exist_ok=True)

        pipeline = pdal.Writer.las(
            filename=f_out,
            extra_dims="rupture=float,rupture_heq_0_03=uint",
            minor_version=4,
            dataformat_id=8,
        ).pipeline(points)
        pipeline.execute()
