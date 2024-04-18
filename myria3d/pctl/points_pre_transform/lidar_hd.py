# function to turn points loaded via pdal into a pyg Data object, with additional channels
from typing import List
import numpy as np
import torch
from torch_geometric.data import Data

COLORS_NORMALIZATION_MAX_VALUE = 255.0 * 256.0
RETURN_NUMBER_NORMALIZATION_MAX_VALUE = 7.0


def lidar_hd_pre_transform(points, pos_keys: List[str], features_keys: List[str], color_keys: List[str]):
    """Turn pdal points into torch-geometric Data object.

    Builds a composite (average) color channel on the fly.     Calculate NDVI on the fly.

    Args:
        las_filepath (str): path to the LAS file.
        pos_keys (List[str]): list of keys for positions and base features
        features_keys (List[str]): list of keys for
    Returns:
        Data: the point cloud formatted for later deep learning training.

    """

    features = pos_keys + features_keys + color_keys
    # Positions and base features
    pos = np.asarray([points[k] for k in pos_keys], dtype=np.float32).transpose()
    # normalization
    if "ReturnNumber" in features:
        occluded_points = points["ReturnNumber"] > 1
        points["ReturnNumber"] = (points["ReturnNumber"]) / (RETURN_NUMBER_NORMALIZATION_MAX_VALUE)
        points["NumberOfReturns"] = (points["NumberOfReturns"]) / (
            RETURN_NUMBER_NORMALIZATION_MAX_VALUE
        )
    else:
        occluded_points = np.zeros(pos.shape[0], dtype=np.bool_)

    for color in color_keys:
        assert points[color].max() <= COLORS_NORMALIZATION_MAX_VALUE
        points[color][:] = points[color] / COLORS_NORMALIZATION_MAX_VALUE
        points[color][occluded_points] = 0.0

    # Additional features :
    # Average color, that will be normalized on the fly based on single-sample
    if "Red" in color_keys and "Green" in color_keys and "Blue" in color_keys:
        rgb_avg = (
            np.asarray([points["Red"], points["Green"], points["Blue"]], dtype=np.float32)
            .transpose()
            .mean(axis=1)
        )
    else:
        rgb_avg = None

    # NDVI
    if "Infrared" in color_keys and "Red" in color_keys:
        ndvi = (points["Infrared"] - points["Red"]) / (points["Infrared"] + points["Red"] + 10**-6)
    else:
        ndvi = None

    additional_color_features = []
    additional_color_keys = []
    if rgb_avg is not None:
        additional_color_features.append(rgb_avg)
        additional_color_keys.append("rgb_avg")
    if ndvi is not None:
        additional_color_features.append(ndvi)
        additional_color_keys.append("ndvi")

    x = np.stack(
        [
            points[name]
            for name in features_keys + color_keys
        ]
        + additional_color_features,
        axis=0,
    ).transpose()
    x_features_names = [s.encode('utf-8') for s in (features_keys + color_keys + additional_color_keys)]
    y = points["Classification"]

    data = Data(pos=torch.from_numpy(pos), x=torch.from_numpy(x), y=y, x_features_names=x_features_names)

    return data
