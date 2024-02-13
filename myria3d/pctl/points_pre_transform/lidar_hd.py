# function to turn points loaded via pdal into a pyg Data object, with additional channels
from colorsys import rgb_to_hsv
from math import pi, sin, cos

import numpy as np
from torch_geometric.data import Data

from pgeof import pgeof
from sklearn.neighbors import NearestNeighbors

COLORS_NORMALIZATION_MAX_VALUE = 255.0 * 256.0
RETURN_NUMBER_NORMALIZATION_MAX_VALUE = 7.0


def lidar_hd_pre_transform(points):
    """Turn pdal points into torch-geometric Data object.

    Builds a composite (average) color channel on the fly.     Calculate NDVI on the fly.

    Args:
        las_filepath (str): path to the LAS file.

    Returns:
        Data: the point cloud formatted for later deep learning training.

    """
    # Positions and base features
    pos = np.asarray([points["X"], points["Y"], points["Z"]], dtype=np.float32).transpose()
    # normalization
    occluded_points = points["ReturnNumber"] > 1

    points["ReturnNumber"] = (points["ReturnNumber"]) / (RETURN_NUMBER_NORMALIZATION_MAX_VALUE)
    points["NumberOfReturns"] = (points["NumberOfReturns"]) / (
        RETURN_NUMBER_NORMALIZATION_MAX_VALUE
    )

    for color in ["Red", "Green", "Blue", "Infrared"]:
        assert points[color].max() <= COLORS_NORMALIZATION_MAX_VALUE
        points[color][:] = points[color] / COLORS_NORMALIZATION_MAX_VALUE
        points[color][occluded_points] = 0.0


    x_features_names = [
        "Intensity",
        "ReturnNumber",
        "NumberOfReturns",
        "Red",
        "Green",
        "Blue",
        "Cos_Hue",
        "Sin_Hue",
        "Shade",
        "Value",
        "Infrared",
        "ScanAngleRank",
        "rgb_avg",
        "ndvi",
        "linearity",
        "planarity",
        "scattering",
        "verticality",
        "normal_x",
        "normal_y",
        "normal_z",
    ]

    # Additional features :
    # Average color, that will be normalized on the fly based on single-sample
    rgb_avg = (
        np.asarray([points["Red"], points["Green"], points["Blue"]], dtype=np.float32)
        .transpose()
        .mean(axis=1)
    )

    # NDVI
    ndvi = (points["Infrared"] - points["Red"]) / (points["Infrared"] + points["Red"] + 10**-6)

    # geometry
    k = min(100, len(points))   # in case there are too few points
    kneigh = NearestNeighbors(n_neighbors=k).fit(pos).kneighbors(pos)
    nn_ptr = np.arange(pos.shape[0] + 1) * k
    nn = kneigh[1].flatten()

    # Make sure xyz are float32 and nn and nn_ptr are uint32
    pos = pos.astype('float32')
    nn_ptr = nn_ptr.astype('uint32')
    nn = nn.astype('uint32')

    # Make sure arrays are contiguous (C-order) and not Fortran-order
    pos = np.ascontiguousarray(pos)
    nn_ptr = np.ascontiguousarray(nn_ptr)
    nn = np.ascontiguousarray(nn)  

    # geof = pgeof(points, nn, nn_ptr, k_min=10, k_step=1, k_min_search=15, verbose=True).transpose()
    geof = pgeof(points, nn, nn_ptr, verbose=True).transpose()

    # Pre-allocate memory
    x = np.empty((points.shape[0], len(x_features_names)))

    # Fill x
    for index, point in enumerate(points):
        hue, shade, value = rgb_to_hsv(point["Red"], point["Green"], point["Blue"])
        x[index] = [
            point["Intensity"],
            point["ReturnNumber"],
            point["NumberOfReturns"],
            point["Red"],
            point["Green"],
            point["Blue"],          
            cos(2 * pi * hue),
            sin(2 * pi * hue),
            shade,
            value,
            point["Infrared"],
            point["ScanAngleRank"],
            rgb_avg[index],
            ndvi[index],
            geof[0][index],    # linearity
            geof[1][index],    # planarity
            geof[2][index],    # scattering
            geof[3][index],    # verticality
            geof[4][index],    # normal_x
            geof[5][index],    # normal_y
            geof[6][index],    # normal_z
            ]

    # # todo
    # x = np.stack(
    #     [
    #         points[name]
    #         for name in [
    #             "Intensity",
    #             "ReturnNumber",
    #             "NumberOfReturns",
    #             "Red",
    #             "Green",
    #             "Blue",
    #             "Infrared",
    #         ]
    #     ]
    #     + [rgb_avg, ndvi],
    #     axis=0,
    # ).transpose()


    y = points["Classification"]

    data = Data(pos=pos, x=x.astype('float32'), y=y, x_features_names=x_features_names)

    return data
