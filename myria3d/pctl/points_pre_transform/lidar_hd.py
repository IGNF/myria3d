# function to turn points loaded via pdal into a pyg Data object, with additional channels
from colorsys import rgb_to_hsv
from math import pi, sin, cos

import numpy as np
from torch_geometric.data import Data



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
    pos = np.asarray(
        [points["X"], points["Y"], points["Z"]], dtype=np.float32
    ).transpose()
    # normalization
    occluded_points = points["ReturnNumber"] > 1

    points["ReturnNumber"] = (points["ReturnNumber"]) / (
        RETURN_NUMBER_NORMALIZATION_MAX_VALUE
    )
    points["NumberOfReturns"] = (points["NumberOfReturns"]) / (
        RETURN_NUMBER_NORMALIZATION_MAX_VALUE
    )

    for color in ["Red", "Green", "Blue", "Infrared"]:
        assert points[color].max() <= COLORS_NORMALIZATION_MAX_VALUE
        points[color][:] = points[color] / COLORS_NORMALIZATION_MAX_VALUE
        # points[color][occluded_points] = 0.0

    x_features_names = [
        "Intensity",
        "ReturnNumber",
        "NumberOfReturns",
        # "Hue",
        "Cos_Hue",
        "Sin_Hue",
        "Shade",
        "Value",
        "Infrared",
        "rgb_avg",
        "ndvi",
    ]

    # creating x

    # # NDVI
    ndvi = (points["Infrared"] - points["Red"]) / (
        points["Infrared"] + points["Red"] + 10**-6
    )

    # Average color, that will be normalized on the fly based on single-sample
    rgb_avg = (
        np.asarray([points["Red"], points["Green"], points["Blue"]], dtype=np.float32)
        .transpose()
        .mean(axis=1)
    )

    # Pre-allocate memory
    x = np.empty((points.shape[0], len(x_features_names)))

    # Fill x
    for index, point in enumerate(points):
        hue, shade, value = rgb_to_hsv(point["Red"], point["Green"], point["Blue"])
        x[index] = [
            point["Intensity"],
            point["ReturnNumber"],
            point["NumberOfReturns"],
            # hue,
            cos(2 * pi * hue),
            sin(2 * pi * hue),
            shade,
            value,
            point["Infrared"],
            rgb_avg[index],
            ndvi[index]
            ]
    #     points["Infrared"] + points["Red"] + 10**-6


    # # Additional features :
    # # Average color, that will be normalized on the fly based on single-sample
    # rgb_avg = (
    #     np.asarray([points["Red"], points["Green"], points["Blue"]], dtype=np.float32)
    #     .transpose()
    #     .mean(axis=1)
    # )

    # # NDVI
    # ndvi = (points["Infrared"] - points["Red"]) / (
    #     points["Infrared"] + points["Red"] + 10**-6
    # )

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

    data = Data(pos=pos, x=x, y=y, x_features_names=x_features_names)

    return data
