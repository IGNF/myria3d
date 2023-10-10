# function to turn points loaded via pdal into a pyg Data object, with additional channels
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
        points[color][occluded_points] = 0.0

    # Additional features :
    # Average color, that will be normalized on the fly based on single-sample
    rgb_avg = (
        np.asarray([points["Red"], points["Green"], points["Blue"]], dtype=np.float32)
        .transpose()
        .mean(axis=1)
    )

    # NDVI
    ndvi = (points["Infrared"] - points["Red"]) / (
        points["Infrared"] + points["Red"] + 10**-6
    )

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

    # geof = pgeof(
    #     points, nn, nn_ptr, k_min=10, k_step=1, k_min_search=15,
    #     verbose=True)

    geof = pgeof(points, nn, nn_ptr, k_min=10, k_step=1, k_min_search=15, verbose=True).transpose()

    # todo
    x = np.stack(
        [
            points[name]
            for name in [
                "Intensity",
                "ReturnNumber",
                "NumberOfReturns",
                "Red",
                "Green",
                "Blue",
                "Infrared",
            ]
        ]
        + [rgb_avg, ndvi]
        + [
            geof[0],    # linearity
            geof[1],    # planarity
            geof[2],    # scattering
            geof[3],    # verticality
            geof[4],    # normal_x
            geof[5],    # normal_y
            geof[6],    # normal_z

        ],
        axis=0,
    ).transpose()
    x_features_names = [
        "Intensity",
        "ReturnNumber",
        "NumberOfReturns",
        "Red",
        "Green",
        "Blue",
        "Infrared",
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
    y = points["Classification"]

    data = Data(pos=pos, x=x, y=y, x_features_names=x_features_names)

    return data
