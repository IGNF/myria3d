import numpy as np
from torch_geometric.data import Data
from numpy.lib.recfunctions import append_fields


COLORS_NORMALIZATION_MAX_VALUE = 255.0 * 256.0
RETURN_NUMBER_NORMALIZATION_MAX_VALUE = 7.0

def lidar_hd_pre_transform(points):
    """Turn pdal points into torch-geometric Data object.

    Builds a composite (average) color channel on the fly. Calculate NDVI on the fly.

    Args:
        points (np.ndarray): points loaded via PDAL

    Returns:
        Data: the point cloud formatted for deep learning training.
    """
    # Positions
    pos = np.asarray([points["X"], points["Y"], points["Z"]], dtype=np.float32).transpose()

    # normalization
    occluded_points = points["ReturnNumber"] > 1

    points["ReturnNumber"] = (points["ReturnNumber"]) / RETURN_NUMBER_NORMALIZATION_MAX_VALUE
    points["NumberOfReturns"] = (points["NumberOfReturns"]) / RETURN_NUMBER_NORMALIZATION_MAX_VALUE
    
    # Ensure all color fields exist, even if missing (filled with 0)
    for color in ["Red", "Green", "Blue", "Infrared"]:
        if color not in points.dtype.names:
            print(f"Color channel {color} not found. Creating fake {color} filled with 0.")
            fake_color = np.zeros(points.shape[0], dtype=np.float32)
            points = append_fields(points, color, fake_color, dtypes=np.float32, usemask=False)

    # Normalize colors if available
    available_colors = []
    for color in ["Red", "Green", "Blue", "Infrared"]:
        if color in points.dtype.names:
            assert points[color].max() <= COLORS_NORMALIZATION_MAX_VALUE, f"{color} max too high!"
            points[color][:] = points[color] / COLORS_NORMALIZATION_MAX_VALUE
            points[color][occluded_points] = 0.0
            available_colors.append(color)
        else:
            print(f"Warning: {color} channel not found. Skipping.")

    # Additional features
    rgb_avg = np.zeros(points.shape[0], dtype=np.float32)
    if all(c in points.dtype.names for c in ["Red", "Green", "Blue"]):
        rgb_avg = (
            np.asarray([points["Red"], points["Green"], points["Blue"]], dtype=np.float32)
            .transpose()
            .mean(axis=1)
        )

    # NDVI
    ndvi = np.zeros(points.shape[0], dtype=np.float32)
    if "Infrared" in points.dtype.names and "Red" in points.dtype.names:
        ndvi = (points["Infrared"] - points["Red"]) / (points["Infrared"] + points["Red"] + 1e-6)

    # Features list: dynamically adapt based on what exists
    x_list = [
        points["Intensity"],
        points["ReturnNumber"],
        points["NumberOfReturns"],
    ]

    x_features_names = [
        "Intensity",
        "ReturnNumber",
        "NumberOfReturns",
    ]

    for color in ["Red", "Green", "Blue", "Infrared"]:
        if color in points.dtype.names:
            x_list.append(points[color])
            x_features_names.append(color)

    # Always add computed features
    x_list += [rgb_avg, ndvi]
    x_features_names += ["rgb_avg", "ndvi"]

    x = np.stack(x_list, axis=0).transpose()

    y = points["Classification"]

    data = Data(pos=pos, x=x, y=y, x_features_names=x_features_names)

    return data
