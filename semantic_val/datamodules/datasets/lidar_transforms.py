import laspy
import numpy as np


def load_las_file(filename):
    """Load a cloud of points and its labels. We transpose to have cloud with shape [n_points, n_features]."""
    las = laspy.read(filename)
    cloud = np.asarray(
        [
            las.x - las.x.min(),
            las.y - las.y.min(),
            las.z,
            las.intensity,
            las.return_num,
            las.num_returns,
        ],
        dtype=np.float32,
    )
    cloud = cloud.transpose()
    labels = las.classification.astype(np.int)
    return cloud, labels


def get_random_subtile_center(cloud: np.ndarray, subtile_width_meters: float = 100.0):
    """
    Randomly select x/y pair (in meters) as potential center of a square subtile of original tile
    (whose x and y coordinates are in meters and in 0m-1000m range).
    """
    half_subtile_width_meters = subtile_width_meters / 2
    low = cloud[:, :2].min(0) + half_subtile_width_meters
    high = cloud[:, :2].max(0) - half_subtile_width_meters

    subtile_center_xy = np.random.uniform(low, high)

    return subtile_center_xy


def get_all_subtile_centers(
    cloud: np.ndarray, subtile_width_meters: float = 100.0, subtile_overlap: float = 0
):
    """Get centers of subtiles of specified width, assuming rectangular form of input cloud."""
    half_subtile_width_meters = subtile_width_meters / 2
    low = cloud[:, :2].min(0) + half_subtile_width_meters
    high = cloud[:, :2].max(0) - half_subtile_width_meters + 1
    centers = [
        (x, y)
        for x in np.arange(start=low[0], stop=high[0], step=subtile_width_meters - subtile_overlap)
        for y in np.arange(start=low[1], stop=high[1], step=subtile_width_meters - subtile_overlap)
    ]
    return centers


def get_subsampling_mask(input_size, subsampling_size):
    """Get a mask to select subsampling_size elements from an iterable of specified size, with replacement if needed."""

    if input_size >= subsampling_size:
        sampled_points_idx = np.random.choice(input_size, subsampling_size, replace=False)
    else:
        sampled_points_idx = np.concatenate(
            [
                np.arange(input_size),
                np.random.choice(input_size, subsampling_size - input_size, replace=True),
            ]
        )
    return sampled_points_idx


def get_subtile_data(
    las,
    las_labels,
    subtile_center_xy,
    input_cloud_size: int = 20000,
    subtile_width_meters: float = 100.0,
):
    """Extract tile points and labels around a subtile center using Chebyshev distance, in meters."""

    chebyshev_distance = np.max(np.abs(las[:, :2] - subtile_center_xy), axis=1)
    mask = chebyshev_distance < (subtile_width_meters / 2)
    cloud = las[mask]
    labels = las_labels[mask]

    input_size = len(cloud)
    sampled_points_idx = get_subsampling_mask(input_size, input_cloud_size)
    cloud = cloud[sampled_points_idx]
    labels = labels[sampled_points_idx]

    return cloud, labels


def transform_labels_for_building_segmentation(labels):
    """Pass from multiple classes to simpler Building/Non-Building labels.
    Initial classes: [  1,   2,   6 (detected building, no validation),  19 (valid building),  20 (surdetection, unspecified),
    21 (building, forgotten), 104, 110 (surdetection, others), 112 (surdetection, vehicule), 114 (surdetection, others), 115 (surdetection, bridges)]
    Final classes: 0 (non-building), 1 (building)
    """
    buildings = (labels == 19) | (labels == 21) | (labels == 6)
    labels[buildings] = 1
    labels[~buildings] = 0
    # labels = np.stack((labels, 1 - labels), axis=-1)
    return labels


def augment(cloud):
    """Data augmentation at training time."""
    # TODO
    return cloud
