""" We define two datasets:
  - TrainDataset with random selection of a sub-tile
  - ValDataset = TestDataset with an exhaustive parsing of the sub-tiles.
"""

import random
import os.path as osp
import time
from typing import List
import numpy as np
import pandas as pd
import shapefile
from torch.utils.data import Dataset, IterableDataset
from torch_geometric.data.data import Data
from semantic_val.utils import utils

log = utils.get_logger(__name__)


class LidarTrainDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        loading_function=None,
        transform=None,
        target_transform=None,
        subtile_width_meters: float = 100,
        train_subtiles_by_tile: int = None,
    ):
        self.files = files
        self.loading_function = loading_function
        self.transform = transform
        self.target_transform = target_transform

        self.subtile_width_meters: float = subtile_width_meters
        self.in_memory_filepath: str = None

        self.nb_files = len(self.files)
        self.train_subtiles_by_tile = train_subtiles_by_tile

    def __len__(self):
        return self.nb_files * self.train_subtiles_by_tile

    def __getitem__(self, idx):
        """Get a subtitle from indexed las file, and apply the transforms specified in datamodule."""
        filepath = self.files[idx]
        data = self.access_or_load_cloud_data(filepath)

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            data = self.target_transform(data)

        return data

    def access_or_load_cloud_data(self, filepath):
        """Get already in-memory cloud data or load it from disk."""
        if self.in_memory_filepath == filepath:
            data = self.data
        else:
            log.debug(f"Loading train file: {filepath}")
            data = self.loading_function(filepath)
            self.in_memory_filepath = filepath
            self.data = data
        return data


class LidarValDataset(IterableDataset):
    def __init__(
        self,
        files,
        loading_function=None,
        transform=None,
        target_transform=None,
        subtile_overlap: float = 0,
        subtile_width_meters: float = 100,
    ):
        self.files = files
        self.loading_function = loading_function
        self.transform = transform
        self.target_transform = target_transform
        self.subtile_overlap = subtile_overlap
        self.subtile_width_meters = subtile_width_meters

    def process_data(self):
        """Yield subtiles from all tiles in an exhaustive fashion."""

        for idx, filepath in enumerate(self.files):
            log.info(f"Predicting for file {idx+1}/{len(self.files)} [{filepath}]")
            tile_data = self.loading_function(filepath)
            centers = self.get_all_subtile_centers(tile_data)
            ts = time.time()
            for tile_data.current_subtile_center in centers:
                if self.transform:
                    data = self.transform(tile_data)
                if data is not None:
                    if self.target_transform:
                        data = self.target_transform(data)
                    yield data

            log.info(f"Took {(time.time() - ts):.6} seconds")

    def __iter__(self):
        return self.process_data()

    def get_all_subtile_centers(self, data: Data):
        """Get centers of square subtiles of specified width, assuming rectangular form of input cloud."""

        half_subtile_width_meters = self.subtile_width_meters / 2
        low = data.pos[:, :2].min(0) + half_subtile_width_meters
        high = data.pos[:, :2].max(0) - half_subtile_width_meters + 1
        centers = [
            (x, y)
            for x in np.arange(
                start=low[0],
                stop=high[0],
                step=self.subtile_width_meters - self.subtile_overlap,
            )
            for y in np.arange(
                start=low[1],
                stop=high[1],
                step=self.subtile_width_meters - self.subtile_overlap,
            )
        ]
        random.shuffle(centers)
        return centers


# PREPARE DATASET


def make_datasplit_csv(
    lasfiles_dir, shapefile_filepath, datasplit_csv_filepath, train_frac=0.8
):
    """Turn the shapefile of tiles metadata into a csv with stratified train-val-test split."""
    df = get_metadata_df_from_shapefile(shapefile_filepath)
    df_split = get_splitted_SemValBuildings202110(df, train_frac=train_frac)
    df_split = create_full_filepath_column(df_split, lasfiles_dir)
    file_not_found_index_list = []
    for filepath in df_split.file_path:
        if not osp.exists(filepath):
            log.warning(
                "file specified but not found, removing {0} from the list (index {1})".format(
                    filepath, df_split.index[df_split["file_path"] == filepath].tolist()
                )
            )
            file_not_found_index_list += df_split.index[
                df_split["file_path"] == filepath
            ].tolist()
    df_split.drop(labels=file_not_found_index_list, inplace=True)
    df_split.to_csv(datasplit_csv_filepath, index=False)


def get_shapefile_records_df(sf):
    """Get the shapefile records as a pd.DataFrame."""
    records = sf.records()
    fields = [x[0] for x in sf.fields][1:]
    df = pd.DataFrame(columns=fields, data=records)
    return df


def get_metadata_df_from_shapefile(filepath):
    """Get the shapefile records of tiles metadata as a formated pd.DataFrame."""
    sf = shapefile.Reader(filepath)
    df = get_shapefile_records_df(sf)
    df = df.sort_values(by="file")
    int_col = ["port", "nb_vehicul", "nb_bati", "nb_veget", "nb_autre"]
    df[int_col] = df[int_col].astype(int)
    return df


def get_splitted_SemValBuildings202110(df, train_frac=0.8):
    """
    Dataset name: "202110_building_val"
    From the formated pd.DataFrame of tiles metadata (output by get_metadata_df_from_shapefile)
    stratify the dataset based on geographical layer and port areas.
    """

    train_tiles = []
    val_tiles = []
    test_tiles = []

    df_to_split = df[df.layer == "71_polygo"]
    tesval_n = int((1 - train_frac) * len(df_to_split))
    train_n = len(df_to_split) - tesval_n

    train, val, test = np.split(
        df_to_split.sample(frac=1, random_state=0),
        [train_n + 1, train_n + int(tesval_n / 2) + 1],
    )
    train_tiles.append(train)
    val_tiles.append(val)
    test_tiles.append(test)
    # print(train.shape, test.shape, val.shape)

    df_to_split = df[df.layer == "forca_polygo"]
    tesval_n = int((1 - train_frac) * len(df_to_split))
    train_n = len(df_to_split) - tesval_n
    train, val, test = np.split(
        df_to_split.sample(frac=1, random_state=0),
        [train_n, train_n + int(tesval_n / 2)],
    )
    train_tiles.append(train)
    val_tiles.append(val)
    test_tiles.append(test)
    # print(train.shape, test.shape, val.shape)

    df_to_split = df[df.layer == "la_grande_motte_polygo"]

    ports = df_to_split.loc[df_to_split.port == 1]
    assert len(ports) == 2
    test_tiles.append(ports.iloc[0:1])
    train_tiles.append(ports.iloc[1:2])

    noports = df_to_split[df_to_split.port == 0].sample(frac=1, random_state=0)
    assert len(noports) == 6
    train_tiles.append(noports.iloc[:4])
    test_tiles.append(noports.iloc[4:5])
    val_tiles.append(noports.iloc[5:6])

    train_tiles = pd.concat(train_tiles)
    test_tiles = pd.concat(test_tiles)
    val_tiles = pd.concat(val_tiles)

    train_tiles["split"] = "train"
    val_tiles["split"] = "val"
    test_tiles["split"] = "test"

    assert len(train_tiles) == 120
    assert len(val_tiles) == 15
    assert len(test_tiles) == 15

    df_split = pd.concat([train_tiles, val_tiles, test_tiles])

    return df_split


def create_full_filepath_column(df, dirpath):
    """Append dirpath as a suffix to file column"""
    df["file_path"] = df["file"].apply(lambda stem: osp.join(dirpath, stem))
    return df
