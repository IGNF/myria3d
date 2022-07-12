import glob
import os
import pickle
import shutil
from typing import List
import numpy as np
from shapely.geometry import Polygon
import shapely
import geopandas
from myria3d.data.loading import _find_file_in_dir

import pandas as pd
import pdal
import tqdm
from osgeo import gdal, ogr, osr
import pdal

# data format
BRIDGE = 17
WIDTH_SUBTILE_CENTER_SQUARE = 50
LAMBERT_93_SRID = 2154  # all json CRS need to be specified
LAMBERT_93_EPSG_STR = f"EPSG:{LAMBERT_93_SRID}"
# vectorization
GDAL_WRITER_WINDOW_SIZE = 0.25
GDAL_WRITER_RESOLUTION = 0.25

MAX_IMPORTANCE = 3

INPUT_DATA_DIR = f"/var/data/cgaydon/mnt/store-lidarhd/projet-LHD/IA/Multiclass-Segmentation/data/20220630_ponts_bdtopo/variations/imp_leq_{MAX_IMPORTANCE}/"
SPLIT_CSV = os.path.join(INPUT_DATA_DIR, "split.txt")
TARGETS_CSV = os.path.join(INPUT_DATA_DIR, "targets_A_B_D.csv")
df = pd.read_csv(SPLIT_CSV)


def main():
    df_targets = pd.DataFrame()
    for p in ["val", "train"]:
        input_dir_phase = os.path.join(INPUT_DATA_DIR, p)
        output_dir_phase = os.path.join(input_dir_phase, "oriented_bboxes")
        shutil.rmtree(output_dir_phase, ignore_errors=True)
        os.makedirs(output_dir_phase, exist_ok=True)
        selection = df[df.split == p].basename.values
        for basename in tqdm.tqdm(selection):
            in_las = _find_file_in_dir(input_dir_phase, basename)
            irregular_json = os.path.join(
                output_dir_phase, basename.replace(".las", ".json")
            )
            rectangular_json = irregular_json.replace(".json", ".obbox.json")
            points = get_XY_centered_points(in_las)
            vectorize_bridge(points, irregular_json)
            # print(f"Vectorized LAS bridge to \n{out_json}.")
            obbox = extract_minimum_oriented_rectangle_shape(
                irregular_json, rectangular_json
            )
            save_geometries_to_geodataframe([obbox], rectangular_json)
            target_pickle = rectangular_json.replace(".json", ".pickle")
            targets = save_regression_targets_to_pickle(obbox, target_pickle)
            data = {"basename": basename}
            data.update(targets)
            df_targets = df_targets.append(targets, ignore_index=True)
    df_targets.to_csv(TARGETS_CSV, index=False)


def vectorize_bridge(points, out_json) -> None:
    """Vectorizes bridge points into a geojson."""
    out_tif = out_json.replace(".json", ".tif")
    pipeline = pdal.Filter.range(limits=f"Classification[{BRIDGE}:{BRIDGE}]").pipeline(
        points
    )
    pipeline.execute()
    points = pipeline.arrays[0]
    if len(points) == 0:
        # no building points in this LAS, so we create an empty geometry
        save_geometries_to_geodataframe([Polygon([])], out_json)
        return
    # else we rasterize the bridge points inot a TIF that we then vectorize
    pipeline = pdal.Writer.gdal(
        filename=out_tif,
        dimension="Classification",
        data_type="uint",
        output_type="max",
        window_size=GDAL_WRITER_WINDOW_SIZE,
        resolution=GDAL_WRITER_RESOLUTION,
        nodata=0,
        override_srs=LAMBERT_93_EPSG_STR,
    ).pipeline(points)
    pipeline.execute()
    gdal_polygonize(out_tif, out_json, epsg_out=None)


def extract_minimum_oriented_rectangle_shape(irregular_json, rectangular_json):
    center_square = make_centered_square(WIDTH_SUBTILE_CENTER_SQUARE)
    s = geopandas.read_file(irregular_json, bbox=center_square).dropna()
    # No bridge
    if len(s) == 0:
        return make_centered_square(0.0)

    # Find biggest area
    shape = s.iloc[s.area.argmax()].geometry
    obbox = shape.minimum_rotated_rectangle
    obbox = obbox.buffer(
        -0.25, cap_style=shapely.geometry.CAP_STYLE.square, resolution=2
    )
    return obbox


from shapely.geometry import Point
from shapely.ops import nearest_points


def save_regression_targets_to_pickle(obbox, target_pickle):
    targets = {k: 0.0 for k in ["Ax", "Ay", "Bx", "By", "D"]}
    if obbox.area:
        # calculate
        # x, y = obbox.exterior.coords.xy
        corners = [Point(o) for o in obbox.exterior.coords]
        # ignore last point which is equals to first point
        corners = corners[:-1]

        # select first point, and its closest neighboor, and calculate average
        # Rectantgular so the nearest point on the polygon can only be a corner.
        p1 = corners.pop(0)
        p2_idx = min(
            range(len(corners)),
            key=lambda other_p_idx: p1.distance(corners[other_p_idx]),
        )
        p2 = corners.pop(p2_idx)

        # remaining corners
        p3, p4 = corners

        A = average_two_points(p1, p2)
        B = average_two_points(p3, p4)
        targets["Ax"] = A.x
        targets["Ay"] = A.y
        targets["Bx"] = B.x
        targets["By"] = B.y
        targets["D"] = p1.distance(p2)
        assert abs(targets["D"] - p3.distance(p4)) <= 0.01

    with open(target_pickle, "wb") as f:
        pickle.dump(targets, f)

    return targets


def average_two_points(p1, p2):
    return Point(np.mean([p1.x, p2.x]), np.mean([p1.y, p2.y]))


# UTILS


def get_pdal_reader(las_path: str) -> pdal.Reader.las:
    """Standard Reader which imposes Lamber 93 SRS.
    Args:
        las_path (str): input LAS path to read.
    Returns:
        pdal.Reader.las: reader to use in a pipeline.
    """
    return pdal.Reader.las(
        filename=las_path,
        nosrs=True,
        override_srs=LAMBERT_93_EPSG_STR,
    )


def get_XY_centered_points(in_las):
    pipeline = pdal.Pipeline() | get_pdal_reader(in_las)
    pipeline.execute()
    points = pipeline.arrays[0]
    points["X"] = points["X"] - (points["X"].max() + points["X"].min()) / 2.0
    points["Y"] = points["Y"] - (points["Y"].max() + points["Y"].min()) / 2.0
    return points


def make_centered_square(width):
    return Polygon(
        [
            (-width / 2, -width / 2),
            (-width / 2, width / 2),
            (width / 2, width / 2),
            (width / 2, -width / 2),
        ]
    )


def save_geometries_to_geodataframe(geometry_list: List[Polygon], out_json: str):
    "Save a list of geometries to a geojson file.."
    s = geopandas.GeoDataFrame({"geometry": geometry_list})
    s.to_file(out_json)


# Rasterization and vectorization


def gdal_polygonize(
    fic_mask, fic_output, epsg_out=None, field_name="value", value_to_polygonize=0
):
    """Polygonisation des valeurs non-nulles d'un raster monobande."""

    src_ds = gdal.Open(fic_mask)
    srcband = src_ds.GetRasterBand(1)
    srcband2 = src_ds.GetRasterBand(1)

    srs = get_ogr_srs(epsg_out)
    ext_driver = get_ogr_driver_by_ext(fic_output)
    drv = ogr.GetDriverByName(ext_driver)

    if os.path.exists(fic_output):
        drv.DeleteDataSource(fic_output)

    dst_datasource = drv.CreateDataSource(fic_output)
    dst_layer = dst_datasource.CreateLayer(fic_output, srs)
    field_defn = ogr.FieldDefn(field_name, ogr.OFTString)
    dst_layer.CreateField(field_defn)

    gdal.Polygonize(
        srcband, srcband2, dst_layer, value_to_polygonize, ["8"], callback=None
    )


def get_ogr_srs(epsg):
    if epsg is not None:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(int(epsg))
        return srs
    else:
        return None


def get_ogr_driver_by_ext(file):
    _, file_extension = os.path.splitext(file)
    if file_extension == ".json" or file_extension == ".geojson":
        return "GeoJson"
    elif file_extension == ".shp":
        return "ESRI Shapefile"
    else:
        raise Exception("Extension de fichier vecteur non comprise")
        return 0


if __name__ == "__main__":
    main()
