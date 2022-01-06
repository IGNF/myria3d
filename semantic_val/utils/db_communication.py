import os
import subprocess
from typing import NamedTuple
import geopandas


class ConnectionData(NamedTuple):
    """information to connect to the database"""

    host: str
    user: str
    pwd: str
    bd_name: str


def db_communication(
    db_cd: ConnectionData,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    srid: int,
    shapefile_path: str,
):
    """
    Create a shapefile with non destructed building on
    the area and saves it. Also add a column "presence" with only 1 in it
    """
    sql_request = f"SELECT st_setsrid(batiment.geometrie,{srid}) AS geometry, 1 as presence  FROM batiment WHERE batiment.geometrie && ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, {srid}) and not gcms_detruit"
    cmd = [
        "pgsql2shp",
        "-f",
        shapefile_path,
        "-h",
        db_cd.host,
        "-u",
        db_cd.user,
        "-P",
        db_cd.pwd,
        db_cd.bd_name,
        sql_request,
    ]
    # This call may yield
    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        # In empty zones, pgsql2shp does not create a shapefile
        if e.output == b'Initializing... \nERROR: Could not determine table metadata (empty table)\n':
            return False

    # read & write to avoid unnacepted 3D shapefile format.
    gdf = geopandas.read_file(shapefile_path)
    gdf[["PRESENCE", "geometry"]].to_file(shapefile_path)

    return True
