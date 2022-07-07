import geopandas
import os

import pandas as pd

from myria3d.data.loading import _find_file_in_dir

MAX_IMPORTANCE = 3
GEOJSON = "/var/data/cgaydon/mnt/store-lidarhd/projet-LHD/IA/Multiclass-Segmentation/data/20220630_ponts_bdtopo/vignettesjeuEntrainement_pontsBDTopo.geojson"
INPUT_SPLIT_CSV = "/var/data/cgaydon/mnt/store-lidarhd/projet-LHD/IA/Multiclass-Segmentation/data/20220630_ponts_bdtopo/split.txt"
OUTPUT_SPLIT_CSV = f"/var/data/cgaydon/mnt/store-lidarhd/projet-LHD/IA/Multiclass-Segmentation/data/20220630_ponts_bdtopo/split.imp_leq_{MAX_IMPORTANCE}.csv"

# load geojson
df = geopandas.read_file(GEOJSON)
# ignore NAN by setting high value (i.e. low importance)
df["_imp_route"] = df["_imp_route"].fillna(666).astype(int)
# filter by max importance 3, 4
print(len(df))
df = df[df["_imp_route"] <= MAX_IMPORTANCE]
print(len(df))
selection = df["nom"].values

df = pd.read_csv(INPUT_SPLIT_CSV)
df = df[df["basename"].isin(selection)]
df.to_csv(OUTPUT_SPLIT_CSV, index=False)
