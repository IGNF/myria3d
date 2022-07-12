import shutil
import geopandas
import os

import pandas as pd

from myria3d.data.loading import _find_file_in_dir

MAX_IMPORTANCE = 3
GEOJSON = "/var/data/cgaydon/mnt/store-lidarhd/projet-LHD/IA/Multiclass-Segmentation/data/20220630_ponts_bdtopo/vignettesjeuEntrainement_pontsBDTopo.geojson"

INPUT_DATA_DIR = "/var/data/cgaydon/mnt/store-lidarhd/projet-LHD/IA/Multiclass-Segmentation/data/20220630_ponts_bdtopo/data/"
OUTPUT_DATA_DIR = f"/var/data/cgaydon/mnt/store-lidarhd/projet-LHD/IA/Multiclass-Segmentation/data/20220630_ponts_bdtopo/variations/imp_leq_{MAX_IMPORTANCE}/"

# prepare folders
shutil.rmtree(OUTPUT_DATA_DIR)
OUTPUT_DATA_DIR_TRAIN = os.path.join(OUTPUT_DATA_DIR, "train")
OUTPUT_DATA_DIR_VAL = os.path.join(OUTPUT_DATA_DIR, "val")
os.makedirs(OUTPUT_DATA_DIR_TRAIN, exist_ok=True)
os.makedirs(OUTPUT_DATA_DIR_VAL, exist_ok=True)


# load geojson
df = geopandas.read_file(GEOJSON)
# ignore NAN by setting high value (i.e. low importance)
df["_imp_route"] = df["_imp_route"].fillna(666).astype(int)
# filter by max importance
print(len(df))
df = df[df["_imp_route"] <= MAX_IMPORTANCE]
print(len(df))
selection = df["nom"].values


# copy the selected files
for basename in selection:
    f = _find_file_in_dir(INPUT_DATA_DIR, basename)
    output_dir_phase = OUTPUT_DATA_DIR_TRAIN
    if "20220630_ponts_bdtopo/data/val/" in f:
        output_dir_phase = OUTPUT_DATA_DIR_VAL
    f_out = os.path.join(output_dir_phase, basename)
    shutil.copy(f, f_out)


# save resulting split.txt
INPUT_SPLIT_CSV = os.path.join(INPUT_DATA_DIR, "split.txt")
OUTPUT_SPLIT_CSV = os.path.join(OUTPUT_DATA_DIR, "split.txt")

df = pd.read_csv(INPUT_SPLIT_CSV)
df = df[df["basename"].isin(selection)]
df.to_csv(OUTPUT_SPLIT_CSV, index=False)
