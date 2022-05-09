import laspy
from glob import glob

import numpy as np

uniques = []
for f in glob(
    "/var/data/cgaydon/mnt/store-lidarhd/projet-LHD/IA/Multiclass-Segmentation/data/20220504_proto23dalles/*.las"
):
    las = laspy.read(f)
    u = np.unique(las.classification).tolist()
    print(f"{f} : {u}")
    uniques += u
print(np.unique(uniques))
