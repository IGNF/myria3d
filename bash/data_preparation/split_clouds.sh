PIPELINE="/home/cgaydon/Segmentation-Validation-Model/bash/data_preparation/pipeline.json"
LASFILES_DIR="/var/data/cgaydon/data/202110_building_val/trainvaltest/"
for f in $(ls ${LASFILES_DIR}*.las);
do
	echo $f ;
	BASENAME=$(basename $f .las);
	echo $BASENAME;
	mkdir ${SPLITTED_DIR}/${BASENAME}/
	pdal translate --readers.las.override_srs="EPSG:2154" --readers.las.nosrs=true --writers.las.forward="all" --writers.las.extra_dims="all" --json $PIPELINE $f ${LASFILES_DIR}/split/${BASENAME}/${BASENAME}_SUB#.las ;
done;