source .env
mkdir ${LASFILES_DIR}split/
for f in $(ls ${LASFILES_DIR}*.las);
do
	echo $f ;
	BASENAME=$(basename $f .las);
	echo $BASENAME;
	mkdir ${LASFILES_DIR}split/${BASENAME}/;
	pdal translate --readers.las.override_srs="EPSG:2154" --readers.las.nosrs=true --writers.las.forward="all" --writers.las.extra_dims="all" --json $PIPELINE $f ${LASFILES_DIR}/split/${BASENAME}/${BASENAME}_SUB#.las ;
done;