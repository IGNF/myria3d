# source activate train.sh
screen -S forest
git checkout forest-classification-explo
source activate myria3d

DATASET_NAME="PureForestID"

# List the data
DATA_DIR_PATH="/mnt/store-lidarhd/projet-LHD/IA/BDForet/Data/PureForestID/lidar/" # se termine avec un slash
cd $DATA_DIR_PATH
SPLIT_CSV_PATH="/home/CGaydon/repositories/myria3d/${DATASET_NAME}-split.csv"
echo "basename,split" >$SPLIT_CSV_PATH
find ./train -type f -printf "%f,train\n" >>$SPLIT_CSV_PATH
find ./val -type f -printf "%f,val\n" >>$SPLIT_CSV_PATH
find ./test -type f -printf "%f,test\n" >>$SPLIT_CSV_PATH
head $SPLIT_CSV_PATH
tail $SPLIT_CSV_PATH

# Create a "mini" version of the dataset using a different split.
# 1/100 file. Sinon biais dans l'ordre des fichiers.
SPLIT_CSV_PATH_MINI="/home/CGaydon/repositories/myria3d/${DATASET_NAME}-mini-split.csv"
echo "basename,split" >$SPLIT_CSV_PATH_MINI
awk 'NR % 1000 == 0' $SPLIT_CSV_PATH >>$SPLIT_CSV_PATH_MINI

head $SPLIT_CSV_PATH_MINI
tail $SPLIT_CSV_PATH_MINI
# Use of dataset or the other

python /home/$USER/repositories/myria3d/run.py \
    task.task_name=fit \
    datamodule.hdf5_file_path="/var/data/CGaydon/myria3d_datasets/PureForestID.hdf5" \
    dataset_description=20231025_forest_classification_explo \
    datamodule.tile_width=50 \
    experiment=RandLaNet_base_run_FR-MultiGPU \
    logger.comet.experiment_name="${DATASET_NAME}" \
    datamodule.data_dir=${DATA_DIR_PATH} \
    datamodule.split_csv_path="${SPLIT_CSV_PATH}" \
    trainer.gpus=[0,2]
# trainer.min_epochs=150 \
# trainer.max_epochs=300
