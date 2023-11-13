# source activate train.sh
screen -S forest

git checkout forest-classification-explo
source activate myria3d

DATASET_NAME="PureForestID"
export LD_LIBRARY_PATH="/var/data/mambaforge-shared/envs/myria3d/lib:$LD_LIBRARY_PATH"
# List the data
DATA_DIR_PATH="/mnt/store-lidarhd/projet-LHD/IA/BDForet/Data/PureForestID/lidar/" # se termine avec un slash
SPLIT_CSV_PATH="/home/CGaydon/repositories/myria3d/${DATASET_NAME}-split.csv"
cd $DATA_DIR_PATH
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

# Test the first model :
MODEL_CHECKPOINT="/mnt/store-lidarhd/projet-LHD/IA/MYRIA3D-SHARED-WORKSPACE/CGaydon/runs/2023-10-25/16-52-57/checkpoints/epoch_000.ckpt"
python /home/$USER/repositories/myria3d/run.py \
    task.task_name=test \
    model.ckpt_path=$MODEL_CHECKPOINT \
    datamodule.hdf5_file_path="/var/data/CGaydon/myria3d_datasets/PureForestID.hdf5" \
    dataset_description=20231025_forest_classification_explo \
    experiment=RandLaNet_base_run_FR-MultiGPU \
    logger.comet.experiment_name="TEST-${DATASET_NAME}" \
    trainer.gpus=[0,2]

# Fine-Tune a pretrained model
python /home/$USER/repositories/myria3d/run.py \
    task.task_name=finetune \
    model.ckpt_path="/mnt/store-lidarhd/projet-LHD/IA/MYRIA3D-SHARED-WORKSPACE/CGaydon/20230930_60k_basic_targetted/20230930_60k_basic_targetted_epoch37_Myria3DV3.4.0.ckpt" \
    datamodule.hdf5_file_path="/var/data/CGaydon/myria3d_datasets/PureForestID.hdf5" \
    dataset_description=20231025_forest_classification_explo \
    datamodule.tile_width=50 \
    experiment=RandLaNet_base_run_FR-MultiGPU-Finetuning \
    logger.comet.experiment_name="${DATASET_NAME}-Finetuning" \
    trainer.gpus=[0,2]

# Prepare and train on geometric features, using list datamodule to be faster
python /home/$USER/repositories/myria3d/run.py \
    experiment=RandLaNet_base_run_FR-Geometric \
    task.task_name=fit \
    dataset_description=20231025_forest_classification_explo_geometric \
    datamodule.tile_width=50 \
    dataset_description.d_in=16 \
    logger.comet.experiment_name="${DATASET_NAME}-Geometric" \
    datamodule.data_dir=${DATA_DIR_PATH} \
    datamodule.split_csv_path="${SPLIT_CSV_PATH}" \
    trainer.gpus=[1]
