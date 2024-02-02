# prepare the new dataset
conda activate myria3d
python /home/$USER/repositories/myria3d/run.py \
    experiment=RandLaNet_base_run_FR \
    task.task_name=create_hdf5 \
    datamodule.hdf5_file_path="/var/data/CGaydon/myria3d_datasets/PureForestV2-single-tree-clean-trees-Train_Val_UnionValTest.hdf5" \
    dataset_description=20231025_forest_classification_explo \
    datamodule.tile_width=50 \
    datamodule.data_dir=/mnt/store-lidarhd/projet-LHD/IA/BDForet/Data/PureForestV2/lidar-single-tree-pdal/ \
    datamodule.split_csv_path="/home/CGaydon/repositories/myria3d/PureForestV2-split-train-val-valtest.csv"
