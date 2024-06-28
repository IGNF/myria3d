<div align="center">

# Myria3D: Aerial Lidar HD Semantic Segmentation with Deep Learning
# Fork adapted to train/infer on non-colorized data
</div>

Myria3D is a deep learning library designed with a focused scope: the multiclass semantic segmentation of large scale, high density aerial Lidar points cloud.

This fork includes option to train on Lidar HD data without RGB attributes. It also implements the PointNet baseline. Two pretrained models are available, RandLaNet and PointNet, in the form of .ckpt files with the best version of the model. The models were trained on the same twelve Lidar HD tiles, list of the tiles trained upon is in trained_model_assets/lidarhd_dataset_split.csv. Training metrics can be observed on Comet: https://shorturl.at/QHZVd

The training was performed on a laptop with 3070Ti GPU (8GB VRAM), 32 GB RAM and i7-12700H. Batch sizes were adapted to the specifications.
PointNet implementation subsamples the 50x50m tiles to 4096 points, then upsamples with 1-nn. RandLaNet remains unchanged.

To train PointNet:

`python run.py experiment=PointNet_baseline`


To train RandLaNet:

`python run.py experiment=RandLaNet_base_run_FR`


To infer PointNet model:

`python run.py task.task_name='predict' predict.src_las='/path/to/las' predict.output_dir='/path/to/output' datamodule.epsg=2154 predict.ckpt_path='${hydra:runtime.cwd}/trained_model_assets/randlanet_norgb_epoch_028.ckpt' trainer.accelerator=gpu predict.gpus=[0]`

(to achieve better results add `predict.subtile_overlap=25`)


To infer RandLaNet model:

`python run.py task.task_name='predict' predict.src_las='/path/to/las' predict.output_dir='/path/to/output' datamodule.epsg=2154 predict.ckpt_path='${hydra:runtime.cwd}/trained_model_assets/pointnet_norgb_epoch_020.ckpt' trainer.accelerator=gpu predict.gpus=[0]`
___

# Comparisons
<p float="left">
  ----------------Ground Truth-------------------------------
  PointNet----------------------------------
  RandLaNet---------------
</p>
<p float="left">
  <img src="https://github.com/Vynikal/myria3d/blob/lidar_hd/im/ex1/gt.png?raw=true" width="250" />
  <img src="https://github.com/Vynikal/myria3d/blob/lidar_hd/im/ex1/pointnet.png?raw=true" width="250" /> 
  <img src="https://github.com/Vynikal/myria3d/blob/lidar_hd/im/ex1/randlanet.png?raw=true" width="250" />
</p>
<p float="left">
  <img src="https://github.com/Vynikal/myria3d/blob/lidar_hd/im/ex2/gt.png?raw=true" width="250" />
  <img src="https://github.com/Vynikal/myria3d/blob/lidar_hd/im/ex2/pointnet.png?raw=true" width="250" /> 
  <img src="https://github.com/Vynikal/myria3d/blob/lidar_hd/im/ex2/randlanet.png?raw=true" width="250" />
</p>
<p float="left">
  <img src="https://github.com/Vynikal/myria3d/blob/lidar_hd/im/ex3/gt.png?raw=true" width="250" />
  <img src="https://github.com/Vynikal/myria3d/blob/lidar_hd/im/ex3/pointnet.png?raw=true" width="250" /> 
  <img src="https://github.com/Vynikal/myria3d/blob/lidar_hd/im/ex3/randlanet.png?raw=true" width="250" />
</p>

Please cite Myria3D if it helped your own research. Here is an example BibTex entry:
```
@misc{gaydon2022myria3d,
  title={Myria3D: Deep Learning for the Semantic Segmentation of Aerial Lidar Point Clouds},
  url={https://github.com/IGNF/myria3d},
  author={Charles Gaydon},
  year={2022},
  note={IGN (French Mapping Agency)},
}
```
