# Segmentation Validation Model [Early Stage Repo]

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

## Description
Situation : a semantic segmentation of High Density Lidar data is performed with geometric rule-based algorithm, whihc is fast to run and sensitive to actual buldings points. However, it yields a high number of false positive. The semantic segmentation must be fully audited to spot errors and missing information. At large scale, this becomes intractable.

This project propose to train a semantic segmentation neural network to confirm or refute automatically the majority of "candidate" buildings obtained from the rule-based algorithm, while also identifying cases of uncertainty for human inspection. We use ~160km² of High Density Lidar data which went trough torough human inspection (identifying most false positive and false negative).

In this repo is the code for:

1) Training and evaluating of the model
2) Inference of a semantic segmentation
3) validation module decision process:
  a) Vectorization from candidate buildings points into candidate building shapes
  b) Decision:
    i) Confirmation, if the proportion of "confirmed" points within a candidate building shape is sufficient.
    ii) Refutation, if the proportion of "refuted" points within a candidate building shape is sufficient.
    iii) Uncertainty, elsewise: candidate building shapes are still identified for faster human inspection.
  c) Update of the point cloud based on those decisions [To be done]
4) Multiobjective hyperparameter Optimization of the decision process (point-level and shape-level thresholds) to maximize decision accuracy and automation.

## How to run
Install dependencies
```yaml
# clone project
git clone https://github.com/CharlesGaydon/Segmentation-Validation-Model
cd Segmentation-Validation-Model

# [OPTIONAL] create conda environment (you may need to run lines manually as conda may not activate properly from bash script)
bash bash/setup_environment/setup_env.sh

# activate using
conda activate validation_module_gpu
```

Rename `.env_example` to `.env` and fill out `LOG PATH`, `DATAMODULE`, and `LOGGER` sections.

Train model with a specific experiment from [configs/experiment/](configs/experiment/)
```yaml
# default
python run.py experiment=PN_debug
```

Evaluate the model and get inference reuslts on the validation dataset
```yaml
# to evaluate and infer at the same time
python run.py experiment=PN_infer_val trainer.resume_from_checkpoint=/path/to/checkpoints.ckpt
# to log IoU without saving predictions to new LAS files 
python run.py experiment=PN_infer_val callbacks.save_preds.save_predictions=false trainer.resume_from_checkpoint=/path/to/checkpoints.ckpt
```
Then, update variable `PREDICTED_LAS_DIRPATH` in [`.env`](.env) with the directory containing inference results.

Make decisions and produce an inspection shapefile from predictions
```yaml

python run.py task=decide
```
Then, update variable `SHAPEFILE_VERSION_TO_TUNE` in [`.env`](.env) with the path to the inspection shapefile.

Run a multi-objectives optimization of decision threshold. The optimization maximizes three metrics: 1) proportion of automated decisions, 2) Refutation accuracy, and 3) Confirmation accuracy.
```yaml
python run.py -m task=tune hparams_search=thresholds_3_objectives hydra.sweeper.n_jobs=3 hydra.sweeper.n_trials=100
```

You can then check optimization results and choose a set of thresholds among the Pareto solutions, then rerun the production of the inspection shapefile with the parameters.
