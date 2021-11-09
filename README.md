<div align="center">

# Semantic Segmentation - Inspection Module

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>

[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/ashleve/lightning-hydra-template)
</div>
<br><br>

## Description
### Context
A fast and sensitive semantic segmentation of High Density Lidar data was performed with geometric rule-based algorithm to identify buildings. It yielded a high number of false positive. Around 160km² of Lidar data was thoroughly inspected to identify false positive and false negative. At larger scale, this kind of human inspection would be intractable.

### Objective
We train a semantic segmentation neural network to confirm or refute automatically the majority of "candidate" buildings points obtained from the rule-based algorithm, while also identifying cases of uncertainty for human inspection. This results in an output point cloud in which only a fraction of the candidate building points remain to be inspected. Inspection is facilitated through the production of an inspection shapefile in order to efficiently select and validate (or invalidate) candidate building points.

### Content

1) Training and evaluating of the model
2) Inference of a semantic segmentation
3) validation module decision process:
    1) Vectorization from candidate buildings points into candidate building shapes
    2) Decision:
        1) Confirmation, if the proportion of "confirmed" points within a candidate building shape is sufficient.
        2) Refutation, if the proportion of "refuted" points within a candidate building shape is sufficient.
        3) Uncertainty, elsewise: candidate building shapes are still identified for faster human inspection.
    3) Update of the point cloud based on those decisions

4) Multiobjective hyperparameter Optimization of the decision process (point-level and shape-level thresholds) to maximize decision accuracy and automation.

Current inspection shapefile might look like this:

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

Evaluate the model and get inference results on the validation dataset
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
Then, update variable `INSPECTION_SHAPEFILE_FOR_OPTIMIZATION` in [`.env`](.env) with the path to the inspection shapefile.

Run a multi-objectives optimization of decision threshold
```yaml
python run.py -m task=tune print_config=false hparams_search=thresholds_2max_confirm hydra.sweeper.n_jobs=3 hydra.sweeper.n_trials=100
python run.py -m task=tune print_config=false hparams_search=thresholds_2max_refute hydra.sweeper.n_jobs=3 hydra.sweeper.n_trials=100

```
The optimization maximizes two metrics: 1) proportion of automated decisions and 2) Decision accuracy, for the chosen decision (confirmation/refutation). 
You can then check optimization results and choose a set of thresholds among the Pareto solutions, then rerun the production of the inspection shapefile with the parameters.
