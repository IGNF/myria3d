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
python run.py experiment=PN_validate trainer.resume_from_checkpoint=/path/to/checkpoints.ckpt fit_the_model=false test_the_model=true
# to log IoU without saving predictions to new LAS files 
python run.py experiment=PN_validate callbacks.save_preds.save_predictions=false trainer.resume_from_checkpoint=/path/to/checkpoints.ckpt fit_the_model=false test_the_model=true
```
To evaluate on test data instead of val data, replace `experiment=PN_validate` by `experiment=PN_test`.

Then, update variable `PREDICTED_LAS_DIRPATH` in [`.env`](.env) with the directory containing inference results.

Make decisions and produce an inspection shapefile from predictions
```yaml

python run.py task=decide
```
Then, update variable `INSPECTION_SHAPEFILE_FOR_OPTIMIZATION` in [`.env`](.env) with the path to the inspection shapefile.

Without changing any parameters, evaluate the decision results with

```yaml
python run.py task=tune
```

Run a multi-objectives optimization of decision threshold, to maximize sensitivity and specificity directly while also maximizing automation:
```yaml
python run.py -m task=tune print_config=false hparams_search=thresholds_sensitivity_specificity_automation +inspection.metrics=[PROPORTION_OF_AUTOMATED_DECISIONS,SENSITIVITY,SPECIFICITY]
```
Alternatively, focus on a single decision at a time, to better understand the automation-error balance.
```yaml
python run.py -m task=tune print_config=false hparams_search=thresholds_2max_confirm +inspection.metrics=[PROPORTION_OF_CONFIRMATION,CONFIRMATION_ACCURACY]
python run.py -m task=tune print_config=false hparams_search=thresholds_2max_refute +inspection.metrics=[PROPORTION_OF_REFUTATION,REFUTATION_ACCURACY]
```

You can then check optimization results and choose a set of thresholds among the ones of the Pareto-front. See related notebooks for plotting. Then rerun the production of the inspection shapefile with the parameters.

Then use optimized threshold to produce final inspection shapefile and update las with predictions.
```yaml

python run.py task=decide inspection.min_confidence_confirmation=X inspection.min_frac_confirmation=X inspection.min_confidence_refutation=X inspection.min_frac_refutation=X inspection.update_las=true
```