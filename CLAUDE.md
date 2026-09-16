# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Myria3D is a PyTorch / PyTorch-Lightning / PyTorch-Geometric library for multiclass semantic
segmentation of large-scale aerial Lidar point clouds (built for IGN's French "Lidar HD"
project). It also hosts a **Flair3D+** extension (single-task and multitask training on
Flair3D-build data). Configuration is driven end-to-end by Hydra.

- `README.md` — project overview, points to hosted docs (https://ignf.github.io/myria3d/).
- `readme_flair3d.md` — the authoritative guide for Flair3D+ workflows (single-task, multitask,
  Pointcept-shortcut training, Hecate/Jean Zay paths, iter-limited schedule). Read it before
  touching anything under `flair3d`/`pointcept` experiment configs or datasets.
- `readme_linear_probing.md` — linear-probing a frozen Flair3D+ multitask backbone on
  DALES/H3D/ECLAIR/PureForest (sweep-LR-then-10-seeds protocol, JZ paths/env vars). Read it before
  touching `GridProbeModel`, `myria3d/pctl/dataset/downstream/`, or `experiment/linear_probe/*.yaml`.
- `docs/source/background/general_design.md` — the design rationale (why RandLA-Net, why
  subsample, why interpolate only at test/predict time, not during train/val).
- `docs/source/guides/development.md` — versioning/CI/CD conventions.

## Environment

Conda/mamba env named `myria3d`, defined in `environment.yml` (Python 3.10, PyTorch 2.1 +
CUDA 11.8, PyG 2.4, Lightning 2.0, pdal, gdal/rasterio, comet_ml, hydra-core 1.1).

```bash
mamba env create -f environment.yml
conda activate myria3d
```

A `.env` file at the repo root is loaded via `python-dotenv` before `fit`/`test`/`finetune`/
`create_hdf5` tasks (not for `predict`, which loads `trained_model_assets/placeholder.env`
instead). At minimum it must set `LOGS_DIR`:

```bash
LOGS_DIR="logs/"
```

## Common commands

Everything routes through `run.py`, a Hydra entry point. The task is selected via
`task.task_name=<fit|test|finetune|predict|create_hdf5>` (defaults to `fit`).

```bash
# Train (fit) with the default debug experiment
python run.py

# Train with a specific experiment config (configs/experiment/*.yaml)
python run.py experiment=RandLaNet_base_run_FR

# Quick single-batch sanity run (fast_dev_run)
python run.py debug=true

# Build an HDF5 cache from LAS/PLY tiles before training
python run.py experiment=<...> task.task_name=create_hdf5 \
  datamodule.split_csv_path=<csv> datamodule.hdf5_file_path=<path>

# Test a trained checkpoint
python run.py experiment=test model.ckpt_path=<ckpt>

# Finetune a checkpoint on a new dataset/config
python run.py experiment=<...> task.task_name=finetune model.ckpt_path=<ckpt>

# Predict on unseen LAS (uses trained_model_assets/ config+ckpt by default)
python run.py predict.src_las=<file_or_glob> predict.output_dir=<dir> \
  datamodule.epsg=<epsg> task.task_name=predict
```

Any config value can be overridden from the CLI (dotted path into the composed config), e.g.
`datamodule.batch_size=10`, `trainer.accelerator=gpu`, `logger=csv` (disables Comet).

Jean Zay slurm launch scripts live under `scripts/jz/` (referenced from `readme_flair3d.md`),
e.g. `sbatch scripts/jz/run_flair3d_plus_train.slurm`, with paths overridable via exported env
vars (`JZ_USER`, `SPLIT_CSV`, `HDF5_PATH`, ...) rather than editing the script.

### Debugging PyG collate crashes

`scripts/debug_scene_dtypes.py` finds the Flair3D+ multitask scene(s) that break PyG collate
(intermittent `torch.cat(): input types can't be cast...` on a DataLoader worker — caused by
one `Data` attribute having a different dtype/ndim across scenes in the same batch). Two
modes: `--mode collate` (default, fast — runs the real training dataloader with a
dtype/shape-checking collate wrapper and stops at the first mismatch) and `--mode scan`
(exhaustive, per-attribute dtype/shape report across a whole split, `--jobs` to parallelize).
Reach for this before hand-debugging a collate crash on the Pointcept-npy pipeline.

### Tests

```bash
python -m pytest -rA -v                 # full suite, from an activated myria3d env
python -m pytest tests/myria3d/models/test_model.py -v   # single file
python -m pytest -k test_predict_as_command -v           # single test
python -m pytest -m "not slow"                            # skip slow-marked tests
```

- `pyproject.toml` wires `pytest` to always collect from `tests/myria3d/` and enforces
  coverage (`--cov-fail-under 75`, HTML report to `htmlcov/`).
- `tests/conftest.py` provides `make_default_hydra_cfg(overrides=[...])` (composes
  `configs/config.yaml` the same way `run.py` does, for tests that call `train()`/`predict()`
  directly) and `run_hydra_decorated_command(...)` (invokes `run.py` as a subprocess via `sh`,
  for CLI-level tests). GPU-only tests are guarded with `tests.runif.RunIf(min_gpus=1)`.
- Toy data is generated on the fly by `myria3d.pctl.dataset.toy_dataset` rather than committed
  as large binary fixtures — check there before adding new LAS/PLY test fixtures.

### Lint / format

```bash
pre-commit run --all-files   # black, isort, flake8, prettier(yaml), basic hygiene hooks
python -m flake8             # what CI runs as the "linter" gate
black --line-length 99 .
isort .
```
Line length is 99 everywhere (black/isort/flake8 agree); flake8 ignores E203/E501.

### Docker / CI

CI (`.github/workflows/cicd.yaml`, self-hosted runner) builds the Docker image, runs the full
pytest suite inside it, runs two example `predict` invocations against fixed CICD assets, then
runs `flake8`. Only pushes to `main`/`staging-*` build+push the image and publish to PyPI. There
is no local Makefile target for this — mirror the docker/pytest/flake8 commands above if you
need to reproduce CI locally.

## Architecture

### Hydra config composition

`configs/config.yaml` is the root config; `run.py` composes it via `@hydra.main` (train/hdf5
tasks) or against `trained_model_assets/<predict_config>.yaml` (predict task, so inference is
reproducible from a frozen config bundled with a checkpoint). Config groups under `configs/`:
`datamodule`, `dataset_description` (feature/class definitions), `model`, `callbacks`, `logger`,
`trainer`, `task`, `predict`, `training_schedule`, `experiment` (full overrides for a specific
run — this is the group you add new entries to for a new experiment, e.g.
`configs/experiment/flair3d_plus/*.yaml`). `configs/hydra/default.yaml` controls run-dir/logging.
Almost every class is instantiated via `hydra.utils.instantiate` against `_target_` keys, so new
components (models, datamodules, callbacks, transforms) are added by writing the class and
wiring a corresponding YAML — not by editing `train.py`/`run.py`.

### Training/eval flow (`myria3d/train.py`)

`train(config)` instantiates datamodule, model, callbacks, logger, trainer from config, then
branches on `config.task.task_name`:
- `fit` → optional LR-finder → `trainer.fit(...)` → immediately followed by a `test` pass on the
  just-trained best checkpoint.
- `test` → `trainer.test(...)` against `config.model.ckpt_path`.
- `finetune` → reloads `Model` from `ckpt_path` with hparams overridden by the *new* config
  (except the `neural_net` architecture group, which is preserved), then fits from scratch
  (epoch 0, no `ckpt_path` passed to `trainer.fit`).

`myria3d/utils/training_schedule.py::resolve_training_schedule` runs first and, when
`config.total_iters` is set, converts a Pointcept-style "N optimizer steps" schedule into
`trainer.max_epochs`/`limit_train_batches`/`check_val_every_n_epoch`, and aligns
`ReduceLROnPlateau` stepping with the validation cadence. This mutates `config` in place — read
it before changing anything schedule- or callback-related for the Pointcept pipeline.

### Model layer (`myria3d/models/`)

- `model.py::Model` — the single-task `LightningModule`. Holds a `MODEL_ZOO` list and
  `get_neural_net_class(name)` factory (substring match on class name) so new architectures
  register themselves by being appended to `MODEL_ZOO`. Key subtlety: during `training`/`val`,
  loss and metrics are computed directly on the **subsampled** point cloud; only in `test`/
  `predict` mode does `forward()` KNN-interpolate logits back to the full-resolution cloud (see
  `general_design.md` for why — interpolation is ~5-10x more expensive per step).
- `modules/pyg_randla_net.py` — the custom PyG-native RandLA-Net backbone (variable-size point
  clouds, no fixed-N requirement, unlike the original third-party implementation this replaced).
- `multitask_model.py` / `modules/pyg_randla_net_multitask.py` — Flair3D+ multitask variant:
  one shared backbone with per-task heads. The Pointcept-npy recipe is `segment` (semantic),
  `forest_2d` / `roads` (`pixel_semantic`, pooled to raster cells), four `nathab_*`
  (`tile_distribution`, WeightedKL over the tile), and `elevation` (regression against
  `z_lidar - DTM`). `MODEL_ZOO` is shared (appended to, not replaced) with the single-task
  model. KNN-interpolation is used for per-point heads only; pooled tasks stay at the
  model's native resolution. Per-task loss weights are auto-balanced by
  `gradnorm_lite.py` (EMA-based GradNorm-lite, ported from Pointcept — see the module
  docstring for what was trimmed vs. the reference impl); `losses.py` holds the
  tile-distribution pooling/loss helpers and `dilated_metrics.py` the buffer
  precision/recall/F1 used for thin pixel-mask tasks (`forest_2d`/`roads`).
- `interpolation.py` — the interpolator used at predict time to go from subsampled to full LAS.

### Data layer (`myria3d/pctl/` — "Point Cloud Transform Library")

- `dataset/hdf5.py` — builds/reads the HDF5 cache: LAS tiles are cut into a fixed mosaic of
  subtiles (`dataset/utils.py::get_mosaic_of_centers`, `tile_width`/`subtile_width`/
  `subtile_overlap_*`), optionally enriched with raster labels, then written per split.
  `create_hdf5` is the entry point invoked by `task.task_name=create_hdf5`.
- `dataset/pointcept_npy.py` + `datamodule/pointcept_npy.py` — alternate loader that reads
  tiles already preprocessed by an external [Pointcept](https://github.com/Pointcept/Pointcept)
  pipeline (`coord.npy`/`color.npy`/... + multitask label arrays) and crops 50m subtiles from
  100m tiles on the fly instead of relying on a pre-built HDF5. Used by
  `experiment=flair3d_plus/multitask`. The cropping itself is `transforms.py::SubtileCrop`:
  a fixed quadrant for val/test, but for train it tries the 4 quadrants in random order and
  takes the first with at least `min_points`, so a tile whose points sit in only 1-2
  quadrants still yields a usable sample instead of a wasted near-empty one (this behavior —
  and the collate dtype pitfalls of degenerate low-point crops — has been the site of several
  recent bugfixes; see git log on this file before changing it).
- `dataset/flair3d.py`, `flair3d_label_remap.py`, `raster_utils.py` — Flair3D+-specific PLY
  label handling and GeoTIFF raster sampling (once per patch, before subtiling — changing this
  logic requires regenerating the HDF5 cache).
- `dataloader/iter_limited_sampler.py` — sampler backing the Pointcept-compatible iter-limited
  schedule (`training_schedule.py` above).
- `transforms/` — `CustomCompose` plus the transform pipeline itself: `preparations` (budget /
  fixed-point sampling), `augmentations` (`none`/`light`/`heavy`), `normalizations`. Composed
  per-split (train/eval/predict) by the datamodule from `configs/datamodule/transforms/`.
- `points_pre_transform/lidar_hd.py` — raw LAS array → PyG `Data` object with the Lidar-HD
  feature schema (must match a `dataset_description` config).

### Callbacks (`myria3d/callbacks/`)

Metric computation is deliberately kept out of the `LightningModule` and lives in callbacks:
`metric_callbacks.py` (single-task IoU etc.), `multitask_metric_callbacks.py` (per-task
IoU/MAE/RMSE/KL, e.g. `val/iou_forest_2d`, `val/kl_nathab_habitat_type`, `val/elevation_mae`), `finetuning_callbacks.py` (swaps the
output layer's class count after loading finetune weights), `comet_callbacks.py`,
`pointcept_pred_dump.py` (writes `roads`/`forest_2d` pixel-semantic predictions back out in
Pointcept's `{patch_id}_logits_network.npy` format — scattered to full-patch raster resolution
and nanmean-merged across overlapping 50m quadrants / DDP ranks — so Pointcept's own external
eval scripts can consume myria3d's predictions).

### Predict flow (`myria3d/predict.py`, `run.py::launch_predict`)

`run.py` globs `predict.src_las` and calls `predict(config)` once per file. Inference always
runs against a config+checkpoint pair — by default `trained_model_assets/` (a frozen,
versioned pair meant for production use), but any `--config-path`/`--config-name` +
`predict.ckpt_path` combination works (see the CICD workflow for the two supported invocation
styles, including inference-time subtile overlap for smoothing: `predict.subtile_overlap`).

## Conventions

- Config groups are the extension point: prefer adding a new YAML under the relevant
  `configs/<group>/` directory (and an `experiment/*.yaml` that composes them) over branching in
  Python on a flag.
- `MODEL_ZOO` in `models/model.py` is the registry for neural net architectures; multitask
  extends it via `models/multitask_model.py` rather than defining a separate zoo.
-   Flair3D-related config paths still say `v12` in places for historical reasons even where
  `label=v20` is the current Flair3D-build label version — see `readme_flair3d.md`.
