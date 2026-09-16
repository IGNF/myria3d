# Linear probing in myria3d

Evaluates the quality of a trained Flair3D+ multitask backbone (`experiment=flair3d_plus/multitask`,
`PyGRandLANetMultiTask`) via **linear probing (LP)** on four downstream datasets already used for
the same purpose in the sibling Pointcept repo (`/data/geist/Pointcept`): **DALES, H3D, ECLAIR**
(per-point semantic segmentation) and **PureForest** (per-tile species classification). The
protocol and dataset definitions/splits/exclusions reproduce Pointcept's own linear-probing setup
(`pointcept/models/grid_probe.py`, `tools/grid_then_seeds.py`) as closely as the two backbones'
architectures allow — see the per-component docstrings for exactly where and why they diverge.

Training that produces the checkpoint being probed is out of scope here — see `readme_flair3d.md`.
Everything below assumes you already have a Flair3D+ multitask checkpoint (`MultiTaskModel`,
`.ckpt`).

---

## Protocol

Two-phase **sweep-LR-then-10-seeds** evaluation, run natively as myria3d Hydra code for
DALES/H3D/ECLAIR (not external orchestration scripts calling `run.py` repeatedly):

1. **Grid phase**: `GridProbeModel` trains 12 independent linear heads ("probes") sharing one
   frozen-backbone forward pass per step, one optimizer per probe, one shared `manual_backward()`
   over the summed per-probe losses (probes have disjoint parameters, so each still gets its own
   correct gradient). LR grid: `{1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1,
   5e-1}` (`configs/model/grid_probe/probe_lrs/lr_grid_12.yaml`). `GridProbeWinnerSelector` tracks
   each probe's best-epoch val metric and writes `grid_search_results.json` (leaderboard + winner).
2. **Seed phase**: the same multi-head trick, this time with `N_SEEDS` (10 by default) probes all
   at the winning LR — independent init comes for free from sequential `nn.Linear` construction
   order (the torch global RNG advances once per probe). `GridProbeSeedEnsembleTester` aggregates
   the post-fit `test` pass (mean/std/min/max per metric) into `seed_ensemble_results.json` +
   one row in `grid_then_seeds_summary.csv`.

Winner selection metric is dataset-specific, matching Pointcept exactly:

| Dataset | `select_metric` | `num_classes` | `ignore_index` | Val split |
|---|---|---|---|---|
| DALES  | mIoU     | 9  (incl. `Unclassified`) | 8    | **none — mirrors test** |
| H3D    | macro-F1 | 12 (incl. `Void`)         | 11   | yes |
| ECLAIR | mIoU     | 11 (no void class)        | null | yes |

**DALES has no val split upstream** (Pointcept's own DALES preprocessing only ever writes
`train`/`test`): `configs/datamodule/downstream/dales_datamodule.yaml` sets `val_dir: test`, i.e.
validation is scored against the same tiles as the final test — a deliberate, documented
convention, not a silently-mislabeled held-out set. Treat DALES val-based winner selection
accordingly (it is effectively selecting against test).

`num_classes` here **includes** the void/unclassified sentinel as a real (if `ignore_index`-
excluded) row of the confusion matrix, matching myria3d's own established convention elsewhere in
the repo (e.g. `flair3d_plus_multitask.yaml`'s `segment` task: 16 classes including `Void` at
index 15) — this differs from Pointcept's own grid-probe configs, which size the `ProbeHead`'s
output to the number of *real* classes only (DALES: 8, H3D: 11) and never predict the void class
at all. Both conventions are equally valid ways to handle an ignore-only sentinel; myria3d's stays
consistent with its own `ignore_index`-must-be-in-range assumption used throughout
`metric_callbacks.py`/`multitask_metric_callbacks.py`.

No Hydra multirun/sweeper plugin is used: the LR sweep happens by construction inside
`GridProbeModel` (12 heads, one run), not via `-m`/`--multirun`.

---

## Environment

```bash
export FLAIR3D_CKPT_PATH=/path/to/flair3d_plus_multitask.ckpt   # the checkpoint being probed
export DOWNSTREAM_DATA_ROOT=/data/geist/Pointcept/data           # parent of dales/h3d/eclair/pureforest
```

mirrors the `FLAIR3D_DATA_ROOT`/`FLAIR3D_CSV_MANIFEST` pattern from `readme_flair3d.md`. On Jean
Zay, point `DOWNSTREAM_DATA_ROOT` at wherever the Pointcept-preprocessed `.npy` copies live
(the JZ-side Pointcept checkout, not Hecate paths).

Expected per-scene layout (produced by Pointcept's own `pointcept/datasets/preprocessing/{dales,h3d,eclair}/preprocess_*.py`,
run outside myria3d): `{DOWNSTREAM_DATA_ROOT}/{dataset}/{split}/{scene_id}/coord.npy` plus
whatever subset of `color.npy` / `strength.npy` / `segment.npy` exists for that dataset.

---

## Running (DALES / H3D / ECLAIR)

```bash
# Grid phase only
python run.py experiment=linear_probe/dales_grid

# Seed phase (probe_lrs is a smoke-test placeholder in the committed config --
# override it with the grid phase's actual winner, see below)
python run.py experiment=linear_probe/dales_seeds \
  model.probe_lrs='[0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]'
```

Swap `dales` for `h3d` / `eclair` throughout. `python run.py` alone runs both `fit` and (per
`myria3d/train.py`'s standard `fit` flow) an automatic post-fit `test` pass — for the grid phase
that test pass is scored on whichever probe `GridProbeWinnerSelector` narrowed
`model.active_probe` to; for the seed phase it is scored with all `N_SEEDS` probes active, which
is what `GridProbeSeedEnsembleTester` aggregates.

### Jean Zay

See `scripts/jz/linear_probe/`:

```bash
export WEIGHT=/path/to/flair3d_plus_multitask.ckpt
export DOWNSTREAM_DATA_ROOT=/lustre/fsn1/projects/rech/unv/usi32yh/data_pointcept

# One job per phase
sbatch scripts/jz/linear_probe/run_grid.slurm dales
sbatch scripts/jz/linear_probe/run_seeds.slurm dales   # reads the grid phase's winning LR

# Or chained, requeue-safe (skips a phase whose result file already exists)
sbatch scripts/jz/linear_probe/run_grid_then_seeds.slurm dales   # or h3d / eclair
```

`N_SEEDS` (default 10) is also overridable via env var. Both phases pin `hydra.run.dir` to a
stable, dataset-specific path (`${LOGS_DIR}/runs/linear_probe/{dataset}/{grid,seeds}`) instead of
Hydra's default timestamped run dir — `@hydra.main` changes the process's working directory to
that path before running, so a script chaining two separate `python run.py` invocations needs a
predictable location to read `grid_search_results.json` back from.

**Manual smoke check** before trusting a full run: launch one grid-phase job for a couple hundred
iterations on one dataset (H3D is a reasonable first pick) and sanity-check that
`grid_search_results.json` picks a plausible winner LR before committing to the full grid-then-10-seeds
run on all three datasets.

### A note on `tile_width`

Pointcept's own GridProbe configs (`configs/{dales,h3d,eclair}/*-lin-grid*.py` in the sibling
repo) never crop these datasets by physical tile width at all — each preprocessed scene is
processed whole and bounded only by a point-count budget (`SphereCrop(point_max=...)`), because
their backbones (PTv3/SpUNet/...) don't need a fixed-extent input the way RandLA-Net's `SubtileCrop`
mosaic does. What each on-disk scene folder's actual physical extent is therefore depends entirely
on which `--chunking`/`--chunk-size` value was used when the JZ-side data was generated
(`preprocess_{dales,h3d}.py`), which is not discoverable from the Pointcept source alone:

- **ECLAIR**: `preprocess_eclair.py --chunking` defaults to `1`, which its own `--help` string
  confirms "keeps original 100×100 m tiles" — exactly Flair3D+'s own 100 m tile / 50 m subtile
  convention. `configs/datamodule/downstream/eclair_datamodule.yaml` sets `tile_width: 100`,
  `subtile_width: 50`, so `SubtileCrop` does a real 2×2 quadrant crop.
- **DALES / H3D**: no reliable native tile width to assume (`preprocess_dales.py --chunking`
  defaults to `3`, but the resulting per-folder extent depends on the raw DALES tile size, which
  itself may differ from what your JZ copy was generated with; `preprocess_h3d.py --chunk-size`
  defaults to `inf`, i.e. **no** tiling — one folder per source file, potentially large and
  irregular). `configs/datamodule/downstream/{dales,h3d}_datamodule.yaml` therefore set
  `tile_width: ${datamodule.subtile_width}`, making `SubtileCrop` a no-op (whatever is on disk is
  treated as the "subtile"), and rely on `MaximumNumNodes`
  (`points_budget_downstream.yaml`, capped at 40k points) as the actual size bound — the same
  point-count-based philosophy as Pointcept's `SphereCrop`, instead of a guessed physical width.

**Verify your actual JZ-side tile extents** before trusting results (e.g.
`np.load(f"{tile}/coord.npy")[:, :2].ptp(0)`), and override `datamodule.tile_width=<meters>` /
`datamodule.subtile_width=<meters>` on the CLI if DALES/H3D tiles turn out to be small and regular
enough for real `SubtileCrop` mosaicking to be worthwhile.

---

## PureForest (species classification — two-repo flow)

PureForest's protocol is intentionally **not** ported to native myria3d Hydra code: myria3d only
extracts and saves per-tile **mean+max pooled hypercolumn features** to a standalone script (not a
Hydra task), in exactly Pointcept's `.npz` schema, so the actual linear probe is trained back in
the sibling Pointcept repo on Hecate using its existing `scripts/probe_pureforest_sklearn.py`,
unmodified.

```bash
python scripts/extract_pureforest_pooled_features.py \
  --ckpt-path "$FLAIR3D_CKPT_PATH" \
  --data-root "$DOWNSTREAM_DATA_ROOT/pureforest" \
  --output-dir stats/pureforest_embeddings \
  --splits train val test --batch-size 8
```

or on Jean Zay: `sbatch scripts/jz/linear_probe/run_extract_pureforest_features.slurm`.

This writes `{split}.npz` per split with `names` (str array), `category` (int64),
`mean_feat`/`max_feat` (**float16** `[N, 928]`, computed from one concatenated 928-channel
hypercolumn vector — no per-scale keys), `class_names` (13 species), and metadata (`feat_dim`,
`channel_blocks`, `weight` — the probed checkpoint path, matching Pointcept's own
`extract_pureforest_pooled_embeddings.py` metadata key name — `data_root`, `split`). Copy the
output directory to Hecate and run:

```bash
python scripts/probe_pureforest_sklearn.py \
  --embeddings-dir /path/to/copied/pureforest_embeddings \
  --channel-blocks 32 128 256 512
```

`--channel-blocks` must be passed explicitly: `probe_pureforest_sklearn.py` normally
auto-resolves per-scale channel widths from `meta.json → Config.fromfile`, which assumes a
Pointcept-loadable config that myria3d's checkpoint doesn't have. The "9 scale slices" ablation
from the PureForest paper is done entirely downstream, by slicing the dense 928-channel vector
with `--scale-slice`; myria3d's extraction script does not need to bake scale boundaries into the
`.npz`.

**Test-split leakage exclusion**: any PureForest *test* tile whose forest polygon (`bdforetv2_id`)
geographically overlaps a Flair3D+ train/val tile is dropped, to prevent leakage into the LP test
set via the very backbone being probed — train/val are never filtered. The exclusion list
(`myria3d/pctl/dataset/downstream/pureforest_assets/flair3d_leakage_excluded_test_tiles.txt`,
~2.2k patch ids) is copied verbatim from the sibling Pointcept repo and applied automatically by
`PureForestDataset` for `split="test"`.

On-disk `coord.npy` is Pointcept's own pre-normalized convention (mean-centered XY, min-shifted Z,
divided by `COORD_SCALE_M=25.0`) — denormalized back to real meters on load, before myria3d's own
point-budget/normalization transforms run.

---

## Architecture notes

- **Multi-scale encoder hypercolumn** (`myria3d/models/modules/hypercolumn.py`): Pointcept's
  hypercolumn concat is an index-gather over `pooling_parent`/`pooling_inverse` bookkeeping that
  grid/voxel-pooling backbones track at pooling time. RandLA-Net's `decimate()` is a random
  per-cloud subsample with no such parent/child index, so myria3d's hypercolumn uses
  `knn_interpolate` (the same primitive `FPModule` already uses to upsample between stages)
  instead — mechanically different, same purpose. Concat order is finest-first: `enc1 (32ch) →
  enc2 (128ch) → enc3 (256ch) → enc4 (512ch)`, 928 channels total.
- **Missing-modality features use the backbone's own learned fill-in, not zero**: the pretrained
  backbone was trained with `learned_masked_feat: true`
  (`configs/model/multitask_default.yaml`) — `RandomDropColor`/`RandomDropStrength` randomly
  zeroed+substituted whole-scene color/intensity 20% of the time during pretraining, and a
  per-point mask selects a learned `nn.Parameter` fill value in their place. DALES has no color
  and H3D has no intensity upstream — `DownstreamNpyDataset` sets `color_mask`/`strength_mask` to
  **all-True** for the fully-absent modality, and `GridProbeModel`/the PureForest extraction
  script reuse the checkpoint's own frozen `color_mask_value`/`strength_mask_value` (loaded by
  `load_frozen_backbone`) instead of a hand-picked zero or mean value. ECLAIR has real color +
  intensity, so no masking applies there.
- **BatchNorm pinning**: the frozen backbone's `.train()` is overridden to always stay in eval
  mode (BatchNorm running stats frozen), even though Lightning calls `.train()` on the whole
  `GridProbeModel` every epoch — mirrors Pointcept's `bn_eval_mode`.
- The `finetune` task/callback path is dead code for this backbone (`FinetuningFreezeUnfreeze`
  calls methods that don't exist on `PyGRandLANet(MultiTask)`) — linear probing gets its own
  `GridProbeModel`, trained via the standard `fit` task, no `train.py` branching.

## Code map

| Component | File |
|---|---|
| Encoder-stage extraction | `myria3d/models/modules/pyg_randla_net_multitask.py::_forward_encoder_stages` |
| Hypercolumn concat | `myria3d/models/modules/hypercolumn.py` |
| Downstream datasets | `myria3d/pctl/dataset/downstream/{base,dales,h3d,eclair}.py` |
| Downstream datamodule | `myria3d/pctl/datamodule/downstream.py` |
| LP LightningModule | `myria3d/models/grid_probe_model.py` (`GridProbeModel`, `load_frozen_backbone`) |
| LP callbacks | `myria3d/callbacks/grid_probe_callbacks.py` |
| PureForest dataset + extraction | `myria3d/pctl/dataset/downstream/pureforest.py`, `scripts/extract_pureforest_pooled_features.py` |
| Configs | `configs/{dataset_description,datamodule/downstream,model/grid_probe,callbacks,experiment/linear_probe}/` |
| JZ orchestration | `scripts/jz/linear_probe/` |

## Tests

```bash
python -m pytest tests/myria3d/models/test_grid_probe_model.py \
  tests/myria3d/models/modules/test_hypercolumn.py \
  tests/myria3d/callbacks/test_grid_probe_callbacks.py \
  tests/myria3d/pctl/dataset/downstream/ \
  tests/myria3d/pctl/datamodule/test_downstream.py \
  tests/myria3d/scripts/test_extract_pureforest_pooled_features.py \
  tests/myria3d/test_linear_probe_hydra_composition.py -v
```
