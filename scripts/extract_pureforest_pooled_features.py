#!/usr/bin/env python3
"""Extract per-tile mean/max-pooled frozen-backbone hypercolumn features on PureForest.

Plain script, not a Hydra task: PureForest's actual linear probe is trained back in
the sibling Pointcept repo (on Hecate), via its existing
``scripts/probe_pureforest_sklearn.py`` -- this script only extracts and saves
features, in exactly Pointcept's ``{split}.npz`` schema, so the ``.npz`` files can be
copied there and read unmodified (pass ``--channel-blocks 32 128 256 512`` explicitly
on the Pointcept side; myria3d has no ``Config.fromfile``-loadable config for that
script's usual meta.json auto-resolution path). See readme_linear_probing.md.

For each tile: one frozen-backbone forward (deterministic, no augmentation) ->
``_forward_encoder_stages`` + ``hypercolumn_concat`` -> per-point hypercolumn feature
-> mean pool (float32 accumulation, nan_to_num-safe) and max pool (native dtype),
both cast to float16 for storage.

Usage::

    python scripts/extract_pureforest_pooled_features.py \\
        --ckpt-path /path/to/flair3d_plus_multitask.ckpt \\
        --data-root data/pureforest \\
        --output-dir stats/pureforest_embeddings \\
        --splits train val test --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Center
from torch_scatter import scatter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torch_geometric.transforms import GridSampling  # noqa: E402

from myria3d.models.grid_probe_model import load_frozen_backbone  # noqa: E402
from myria3d.models.modules.hypercolumn import (  # noqa: E402
    DEFAULT_SCALES,
    STAGE_CHANNELS,
    hypercolumn_concat,
)
from myria3d.pctl.dataset.downstream.pureforest import (  # noqa: E402
    CLASS_NAMES,
    PureForestDataset,
)
from myria3d.pctl.transforms.compose import CustomCompose  # noqa: E402
from myria3d.pctl.transforms.transforms import (  # noqa: E402
    NormalizePos,
    NullifyLowestZ,
    StandardizeRGBAndIntensity,
    resolve_x_feature_names,
)

CHANNEL_BLOCKS = tuple(STAGE_CHANNELS[s] for s in DEFAULT_SCALES)
FEAT_DIM = sum(CHANNEL_BLOCKS)


def build_extract_transform(voxel: float) -> CustomCompose:
    """Deterministic, augmentation-free pipeline matching how the backbone was
    pretrained (Center/NullifyLowestZ/NormalizePos/StandardizeRGBAndIntensity, same
    voxel size as `configs/datamodule/transforms/preparations/points_budget_downstream.yaml`).
    PureForest tiles are already ~50 m single plots (see readme_linear_probing.md) --
    no SubtileCrop needed."""
    return CustomCompose(
        [
            GridSampling(voxel),
            Center(),
            NullifyLowestZ(),
            NormalizePos(subtile_width=50),
            StandardizeRGBAndIntensity(),
        ]
    )


def fill_strength_mask_value(
    x: torch.Tensor, names: List[str], strength_mask_value
) -> torch.Tensor:
    """PureForest has no intensity asset at all (`strength_mask` is all-True, see
    `load_pureforest_scene`) -- every point's Intensity column must come from the
    frozen backbone's own learned fill-in value, not a hand-picked zero. Mirrors
    `GridProbeModel._fill_masked_features`'s strength-only case."""
    if strength_mask_value is None or "Intensity" not in names:
        return x
    idx = names.index("Intensity")
    fill = strength_mask_value.to(dtype=x.dtype, device=x.device)[:, 0].reshape(())
    out = x.clone()
    out[:, idx] = fill
    return out


@torch.no_grad()
def extract_tile_features(backbone, batch: Batch, mask_values: dict) -> torch.Tensor:
    names = resolve_x_feature_names(batch)
    x = fill_strength_mask_value(batch.x, names, mask_values.get("strength_mask_value"))
    stages = backbone._forward_encoder_stages(x, batch.pos, batch.batch, batch.ptr)
    return hypercolumn_concat(stages, batch.pos, batch.batch, scales=DEFAULT_SCALES, k=1)


def pool_mean_max(feat: torch.Tensor, batch_index: torch.Tensor, num_graphs: int):
    mean_feat = scatter(feat.float(), batch_index, dim=0, dim_size=num_graphs, reduce="mean")
    mean_feat = torch.nan_to_num(mean_feat, nan=0.0, posinf=0.0, neginf=0.0)
    max_feat = scatter(feat, batch_index, dim=0, dim_size=num_graphs, reduce="max")
    return mean_feat, max_feat


def run_split(
    *,
    backbone,
    mask_values: dict,
    dataset: PureForestDataset,
    transform: CustomCompose,
    batch_size: int,
    device: torch.device,
) -> dict:
    names: List[str] = []
    categories: List[int] = []
    mean_chunks: List[np.ndarray] = []
    max_chunks: List[np.ndarray] = []

    n = len(dataset)
    for start in range(0, n, batch_size):
        data_list = []
        batch_names = []
        batch_cats = []
        for idx in range(start, min(start + batch_size, n)):
            data = dataset[idx]
            data = transform(data)
            if data is None or data.num_nodes == 0:
                print(f"[extract] skipping empty tile after transform: {dataset.names[idx]}")
                continue
            data_list.append(data)
            batch_names.append(data.patch_id)
            batch_cats.append(int(data.category))
        if not data_list:
            continue

        batch = Batch.from_data_list(data_list).to(device)
        feat = extract_tile_features(backbone, batch, mask_values)
        mean_feat, max_feat = pool_mean_max(feat, batch.batch, len(data_list))

        mean_chunks.append(mean_feat.cpu().numpy().astype(np.float16))
        max_chunks.append(max_feat.cpu().numpy().astype(np.float16))
        names.extend(batch_names)
        categories.extend(batch_cats)
        print(f"[extract] {dataset.split}: {min(start + batch_size, n)}/{n}")

    return {
        "names": np.asarray(names),
        "category": np.asarray(categories, dtype=np.int64),
        "mean_feat": np.concatenate(mean_chunks, axis=0),
        "max_feat": np.concatenate(max_chunks, axis=0),
    }


def save_split_npz(output_path: Path, payload: dict, meta: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        names=payload["names"],
        category=payload["category"],
        mean_feat=payload["mean_feat"],
        max_feat=payload["max_feat"],
        class_names=np.asarray(CLASS_NAMES),
        **{k: np.asarray(v) for k, v in meta.items()},
    )
    size_mb = output_path.stat().st_size / 2**20
    n, c = payload["mean_feat"].shape
    print(f"[extract] wrote {output_path}  ({n:,} tiles x {c}ch mean/max, {size_mb:.1f} MB)")


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ckpt-path", required=True, help="Flair3D+ multitask checkpoint.")
    parser.add_argument("--data-root", required=True, help="e.g. data/pureforest")
    parser.add_argument(
        "--output-dir", required=True, help="Directory for {split}.npz + meta.json"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"]
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--voxel", type=float, default=0.1)
    parser.add_argument("--num-features", type=int, default=5)
    parser.add_argument("--num-neighbors", type=int, default=16)
    parser.add_argument("--decimation", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda / cpu (default: auto).")
    parser.add_argument("--max-tiles", type=int, default=None, help="Smoke-test cap per split.")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[extract] device={device}  ckpt={args.ckpt_path}  channel_blocks={CHANNEL_BLOCKS}")

    backbone, mask_values = load_frozen_backbone(
        args.ckpt_path,
        num_features=args.num_features,
        num_neighbors=args.num_neighbors,
        decimation=args.decimation,
    )
    backbone = backbone.to(device)

    transform = build_extract_transform(args.voxel)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_common = {
        # "weight" (not "ckpt_path"): matches Pointcept's own
        # extract_pureforest_pooled_embeddings.py metadata key name.
        "weight": str(args.ckpt_path),
        "data_root": str(args.data_root),
        "feat_dim": FEAT_DIM,
        "channel_blocks": list(CHANNEL_BLOCKS),
        "voxel": args.voxel,
        "class_names": CLASS_NAMES,
    }

    for split in args.splits:
        dataset = PureForestDataset(
            args.data_root,
            split,
            # Test-only leakage filter (fact 9): drop test tiles whose forest polygon
            # overlaps a Flair3D+ train/val tile, never applied to train/val -- matches
            # every Pointcept PureForest config (`exclude_flair3d_leakage_tiles=True`
            # set only on `data.test`).
            exclude_flair3d_leakage_tiles=(split == "test"),
        )
        if args.max_tiles is not None:
            dataset.names = dataset.names[: int(args.max_tiles)]
            print(f"[extract] capped {split} at {len(dataset.names)} tiles")

        payload = run_split(
            backbone=backbone,
            mask_values=mask_values,
            dataset=dataset,
            transform=transform,
            batch_size=max(1, int(args.batch_size)),
            device=device,
        )
        save_split_npz(
            output_dir / f"{split}.npz",
            payload,
            {**meta_common, "split": split, "num_tiles": int(payload["category"].shape[0])},
        )

    meta_path = output_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({**meta_common, "splits": list(args.splits)}, f, indent=2)
    print(f"[extract] wrote {meta_path}")


if __name__ == "__main__":
    main()
