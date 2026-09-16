import json
import os.path as osp
from typing import Optional, Sequence

from myria3d.pctl.dataset.downstream.base import DownstreamNpyDataset, list_scene_dirs

# Confirmed against pointcept/datasets/eclair.py (sibling repo): review_category is an
# allow-list, not a deny-list -- only "approved" scenes survive an
# `include_pseudo=False` filter (the docstring there calls the excluded tiles
# "pseudo (rejected)", but the code checks equality with "approved", not inequality
# with "pseudo"/"rejected" -- any other value, including missing metadata, is dropped).
_APPROVED_REVIEW_CATEGORY = "approved"


def _is_approved(scene_dir: str) -> bool:
    meta_path = osp.join(scene_dir, "meta.json")
    if not osp.isfile(meta_path):
        raise FileNotFoundError(f"Missing meta.json for include_pseudo=False filter: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("review_category") == _APPROVED_REVIEW_CATEGORY


class EclairDataset(DownstreamNpyDataset):
    """ECLAIR: real color + intensity (no learned-mask fill-in needed -- its
    `return_number`/`number_of_returns` npy assets are simply unused, matching
    myria3d's 5-feature convention). Train split optionally filtered down to
    ``meta.json["review_category"] == "approved"`` scenes (mirrors
    ``pointcept/datasets/eclair.py``'s ``include_pseudo`` filter exactly). Val/test
    are already ground-truth-only in the official split, so this filter is only ever
    meaningful on ``split_dir="train"``.
    """

    def __init__(
        self,
        data_root: str,
        split_dir: str,
        include_pseudo: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("has_color", True)
        kwargs.setdefault("has_strength", True)
        kwargs.setdefault("label_key", "segment")
        scene_dirs: Optional[Sequence[str]] = kwargs.pop("scene_dirs", None)
        if scene_dirs is None:
            scene_dirs = list_scene_dirs(data_root, split_dir)
        if split_dir == "train" and not include_pseudo:
            scene_dirs = [d for d in scene_dirs if _is_approved(d)]
        super().__init__(data_root, split_dir, scene_dirs=list(scene_dirs), **kwargs)
