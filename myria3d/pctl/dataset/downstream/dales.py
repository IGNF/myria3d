from myria3d.pctl.dataset.downstream.base import DownstreamNpyDataset


class DalesDataset(DownstreamNpyDataset):
    """DALES (aerial LiDAR semantic segmentation): no color channel in the Pointcept
    preprocessing -- ``color_mask`` is set all-True so the frozen backbone's learned
    ``color_mask_value`` fill-in is used instead of a naive zero (see
    ``GridProbeModel._fill_masked_features``). DALES also has no val split upstream:
    point ``datamodule.val_dir`` at the same folder as ``test_dir`` (see
    readme_linear_probing.md), a deliberate, documented convention.
    """

    def __init__(self, data_root: str, split_dir: str, **kwargs):
        kwargs.setdefault("has_color", False)
        kwargs.setdefault("has_strength", True)
        kwargs.setdefault("label_key", "segment")
        super().__init__(data_root, split_dir, **kwargs)
