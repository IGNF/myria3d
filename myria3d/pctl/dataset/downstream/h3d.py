from myria3d.pctl.dataset.downstream.base import DownstreamNpyDataset


class H3dDataset(DownstreamNpyDataset):
    """H3D (Hessigheim 3D, aerial LiDAR semantic segmentation): no intensity channel
    in the Pointcept preprocessing -- ``strength_mask`` is set all-True so the frozen
    backbone's learned ``strength_mask_value`` fill-in is used instead of a naive
    zero (see ``GridProbeModel._fill_masked_features``).
    """

    def __init__(self, data_root: str, split_dir: str, **kwargs):
        kwargs.setdefault("has_color", True)
        kwargs.setdefault("has_strength", False)
        kwargs.setdefault("label_key", "segment")
        super().__init__(data_root, split_dir, **kwargs)
