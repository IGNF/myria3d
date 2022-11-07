from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater


class GeometricNoneProofDataloader(DataLoader):
    """Torch geometric's dataloader is a simple torch Dataloader with a different Collater.

    This overrides the collater with a NoneProof one that will not fail if some Data is None.

    """

    def __init__(self, *args, **kwargs) -> None:
        # Allow overriding - this is needed for pytorch lightning "overfit_batches" function, which
        # update the dataloader with a different collater.
        kwargs["collate_fn"] = kwargs.get("collate_fn", GeometricNoneProofCollater())
        super().__init__(*args, **kwargs)


class GeometricNoneProofCollater(Collater):
    """A Collater that returns None when given empty batch."""

    def __init__(self, follow_batch=None, exclude_keys=None):
        super().__init__(follow_batch, exclude_keys)

    def __call__(self, data_list):
        if data_list is None:
            return None
        data_list = [d for d in data_list if d is not None]
        if not data_list:
            # empty
            return None
        return super().__call__(data_list)
