from typing import Callable, List
from torch_geometric.transforms import BaseTransform


class CustomCompose(BaseTransform):
    """
    Composes several transforms together.
    Edited to bypass downstream transforms if None is returned by a transform.
    Args:
        transforms (List[Callable]): List of transforms to compose.
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, data):
        for transform in self.transforms:
            if isinstance(data, (list, tuple)):
                data = [transform(d) for d in data]
                data = [d for d in data if d is not None]
                if len(data) == 0:
                    return None
            else:
                data = transform(data)
                if data is None:
                    return None
        return data
