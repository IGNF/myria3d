import numpy as np
import pytest
import torch_geometric
import torch

from myria3d.pctl.transforms.transforms import TargetTransform, DropPointsByClass


def test_TargetTransform_with_valid_config():
    # 2 are turned into 1.
    classification_preprocessing_dict = {2: 1}
    # 1 becomes 0, and 6 becomes 1.
    classification_dict = {1: "unclassified", 6: "building"}
    tt = TargetTransform(classification_preprocessing_dict, classification_dict)

    y = np.array([1, 1, 2, 2, 6, 6])
    data = torch_geometric.data.Data(x=None, y=y)
    assert np.array_equal(tt(data).y, np.array([0, 0, 0, 0, 1, 1]))


def test_TargetTransform_throws_type_error_if_invalid_classification_dict():
    classification_preprocessing_dict = {2: 1}
    classification_dict = {1: "unclassified", 2: "ground", 6: "building"}
    tt = TargetTransform(classification_preprocessing_dict, classification_dict)

    invalid_input_data = torch_geometric.data.Data(x=None, y=np.array([1, 1, 1, 2, 99999, 1]))
    with pytest.raises(TypeError):
        # error content:
        # int() argument must be a string, a bytes-like object or a number, not 'NoneType'
        _ = tt(invalid_input_data)


def test_DropPointsByClass():
    # points with class 65 are droped.
    y = torch.Tensor([1, 65, 65, 2, 65])
    x = torch.rand((5, 3))
    data = torch_geometric.data.Data(x=x, y=y)
    drop_transforms = DropPointsByClass()
    transformed_data = drop_transforms(data)
    assert torch.equal(transformed_data.y, torch.Tensor([1, 2]))
    assert transformed_data.x.size(0) == 2

    # No modification
    x = torch.rand((3, 3))
    y = torch.Tensor([1, 2, 3])
    data = torch_geometric.data.Data(x=x, y=y)
    transformed_data = drop_transforms(data)
    assert torch.equal(data.x, transformed_data.x)
    assert torch.equal(data.y, transformed_data.y)
