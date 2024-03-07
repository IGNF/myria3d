import numpy as np
import pytest
import torch
import torch_geometric

from myria3d.pctl.transforms.transforms import DropPointsByClass, TargetTransform


def test_TargetTransform_with_valid_config():
    # 2 are turned into 1.
    classification_preprocessing_dict = {2: 1}
    # 1 becomes 0, and 6 becomes 1.
    classification_dict = {1: "unclassified", 6: "building"}
    tt = TargetTransform(classification_preprocessing_dict, classification_dict)
    y = np.array([1, 1, 2, 2, 6, 6])
    idx = np.arange(6)
    data = torch_geometric.data.Data(x=None, y=y, idx_in_original_cloud=idx)
    out_data = tt(data)
    assert np.array_equal(out_data.y, np.array([0, 0, 0, 0, 1, 1]))
    assert np.array_equal(out_data.idx_in_original_cloud, idx)


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
    idx = np.arange(5)  # Not a tensor
    data = torch_geometric.data.Data(x=x, y=y, idx_in_original_cloud=idx)
    drop_transforms = DropPointsByClass()
    transformed_data = drop_transforms(data)
    assert torch.equal(transformed_data.y, torch.Tensor([1, 2]))
    assert transformed_data.x.size(0) == 2
    print(type(transformed_data.idx_in_original_cloud))
    assert isinstance(transformed_data.idx_in_original_cloud, np.ndarray)
    assert transformed_data.idx_in_original_cloud.size == 2
    assert np.all(transformed_data.idx_in_original_cloud == np.array([0, 3]))

    # No modification
    x = torch.rand((3, 3))
    y = torch.Tensor([1, 2, 3])
    data = torch_geometric.data.Data(x=x, y=y)
    transformed_data = drop_transforms(data)
    assert torch.equal(data.x, transformed_data.x)
    assert torch.equal(data.y, transformed_data.y)

    # Keep one point only
    y = torch.Tensor([1, 65, 65, 65, 65])
    x = torch.rand((5, 3))
    idx = np.arange(5)  # Not a tensor
    data = torch_geometric.data.Data(x=x, y=y, idx_in_original_cloud=idx)
    transformed_data = drop_transforms(data)
    assert torch.equal(transformed_data.y, torch.Tensor([1]))
    assert transformed_data.x.size(0) == 1
    assert isinstance(transformed_data.idx_in_original_cloud, np.ndarray)
    assert transformed_data.idx_in_original_cloud.shape[0] == 1
    assert np.all(transformed_data.idx_in_original_cloud == np.array([0]))
