import numpy as np
import pytest
import torch
import torch_geometric

from myria3d.pctl.transforms.transforms import (
    DropPointsByClass,
    MinimumNumNodes,
    TargetTransform,
    subsample_data,
)


@pytest.mark.parametrize(
    "x,idx,choice,nb_out_nodes",
    [
        # Standard use case with choice contiaining indices
        (
            torch.Tensor([10, 11, 12, 13, 14]),
            np.array([20, 21, 22, 23, 24]),
            torch.IntTensor([0, 1, 4]),
            3,
        ),
        # Edge case with choice containing indices: select no point
        (
            torch.Tensor([10, 11, 12, 13, 14]),
            np.array([20, 21, 22, 23, 24]),
            torch.IntTensor([]),
            0,
        ),
        # Edge case with choice containing indices: select one point
        (
            torch.Tensor([10, 11, 12, 13, 14]),
            np.array([20, 21, 22, 23, 24]),
            torch.IntTensor([1]),
            1,
        ),
        # Edge case with choice containing indices: input array with one point
        (
            torch.Tensor([10]),
            np.array([20]),
            torch.IntTensor([0]),
            1,
        ),
        # Edge case with choice containing indices: input array with one point
        (
            torch.Tensor([10]),
            np.array([20]),
            torch.IntTensor([]),
            0,
        ),
        # Standard use case with choice as boolean array
        (
            torch.Tensor([10, 11, 12, 13, 14]),
            np.array([20, 21, 22, 23, 24]),
            torch.BoolTensor([True, True, False, True, False]),
            3,
        ),
        # Edge case with choice as boolean array: select no point
        (
            torch.Tensor([10, 11, 12, 13, 14]),
            np.array([20, 21, 22, 23, 24]),
            torch.BoolTensor([False, False, False, False, False]),
            0,
        ),
        # Edge case with choice as boolean array: select one point
        (
            torch.Tensor([10, 11, 12, 13, 14]),
            np.array([20, 21, 22, 23, 24]),
            torch.BoolTensor([False, True, False, False, False]),
            1,
        ),
        # Edge case with choice as boolean array: input array with one point
        (
            torch.Tensor([10]),
            np.array([20]),
            torch.BoolTensor([True]),
            1,
        ),
        # Edge case with choice as boolean array: input array with one point
        (
            torch.Tensor([10]),
            np.array([20]),
            torch.BoolTensor([False]),
            0,
        ),
    ],
)
def test_subsample_data(x, idx, choice, nb_out_nodes):
    num_nodes = x.size(0)
    data = torch_geometric.data.Data(x=x, idx_in_original_cloud=idx, num_nodes=num_nodes)
    transformed_data = subsample_data(data, num_nodes, choice)
    assert transformed_data.num_nodes == nb_out_nodes
    assert isinstance(transformed_data.x, torch.Tensor)
    assert transformed_data.x.size(0) == nb_out_nodes
    assert isinstance(transformed_data.idx_in_original_cloud, np.ndarray)
    # Check that "idx_in_original_cloud" key is not modified
    assert transformed_data.idx_in_original_cloud.shape[0] == num_nodes


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


@pytest.mark.parametrize("input_nodes,min_nodes", [(5, 10), (1, 10), (15, 10)])
def test_MinimumNumNodes(input_nodes, min_nodes):
    x = torch.rand((input_nodes, 3))
    idx = np.arange(input_nodes)  # Not a tensor
    data = torch_geometric.data.Data(x=x, idx_in_original_cloud=idx)
    transform = MinimumNumNodes(min_nodes)

    transformed_data = transform(data)
    expected_nodes = max(input_nodes, min_nodes)
    assert transformed_data.num_nodes == expected_nodes
    assert isinstance(transformed_data.x, torch.Tensor)
    assert transformed_data.x.size(0) == expected_nodes
    # Check that "idx_in_original_cloud" key is not modified
    assert isinstance(transformed_data.idx_in_original_cloud, np.ndarray)
    assert transformed_data.idx_in_original_cloud.shape[0] == input_nodes
